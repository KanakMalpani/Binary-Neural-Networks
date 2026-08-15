"""Real distillation path for wrap / WC-O (W3.T08).

Goes beyond ``scripts/distill_sketch.py``: multi-batch KD, optional CE mix,
STE on targeted FFN Linears, restore-to-Linear for packing, and measured
cosine before/after vs a frozen FP teacher.

Not BitDistill-scale. Thesis lock: accuracy tools only — never report
compression as wall-clock speedup or GPU 32× from ``sign()``.
"""

from __future__ import annotations

import copy
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..layers import BinaryLinear
from ..ste import SignMode, clip_weights_
from .metrics import measure_agreement
from .qat import (
    LogitLoss,
    WeightOnlySTELinear,
    _restore_binary_to_linear,
    _swap_linear_to_binary,
    agreement_loss,
    temporary_sign_mode,
)


@dataclass
class DistillConfig:
    steps: int = 100
    lr: float = 1e-3
    temperature: float = 2.0
    alpha_ce: float = 0.0  # 0 → pure KD; >0 mixes CE when labels provided
    layer_names: list[str] | None = None
    drop_in_threshold: float = 0.85
    logit_loss: LogitLoss = "kd"
    fold_alpha: bool = True
    train_targets_only: bool = False
    hidden_mse: float = 0.0
    binarize_activations: bool = True
    sign_mode: SignMode | None = None


@dataclass
class DistillReport:
    """Honest before/after agreement + training summary."""

    steps: int
    skipped: bool
    cosine_before: float | None = None
    cosine_after: float | None = None
    cosine_uplift: float | None = None
    last_loss: float | None = None
    restored_linears: list[str] = field(default_factory=list)
    drop_in_ok_before: bool | None = None
    drop_in_ok_after: bool | None = None
    notes: str = (
        "Multi-batch STE KD into BinaryLinear FFN targets; restored to nn.Linear "
        "for packing. Toy/demo scale — not BitDistill. Dual-metric: cosine is "
        "measured; compression stays theoretical elsewhere."
    )
    reason: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _as_batch_list(
    batches: Sequence[Tensor] | Tensor | Iterator[tuple[Tensor, Tensor | None]],
    labels: Sequence[Tensor] | Tensor | None,
) -> list[tuple[Tensor, Tensor | None]]:
    """Normalise inputs into a non-empty list of (x, y|None) pairs."""
    if isinstance(batches, Tensor):
        y = labels if isinstance(labels, Tensor) else None
        return [(batches, y)]

    if isinstance(batches, Sequence):
        out: list[tuple[Tensor, Tensor | None]] = []
        if labels is not None and isinstance(labels, Sequence) and not isinstance(
            labels, Tensor
        ):
            for x, y in zip(batches, labels, strict=True):
                out.append((x, y))
            return out
        for x in batches:
            if isinstance(x, tuple):
                out.append((x[0], x[1] if len(x) > 1 else None))
            else:
                out.append((x, None))
        return out

    out = []
    for item in batches:
        if isinstance(item, tuple):
            out.append((item[0], item[1] if len(item) > 1 else None))
        else:
            out.append((item, None))
    return out


def _collect_targets(
    model: nn.Module, layer_names: list[str] | None
) -> list[tuple[str, nn.Linear]]:
    targets: list[tuple[str, nn.Linear]] = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        lname = name.lower()
        if layer_names is not None:
            if name not in layer_names:
                continue
        elif not any(k in lname for k in ("ffn", "mlp", "fc1", "fc2", "intermediate")):
            continue
        targets.append((name, mod))
    return targets


def _set_module(root: nn.Module, path: str, new: nn.Module) -> None:
    parts = path.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new)


def distill_binary_student(
    student: nn.Module,
    teacher: nn.Module,
    batches: Sequence[Tensor] | Tensor | Iterator[tuple[Tensor, Tensor | None]],
    *,
    cfg: DistillConfig | None = None,
    labels: Sequence[Tensor] | Tensor | None = None,
) -> DistillReport:
    """KD + optional CE into STE BinaryLinear FFN layers, then restore Linear.

    Measures cosine vs ``teacher`` on the first batch before and after training
    so callers can report honest uplift vs cold PTQ (WC-O4 / W3.T08).
    """
    cfg = cfg or DistillConfig()
    if cfg.steps <= 0:
        return DistillReport(steps=0, skipped=True, reason="steps<=0")

    targets = _collect_targets(student, cfg.layer_names)
    if not targets:
        return DistillReport(steps=0, skipped=True, reason="no target Linears")

    batch_list = _as_batch_list(batches, labels)
    if not batch_list:
        return DistillReport(steps=0, skipped=True, reason="empty batches")

    x0, _y0 = batch_list[0]
    teacher = teacher.eval()
    student.eval()
    with torch.no_grad():
        t0 = teacher(x0)
        s0 = student(x0)
        before = measure_agreement(t0, s0, drop_in_threshold=cfg.drop_in_threshold)

    target_names = [n for n, _ in targets]
    for name, lin in targets:
        _set_module(
            student,
            name,
            _swap_linear_to_binary(lin, binarize_activations=cfg.binarize_activations),
        )

    student.train()
    if cfg.train_targets_only:
        name_set = set(target_names)
        params = []
        for n, p in student.named_parameters():
            owner = n.rsplit(".", 1)[0] if "." in n else n
            if owner in name_set or n in name_set:
                params.append(p)
            else:
                p.requires_grad_(False)
        opt = torch.optim.Adam(params, lr=cfg.lr)
    else:
        opt = torch.optim.Adam(
            [p for p in student.parameters() if p.requires_grad],
            lr=cfg.lr,
        )
    last_loss = 0.0
    n_steps = 0
    cursor = 0
    s_hooks: list = []
    t_hooks: list = []
    s_cache: dict = {}
    t_cache: dict = {}
    if cfg.hidden_mse > 0:
        from .qat import _hidden_hooks

        s_cache, s_hooks = _hidden_hooks(student, target_names)
        t_cache, t_hooks = _hidden_hooks(teacher, target_names)
    try:
        with temporary_sign_mode(cfg.sign_mode):
            while n_steps < cfg.steps:
                x, y = batch_list[cursor % len(batch_list)]
                cursor += 1
                opt.zero_grad(set_to_none=True)
                s_logits = student(x)
                with torch.no_grad():
                    t_logits = teacher(x)
                loss = agreement_loss(
                    s_logits,
                    t_logits,
                    kind=cfg.logit_loss,
                    temperature=cfg.temperature,
                )
                if cfg.alpha_ce > 0 and y is not None:
                    ce = F.cross_entropy(
                        s_logits.reshape(-1, s_logits.shape[-1]), y.reshape(-1)
                    )
                    loss = cfg.alpha_ce * ce + (1.0 - cfg.alpha_ce) * loss
                if cfg.hidden_mse > 0 and s_cache and t_cache:
                    hid = torch.zeros((), device=s_logits.device, dtype=s_logits.dtype)
                    n_h = 0
                    for key in target_names:
                        if key in s_cache and key in t_cache:
                            hid = hid + F.mse_loss(s_cache[key], t_cache[key].detach())
                            n_h += 1
                    if n_h:
                        loss = loss + cfg.hidden_mse * (hid / n_h)
                loss.backward()
                opt.step()
                clip_weights_(student)
                last_loss = float(loss.detach().item())
                n_steps += 1
    finally:
        for h in s_hooks + t_hooks:
            h.remove()
        if cfg.train_targets_only:
            for p in student.parameters():
                p.requires_grad_(True)

    student.eval()
    restored: list[str] = []
    named = dict(student.named_modules())
    for name, _ in targets:
        bl = named.get(name)
        if not isinstance(bl, (BinaryLinear, WeightOnlySTELinear)):
            continue
        _set_module(
            student, name, _restore_binary_to_linear(bl, fold_alpha=cfg.fold_alpha)
        )
        restored.append(name)

    with torch.no_grad():
        after = measure_agreement(
            teacher(x0), student(x0), drop_in_threshold=cfg.drop_in_threshold
        )

    # measure_agreement always returns finite measured cosine; narrow for mypy.
    assert before.measured and after.measured
    cos_before = float(before.cosine)
    cos_after = float(after.cosine)
    uplift = cos_after - cos_before
    return DistillReport(
        steps=n_steps,
        skipped=False,
        cosine_before=cos_before,
        cosine_after=cos_after,
        cosine_uplift=uplift,
        last_loss=last_loss,
        restored_linears=restored,
        drop_in_ok_before=bool(before.drop_in_ok),
        drop_in_ok_after=bool(after.drop_in_ok),
    )


def distill_from_teacher_copy(
    model: nn.Module,
    batches: Sequence[Tensor] | Tensor,
    *,
    cfg: DistillConfig | None = None,
) -> tuple[nn.Module, DistillReport]:
    """Convenience: deepcopy ``model`` as teacher, distill student in-place."""
    teacher = copy.deepcopy(model).eval()
    report = distill_binary_student(model, teacher, batches, cfg=cfg)
    return model, report
