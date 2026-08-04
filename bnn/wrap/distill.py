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
from ..ste import clip_weights_
from .metrics import measure_agreement
from .qat import _swap_linear_to_binary


@dataclass
class DistillConfig:
    steps: int = 100
    lr: float = 1e-3
    temperature: float = 2.0
    alpha_ce: float = 0.0  # 0 → pure KD; >0 mixes CE when labels provided
    layer_names: list[str] | None = None
    drop_in_threshold: float = 0.85


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

    for name, lin in targets:
        _set_module(student, name, _swap_linear_to_binary(lin))

    student.train()
    opt = torch.optim.Adam(
        [p for p in student.parameters() if p.requires_grad],
        lr=cfg.lr,
    )
    T = float(cfg.temperature)
    last_loss = 0.0
    n_steps = 0
    cursor = 0
    while n_steps < cfg.steps:
        x, y = batch_list[cursor % len(batch_list)]
        cursor += 1
        opt.zero_grad(set_to_none=True)
        s_logits = student(x)
        with torch.no_grad():
            t_logits = teacher(x)
        kd = F.kl_div(
            F.log_softmax(s_logits / T, dim=-1),
            F.softmax(t_logits / T, dim=-1),
            reduction="batchmean",
        ) * (T * T)
        if cfg.alpha_ce > 0 and y is not None:
            ce = F.cross_entropy(s_logits.reshape(-1, s_logits.shape[-1]), y.reshape(-1))
            loss = cfg.alpha_ce * ce + (1.0 - cfg.alpha_ce) * kd
        else:
            loss = kd
        loss.backward()
        opt.step()
        clip_weights_(student)
        last_loss = float(loss.detach().item())
        n_steps += 1

    student.eval()
    restored: list[str] = []
    named = dict(student.named_modules())
    for name, _ in targets:
        bl = named.get(name)
        if not isinstance(bl, BinaryLinear):
            continue
        lin = nn.Linear(bl.in_features, bl.out_features, bias=bl.bias is not None)
        with torch.no_grad():
            lin.weight.copy_(bl.weight)
            if bl.bias is not None and lin.bias is not None:
                lin.bias.copy_(bl.bias)
        _set_module(student, name, lin)
        restored.append(name)

    with torch.no_grad():
        after = measure_agreement(
            teacher(x0), student(x0), drop_in_threshold=cfg.drop_in_threshold
        )

    uplift = float(after.cosine - before.cosine)
    return DistillReport(
        steps=n_steps,
        skipped=False,
        cosine_before=float(before.cosine),
        cosine_after=float(after.cosine),
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
