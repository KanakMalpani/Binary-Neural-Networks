"""Light STE/QAT recovery after wrap (few steps — not full BitDistill).

Limits: recovers cosine/accuracy only modestly on toy stacks; production HF
models need real data + longer distillation (see docs/12, docs/33).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..layers import BinaryLinear
from ..ste import SignMode, clip_weights_, get_binary_sign_fn, get_sign_mode, set_sign_mode

LogitLoss = Literal["kd", "mse", "cosine"]

_FFN_KEYS = ("ffn", "mlp", "fc1", "fc2", "intermediate")


class WeightOnlySTELinear(nn.Module):
    """STE on weights only; full-precision activations.

    Packed ``binary_xnor`` wrap still signs activations at inference. This
    module is a QAT experiment (train/infer mismatch) — not a new kernel.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.weight._bnn_clip = True  # type: ignore[attr-defined]
        self.alpha = nn.Parameter(torch.ones(out_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)
        nn.init.xavier_uniform_(self.weight)
        with torch.no_grad():
            self.weight.clamp_(-1, 1)
            self.alpha.fill_(self.weight.abs().mean().clamp(min=1e-4).item())

    def forward(self, x: Tensor) -> Tensor:
        sign = get_binary_sign_fn()
        w_b = sign(self.weight) * self.alpha.unsqueeze(1)
        return F.linear(x, w_b, self.bias)


SteLinear = BinaryLinear | WeightOnlySTELinear


def _set_module(root: nn.Module, path: str, new: nn.Module) -> None:
    parts = path.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new)


def _collect_target_linears(
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
        elif not any(k in lname for k in _FFN_KEYS):
            continue
        targets.append((name, mod))
    return targets


def _init_alpha_from_weight_(ste: SteLinear, weight: Tensor) -> None:
    """Per-out-channel absmean — matches wrap calib, not leftover Xavier."""
    with torch.no_grad():
        ste.alpha.copy_(weight.detach().abs().mean(dim=1).clamp(min=1e-4))


def _swap_linear_to_binary(
    lin: nn.Linear, *, binarize_activations: bool = True
) -> SteLinear:
    """Replace ``nn.Linear`` with an STE module on the same device/dtype."""
    cls: type[SteLinear] = BinaryLinear if binarize_activations else WeightOnlySTELinear
    ste = cls(
        lin.in_features, lin.out_features, bias=lin.bias is not None
    ).to(device=lin.weight.device, dtype=lin.weight.dtype)
    with torch.no_grad():
        ste.weight.copy_(lin.weight)
        _init_alpha_from_weight_(ste, lin.weight)
        if lin.bias is not None and ste.bias is not None:
            ste.bias.copy_(lin.bias)
    return ste


def _restore_binary_to_linear(
    bl: SteLinear, *, fold_alpha: bool = False
) -> nn.Linear:
    """Restore packable ``nn.Linear`` from STE (same device/dtype).

    ``fold_alpha=True`` bakes the learned per-out-channel scale into latent
    magnitudes so wrap absmean calib recovers STE ``alpha`` (sign bits kept).
    Default ``False`` is a lossless round-trip of latent weights (unit tests).
    """
    lin = nn.Linear(
        bl.in_features, bl.out_features, bias=bl.bias is not None
    ).to(device=bl.weight.device, dtype=bl.weight.dtype)
    with torch.no_grad():
        if fold_alpha:
            signs = bl.weight.detach().ge(0).to(bl.weight.dtype).mul(2).sub(1)
            lin.weight.copy_(signs * bl.alpha.detach().reshape(-1, 1))
        else:
            lin.weight.copy_(bl.weight)
        if bl.bias is not None and lin.bias is not None:
            lin.bias.copy_(bl.bias)
    return lin


def agreement_loss(
    student_out: Tensor,
    teacher_out: Tensor,
    *,
    kind: LogitLoss = "kd",
    temperature: float = 2.0,
) -> Tensor:
    """Logit-space loss for wrap recovery. ``cosine`` matches the drop-in metric."""
    if kind == "mse":
        return F.mse_loss(student_out, teacher_out)
    if kind == "cosine":
        s = student_out.reshape(student_out.shape[0], -1)
        tgt = teacher_out.reshape(teacher_out.shape[0], -1)
        return (1.0 - F.cosine_similarity(s, tgt, dim=1)).mean()
    tau = float(temperature)
    return F.kl_div(
        F.log_softmax(student_out / tau, dim=-1),
        F.softmax(teacher_out / tau, dim=-1),
        reduction="batchmean",
    ) * (tau * tau)


@contextmanager
def temporary_sign_mode(mode: SignMode | None) -> Iterator[None]:
    if mode is None:
        yield
        return
    prev = get_sign_mode()
    set_sign_mode(mode)
    try:
        yield
    finally:
        set_sign_mode(prev)


def _hidden_hooks(
    model: nn.Module, names: list[str]
) -> tuple[dict[str, Tensor], list[torch.utils.hooks.RemovableHandle]]:
    cache: dict[str, Tensor] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []
    mods = dict(model.named_modules())
    for n in names:
        if n not in mods:
            continue

        def _hook(_mod: nn.Module, _inp: tuple, out: Tensor, key: str = n) -> None:
            cache[key] = out

        handles.append(mods[n].register_forward_hook(_hook))
    return cache, handles


def light_qat_recover(
    model: nn.Module,
    calib_x: torch.Tensor,
    *,
    teacher: nn.Module | None = None,
    steps: int = 50,
    lr: float = 1e-3,
    layer_names: list[str] | None = None,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    logit_loss: LogitLoss = "kd",
    temperature: float = 2.0,
    fold_alpha: bool = True,
    train_targets_only: bool = False,
    hidden_mse: float = 0.0,
    binarize_activations: bool = True,
    sign_mode: SignMode | None = None,
) -> dict:
    """Short STE fine-tune on named Linears (default: modules named *ffn* / *mlp*).

    If ``teacher`` is given, distill via ``logit_loss`` (kd / mse / cosine);
    else requires ``loss_fn``. Learned STE ``alpha`` is folded into restored
    Linear magnitudes by default so wrap calib matches QAT.
    """
    if steps <= 0:
        return {"steps": 0, "skipped": True}

    if teacher is None and loss_fn is None:
        raise ValueError(
            "light_qat_recover requires teacher=... or loss_fn=... "
            "(self-argmax CE fallback removed — it was a no-op / harmful)"
        )

    targets = _collect_target_linears(model, layer_names)
    if not targets:
        return {"steps": 0, "skipped": True, "reason": "no target Linears"}

    target_names = [n for n, _ in targets]
    for name, lin in targets:
        _set_module(
            model, name, _swap_linear_to_binary(lin, binarize_activations=binarize_activations)
        )

    model.train()
    if teacher is not None:
        teacher.eval()

    if train_targets_only:
        name_set = set(target_names)
        params = []
        for n, p in model.named_parameters():
            owner = n.rsplit(".", 1)[0] if "." in n else n
            if owner in name_set or n in name_set:
                params.append(p)
            else:
                p.requires_grad_(False)
        opt = torch.optim.Adam(params, lr=lr)
    else:
        opt = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr,
        )

    s_cache: dict[str, Tensor] = {}
    t_cache: dict[str, Tensor] = {}
    s_hooks: list[torch.utils.hooks.RemovableHandle] = []
    t_hooks: list[torch.utils.hooks.RemovableHandle] = []
    if hidden_mse > 0 and teacher is not None:
        s_cache, s_hooks = _hidden_hooks(model, target_names)
        t_cache, t_hooks = _hidden_hooks(teacher, target_names)

    last_loss = 0.0
    try:
        with temporary_sign_mode(sign_mode):
            for _ in range(steps):
                opt.zero_grad(set_to_none=True)
                student_out = model(calib_x)
                if loss_fn is not None:
                    loss = loss_fn(student_out, calib_x)
                elif teacher is not None:
                    with torch.no_grad():
                        t_out = teacher(calib_x)
                    loss = agreement_loss(
                        student_out, t_out, kind=logit_loss, temperature=temperature
                    )
                    if hidden_mse > 0 and s_cache and t_cache:
                        hid = torch.zeros((), device=student_out.device, dtype=student_out.dtype)
                        n_h = 0
                        for key in target_names:
                            if key in s_cache and key in t_cache:
                                hid = hid + F.mse_loss(s_cache[key], t_cache[key].detach())
                                n_h += 1
                        if n_h:
                            loss = loss + hidden_mse * (hid / n_h)
                else:
                    raise ValueError("unreachable: teacher/loss_fn required")

                loss.backward()
                opt.step()
                clip_weights_(model)
                last_loss = float(loss.detach().item())
    finally:
        for h in s_hooks + t_hooks:
            h.remove()
        if train_targets_only:
            for p in model.parameters():
                p.requires_grad_(True)

    model.eval()
    restored: list[str] = []
    named = dict(model.named_modules())
    for name, _ in targets:
        ste = named.get(name)
        if not isinstance(ste, (BinaryLinear, WeightOnlySTELinear)):
            continue
        _set_module(model, name, _restore_binary_to_linear(ste, fold_alpha=fold_alpha))
        restored.append(name)

    return {
        "steps": steps,
        "skipped": False,
        "last_loss": last_loss,
        "restored_linears": restored,
        "logit_loss": logit_loss if loss_fn is None else "custom",
        "fold_alpha": fold_alpha,
        "binarize_activations": binarize_activations,
        "sign_mode": sign_mode or get_sign_mode(),
        "note": "Light STE only; production needs BitDistill-scale QAT on real data",
    }
