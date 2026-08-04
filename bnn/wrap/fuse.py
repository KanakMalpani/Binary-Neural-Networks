"""BN fusion helpers for the optimiser / wrap path (W3.T09).

Folds BatchNorm into preceding Linear / BiReal conv scales for **eval**
throughput. Training STE paths should leave BN unfused.

Thesis lock: fusion is an inference optimisation; it does not invent GPU 32×
from ``sign()``. Compression ratios stay theoretical pack figures.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import torch
import torch.nn as nn

from ..layers import BiRealBlock, fuse_binary_conv_bn_, fuse_bireal_bn_


@dataclass
class FuseReport:
    """What was fused (and what was skipped) during optimiser prep."""

    linear_bn_pairs: list[str] = field(default_factory=list)
    bireal_blocks: int = 0
    skipped: list[str] = field(default_factory=list)
    notes: str = "Eval-only BN fold; safe after running_stats are populated."

    def to_dict(self) -> dict:
        return asdict(self)


@torch.no_grad()
def fuse_linear_bn1d_(linear: nn.Linear, bn: nn.BatchNorm1d) -> nn.Linear:
    """Fold ``BatchNorm1d`` into a preceding ``nn.Linear`` (eval).

    ``y = BN(Wx + b)`` becomes a single Linear with updated weight/bias.
    Requires populated ``running_mean`` / ``running_var``.
    """
    if not isinstance(linear, nn.Linear) or not isinstance(bn, nn.BatchNorm1d):
        raise TypeError("fuse_linear_bn1d_ expects (nn.Linear, nn.BatchNorm1d)")
    if bn.running_mean is None or bn.running_var is None:
        raise RuntimeError("BN has no running stats; run a few batches in train/eval first")
    if bn.num_features != linear.out_features:
        raise ValueError(
            f"BN features {bn.num_features} != Linear out_features {linear.out_features}"
        )

    eps = float(bn.eps)
    std = torch.sqrt(bn.running_var + eps)
    scale = bn.weight / std  # (out,)

    linear.weight.mul_(scale.unsqueeze(1))
    bias = bn.bias - scale * bn.running_mean
    if linear.bias is None:
        linear.register_parameter("bias", nn.Parameter(bias.clone()))
    else:
        linear.bias.mul_(scale).add_(bias)

    # Identity BN so accidental re-apply is harmless
    bn.weight.fill_(1)
    bn.bias.zero_()
    bn.running_mean.zero_()
    bn.running_var.fill_(1)
    return linear


def _fuse_ordered_pairs(
    names: list[str],
    modules: list[nn.Module],
) -> list[str]:
    """Fuse consecutive Linear→BN1d in a flat ordered list; return Linear names."""
    fused: list[str] = []
    i = 0
    while i < len(modules) - 1:
        a, b = modules[i], modules[i + 1]
        if isinstance(a, nn.Linear) and isinstance(b, nn.BatchNorm1d):
            try:
                fuse_linear_bn1d_(a, b)
                fused.append(names[i])
            except (ValueError, RuntimeError):
                pass
            i += 2
        else:
            i += 1
    return fused


@torch.no_grad()
def fuse_sequential_linear_bn_(root: nn.Module, prefix: str = "") -> list[str]:
    """Fuse adjacent ``Linear → BatchNorm1d`` pairs under ``root``.

    Walks each container's **direct children in order** (Sequential or named),
    then recurses into submodules. Returns dotted names of fused Linears.
    """
    fused: list[str] = []
    children = list(root.named_children())
    if len(children) >= 2:
        names = [f"{prefix}.{n}" if prefix else n for n, _ in children]
        mods = [m for _, m in children]
        fused.extend(_fuse_ordered_pairs(names, mods))

    for name, child in children:
        if list(child.children()):
            child_prefix = f"{prefix}.{name}" if prefix else name
            fused.extend(fuse_sequential_linear_bn_(child, prefix=child_prefix))

    seen: set[str] = set()
    out: list[str] = []
    for n in fused:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


@torch.no_grad()
def fuse_bn_for_wrap_(
    model: nn.Module,
    *,
    bireal: bool = True,
    linear_bn: bool = True,
) -> FuseReport:
    """Optimiser-path BN fuse: Linear+BN1d and/or BiRealBlock BN→alpha.

    Call after a short calib pass so BN running stats exist, **before**
    packing Linears. Idempotent for BiReal (flag); Linear+BN pairs become
    identity BN after the first fold.
    """
    report = FuseReport()
    model.eval()

    if linear_bn:
        try:
            report.linear_bn_pairs = fuse_sequential_linear_bn_(model)
        except Exception as exc:  # noqa: BLE001 — surface as skip notes
            report.skipped.append(f"linear_bn: {exc}")

    if bireal:
        before = sum(
            1 for m in model.modules() if isinstance(m, BiRealBlock) and m._bn_fused
        )
        fuse_bireal_bn_(model)
        after = sum(
            1 for m in model.modules() if isinstance(m, BiRealBlock) and m._bn_fused
        )
        report.bireal_blocks = after - before

    return report


__all__ = [
    "FuseReport",
    "fuse_bn_for_wrap_",
    "fuse_linear_bn1d_",
    "fuse_sequential_linear_bn_",
    "fuse_binary_conv_bn_",
    "fuse_bireal_bn_",
]
