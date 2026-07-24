"""Calibration scales for wrap (absmean / percentile; optional per-channel).

Not magic PTQ — improves alpha for binary/ternary vs naive global sign-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

ScaleMethod = Literal["absmean", "percentile"]


@dataclass
class CalibConfig:
    method: ScaleMethod = "absmean"
    percentile: float = 99.0
    per_channel: bool = True
    max_batches: int = 4


def absmean_scale(w: Tensor, *, per_channel: bool = True) -> Tensor:
    """Per-output-channel absmean (or global scalar)."""
    w = w.detach().float()
    if per_channel and w.ndim >= 2:
        # Linear weight: (out, in)
        dims = tuple(range(1, w.ndim))
        return w.abs().mean(dim=dims).clamp(min=1e-8)
    return w.abs().mean().clamp(min=1e-8).reshape(())


def percentile_scale(
    w: Tensor,
    percentile: float = 99.0,
    *,
    per_channel: bool = True,
) -> Tensor:
    w = w.detach().float()
    q = float(percentile) / 100.0
    if per_channel and w.ndim >= 2:
        flat = w.reshape(w.shape[0], -1).abs()
        # torch.quantile along last dim
        return torch.quantile(flat, q, dim=1).clamp(min=1e-8)
    return torch.quantile(w.abs().flatten(), q).clamp(min=1e-8).reshape(())


def scale_from_weight(
    w: Tensor,
    cfg: CalibConfig | None = None,
) -> Tensor:
    cfg = cfg or CalibConfig()
    if cfg.method == "percentile":
        return percentile_scale(w, cfg.percentile, per_channel=cfg.per_channel)
    return absmean_scale(w, per_channel=cfg.per_channel)


def calibrate_linear_scales(
    weight: Tensor,
    *,
    cfg: CalibConfig | None = None,
    activation_batches: list[Tensor] | None = None,
) -> Tensor:
    """Return alpha/scale for a Linear weight.

    If ``activation_batches`` is provided, optionally blend with activation
    absmean (channel-wise on last dim) — still weight-primary for PTQ wrap.
    """
    cfg = cfg or CalibConfig()
    alpha = scale_from_weight(weight, cfg)
    if activation_batches:
        # Light act-aware nudge: geometric mean with act absmean (keeps weight primary)
        acts = torch.cat(
            [a.detach().float().reshape(-1, a.shape[-1]) for a in activation_batches[: cfg.max_batches]],
            dim=0,
        )
        act_s = acts.abs().mean().clamp(min=1e-8)
        if alpha.ndim == 0:
            alpha = (alpha * act_s).sqrt()
        else:
            # Keep per-channel weight scale; mild global act factor
            alpha = alpha * (act_s.sqrt().clamp(0.5, 2.0))
    return alpha
