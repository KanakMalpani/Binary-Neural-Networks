"""Calibration scales for wrap (absmean / percentile; optional per-channel).

Not magic PTQ — improves alpha for binary/ternary vs naive global sign-only.

**Unified entry (W3.T01):** prefer ``calibrate(...)`` which dispatches to
weight-level or model-level APIs. Legacy helpers (``calibrate_linear_scales``,
``absmean_scale``, ``percentile_scale``) remain stable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal, overload

import torch
import torch.nn as nn
from torch import Tensor

from .policy import WrapPolicy, select_linears

ScaleMethod = Literal["absmean", "percentile"]


@dataclass
class CalibConfig:
    method: ScaleMethod = "absmean"
    percentile: float = 99.0
    per_channel: bool = True
    max_batches: int = 4


@dataclass
class LayerScale:
    """Per-Linear calibration scale produced by ``calibrate_model``."""

    name: str
    scale: list[float]
    method: str
    weight_shape: list[int]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CalibReport:
    """Model-level calibration summary (W3.T01 unified report)."""

    method: str
    per_channel: bool
    layers: list[LayerScale] = field(default_factory=list)
    notes: str = (
        "Weight-primary PTQ scales; optional activation nudge stays secondary. "
        "Compression claims remain theoretical pack ratios."
    )

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "per_channel": self.per_channel,
            "layers": [L.to_dict() for L in self.layers],
            "notes": self.notes,
            "n_layers": len(self.layers),
        }

    def scales_by_name(self) -> dict[str, Tensor]:
        return {
            L.name: torch.tensor(L.scale, dtype=torch.float32) for L in self.layers
        }


def absmean_scale(w: Tensor, *, per_channel: bool = True) -> Tensor:
    """Per-output-channel absmean (or global scalar)."""
    w = w.detach().float()
    if per_channel and w.ndim >= 2:
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
    absmean — still weight-primary for PTQ wrap.
    """
    cfg = cfg or CalibConfig()
    alpha = scale_from_weight(weight, cfg)
    if activation_batches:
        acts = torch.cat(
            [a.detach().float().reshape(-1, a.shape[-1]) for a in activation_batches[: cfg.max_batches]],
            dim=0,
        )
        act_s = acts.abs().mean().clamp(min=1e-8)
        if alpha.ndim == 0:
            alpha = (alpha * act_s).sqrt()
        else:
            alpha = alpha * (act_s.sqrt().clamp(0.5, 2.0))
    return alpha


def calibrate_model(
    model: nn.Module,
    *,
    cfg: CalibConfig | None = None,
    policy: WrapPolicy | str = "all_large_linear",
    min_in_features: int = 32,
    min_out_features: int = 0,
    activation_batches: list[Tensor] | None = None,
) -> CalibReport:
    """Compute PTQ scales for every eligible Linear (unified model entry)."""
    cfg = cfg or CalibConfig()
    to_cal, _skipped = select_linears(
        model,
        policy=policy,  # type: ignore[arg-type]
        min_in_features=min_in_features,
        min_out_features=min_out_features,
        skip_attn=True,
    )
    layers: list[LayerScale] = []
    for name, lin in to_cal:
        alpha = calibrate_linear_scales(
            lin.weight.data, cfg=cfg, activation_batches=activation_batches
        )
        flat = alpha.detach().float().reshape(-1).tolist()
        layers.append(
            LayerScale(
                name=name,
                scale=flat,
                method=cfg.method,
                weight_shape=list(lin.weight.shape),
            )
        )
    return CalibReport(
        method=cfg.method,
        per_channel=cfg.per_channel,
        layers=layers,
    )


@overload
def calibrate(
    target: Tensor,
    cfg: CalibConfig | None = None,
    *,
    activation_batches: list[Tensor] | None = None,
) -> Tensor: ...


@overload
def calibrate(
    target: nn.Module,
    cfg: CalibConfig | None = None,
    *,
    activation_batches: list[Tensor] | None = None,
    policy: WrapPolicy | str = "all_large_linear",
    min_in_features: int = 32,
    min_out_features: int = 0,
) -> CalibReport: ...


def calibrate(
    target: Tensor | nn.Module,
    cfg: CalibConfig | None = None,
    *,
    activation_batches: list[Tensor] | None = None,
    policy: WrapPolicy | str = "all_large_linear",
    min_in_features: int = 32,
    min_out_features: int = 0,
) -> Tensor | CalibReport:
    """Unified calibrate entrypoint (W3.T01).

    * ``Tensor`` → ``calibrate_linear_scales``
    * ``nn.Module`` → ``calibrate_model``
    """
    if isinstance(target, Tensor):
        return calibrate_linear_scales(
            target, cfg=cfg, activation_batches=activation_batches
        )
    if isinstance(target, nn.Module):
        return calibrate_model(
            target,
            cfg=cfg,
            policy=policy,
            min_in_features=min_in_features,
            min_out_features=min_out_features,
            activation_batches=activation_batches,
        )
    raise TypeError(f"calibrate expects Tensor or nn.Module, got {type(target)}")
