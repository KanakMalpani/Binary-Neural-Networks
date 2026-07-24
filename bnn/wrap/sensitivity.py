"""Layer-wise sensitivity scoring for wrap decisions (W3.T05).

Ablates one Linear at a time and ranks by cosine drop vs FP teacher.
Fragile layers are suggested for skip — honesty over aggressive pack rates.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import torch
import torch.nn as nn
from torch import Tensor

from .calibrate import CalibConfig
from .metrics import measure_agreement
from .packed_linear import PackedBinaryXNORLinear, TernaryWeightOnlyLinear
from .policy import select_linears

ScoreMode = Literal["binary_xnor", "ternary_weight_only"]


@dataclass
class LayerSensitivity:
    name: str
    in_features: int
    out_features: int
    cosine: float
    cosine_drop: float
    top1_agreement: float | None
    recommended: ScoreMode | Literal["skip"]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SensitivityReport:
    baseline_cosine: float
    layers: list[LayerSensitivity] = field(default_factory=list)
    skip_suggested: list[str] = field(default_factory=list)
    wrap_suggested: list[str] = field(default_factory=list)
    drop_in_threshold: float = 0.85
    mode_scored: ScoreMode = "binary_xnor"

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_cosine": self.baseline_cosine,
            "layers": [L.to_dict() for L in self.layers],
            "skip_suggested": list(self.skip_suggested),
            "wrap_suggested": list(self.wrap_suggested),
            "drop_in_threshold": self.drop_in_threshold,
            "mode_scored": self.mode_scored,
            "thesis_note": (
                "Sensitivity ranks accuracy risk of wrapping each layer; "
                "compression remains theoretical — dual-metric."
            ),
        }


def _set_module(root: nn.Module, path: str, new: nn.Module) -> None:
    parts = path.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new)


def _wrap_one_linear(
    lin: nn.Linear,
    mode: ScoreMode,
    *,
    calib: CalibConfig | None,
) -> nn.Module:
    w = lin.weight.data
    b = lin.bias.data if lin.bias is not None else None
    if mode == "binary_xnor":
        return PackedBinaryXNORLinear(w, b, calib=calib)
    if mode == "ternary_weight_only":
        per_ch = True if calib is None else calib.per_channel
        return TernaryWeightOnlyLinear(w, b, per_channel=per_ch, calib=calib)
    raise ValueError(mode)


def score_layer_sensitivity(
    model: nn.Module,
    calib_inputs: Tensor,
    *,
    mode: ScoreMode = "binary_xnor",
    policy: str = "all_large_linear",
    min_in_features: int = 32,
    min_out_features: int = 0,
    drop_in_threshold: float = 0.85,
    skip_fragile: bool = True,
    fragile_drop: float = 0.05,
    calib: CalibConfig | None = None,
) -> SensitivityReport:
    """Score each eligible Linear by cosine drop when wrapped alone.

    ``fragile_drop``: if ``baseline_cosine - layer_cosine >= fragile_drop``,
    suggest skip (also if cosine falls below ``drop_in_threshold``).
    """
    teacher = copy.deepcopy(model)
    teacher.eval()
    model = model.eval()

    with torch.no_grad():
        t_logits = teacher(calib_inputs)
        base_logits = model(calib_inputs)
    base = measure_agreement(t_logits, base_logits, drop_in_threshold=drop_in_threshold)
    baseline_cos = float(base.cosine)

    candidates, _skipped = select_linears(
        model,
        policy=policy,  # type: ignore[arg-type]
        min_in_features=min_in_features,
        min_out_features=min_out_features,
        skip_attn=True,
    )

    layers: list[LayerSensitivity] = []
    skip_suggested: list[str] = []
    wrap_suggested: list[str] = []

    for name, lin in candidates:
        probe = copy.deepcopy(model)
        _set_module(probe, name, _wrap_one_linear(lin, mode, calib=calib))
        probe.eval()
        with torch.no_grad():
            s_logits = probe(calib_inputs)
        eff = measure_agreement(t_logits, s_logits, drop_in_threshold=drop_in_threshold)
        drop = max(0.0, baseline_cos - float(eff.cosine))
        fragile = drop >= fragile_drop or float(eff.cosine) < drop_in_threshold
        if fragile and skip_fragile:
            rec: ScoreMode | Literal["skip"] = "skip"
            reason = f"fragile: cosine={eff.cosine:.4f} drop={drop:.4f}"
            skip_suggested.append(name)
        else:
            rec = mode
            reason = f"ok: cosine={eff.cosine:.4f} drop={drop:.4f}"
            wrap_suggested.append(name)
        layers.append(
            LayerSensitivity(
                name=name,
                in_features=int(lin.in_features),
                out_features=int(lin.out_features),
                cosine=float(eff.cosine),
                cosine_drop=float(drop),
                top1_agreement=eff.top1_agreement,
                recommended=rec,
                reason=reason,
            )
        )

    layers.sort(key=lambda L: L.cosine_drop, reverse=True)
    return SensitivityReport(
        baseline_cosine=baseline_cos,
        layers=layers,
        skip_suggested=skip_suggested,
        wrap_suggested=wrap_suggested,
        drop_in_threshold=drop_in_threshold,
        mode_scored=mode,
    )


def apply_sensitivity_skips(
    skip_name_substr: list[str] | None,
    sensitivity: SensitivityReport,
) -> list[str]:
    """Deprecated helper — prefer ``exclude_exact=sensitivity.skip_suggested``.

    Kept for API stability; merges suggested names into a substr list (exact
    names are safer via ``wrap_model(..., exclude_exact=...)``).
    """
    import warnings

    warnings.warn(
        "apply_sensitivity_skips merges into substr skips; prefer "
        "wrap_model(..., exclude_exact=report.skip_suggested)",
        DeprecationWarning,
        stacklevel=2,
    )
    base = list(skip_name_substr or [])
    for name in sensitivity.skip_suggested:
        if name not in base:
            base.append(name)
    return base
