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


LayerMode = Literal["binary_xnor", "ternary_weight_only", "skip"]

# Theoretical packed bytes per weight element, by mode. Compression stays
# theoretical (dual-metric rule) — these drive the search objective only.
_BYTES_PER_ELEM: dict[LayerMode, float] = {
    "binary_xnor": 1.0 / 8.0,
    "ternary_weight_only": 2.0 / 8.0,
    "skip": 4.0,
}


@dataclass
class SearchAssignment:
    """Chosen mode for one Linear, with the evidence behind the choice."""

    name: str
    mode: LayerMode
    cosine: float
    weight_elems: int
    packed_bytes: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ModeSearchReport:
    """Result of the per-layer binary / ternary / skip search (W3.T06)."""

    baseline_cosine: float
    final_cosine: float
    quality_floor: float
    met_floor: bool
    assignments: list[SearchAssignment] = field(default_factory=list)
    probes: int = 0

    @property
    def binary(self) -> list[str]:
        return [a.name for a in self.assignments if a.mode == "binary_xnor"]

    @property
    def ternary(self) -> list[str]:
        return [a.name for a in self.assignments if a.mode == "ternary_weight_only"]

    @property
    def skipped(self) -> list[str]:
        return [a.name for a in self.assignments if a.mode == "skip"]

    def compression(self) -> float:
        """Theoretical weight compression over the searched layers only."""
        elems = sum(a.weight_elems for a in self.assignments)
        packed = sum(a.packed_bytes for a in self.assignments)
        if not elems or packed <= 0:
            return 1.0
        return (elems * 4.0) / packed

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_cosine": self.baseline_cosine,
            "final_cosine": self.final_cosine,
            "quality_floor": self.quality_floor,
            "met_floor": self.met_floor,
            "probes": self.probes,
            "binary": self.binary,
            "ternary": self.ternary,
            "skipped": self.skipped,
            "theoretical_compression_searched_layers": self.compression(),
            "assignments": [a.to_dict() for a in self.assignments],
            "thesis_note": (
                "Compression is a theoretical pack ratio; cosine is measured. "
                "Never report compression as an end-to-end speedup."
            ),
        }


def search_layer_modes(
    model: nn.Module,
    calib_inputs: Tensor,
    *,
    quality_floor: float = 0.90,
    policy: str = "all_large_linear",
    min_in_features: int = 32,
    min_out_features: int = 0,
    calib: CalibConfig | None = None,
    max_relaxations: int | None = None,
) -> ModeSearchReport:
    """Pick binary / ternary / skip **per layer** to maximise theoretical
    compression while keeping measured output cosine at or above
    ``quality_floor`` (W3.T06).

    Strategy: start from the most aggressive assignment (everything binary),
    then repeatedly relax the single layer that is costing the most quality —
    binary → ternary → skip — remeasuring the *whole* model each time. Relaxing
    greedily by measured damage is what makes this better than a per-layer
    threshold: layer interactions only show up in the joint measurement.

    Cost is ``O(L)`` probes in the common case rather than the ``3**L`` of an
    exhaustive search, which is why it stays usable on real stacks.
    """
    teacher = copy.deepcopy(model).eval()
    base_model = model.eval()

    with torch.no_grad():
        t_logits = teacher(calib_inputs)
        baseline_cos = float(
            measure_agreement(t_logits, base_model(calib_inputs)).cosine
        )

    candidates, _skipped = select_linears(
        base_model,
        policy=policy,  # type: ignore[arg-type]
        min_in_features=min_in_features,
        min_out_features=min_out_features,
        skip_attn=True,
    )
    if not candidates:
        return ModeSearchReport(
            baseline_cosine=baseline_cos,
            final_cosine=baseline_cos,
            quality_floor=quality_floor,
            met_floor=baseline_cos >= quality_floor,
        )

    elems = {name: int(lin.weight.numel()) for name, lin in candidates}
    order: list[str] = [name for name, _ in candidates]
    chosen: dict[str, LayerMode] = dict.fromkeys(order, "binary_xnor")
    reasons: dict[str, str] = {}
    probes = 0

    def build(assignment: dict[str, LayerMode]) -> nn.Module:
        probe = copy.deepcopy(base_model)
        for name, lin in candidates:
            mode = assignment[name]
            if mode == "skip":
                continue
            _set_module(probe, name, _wrap_one_linear(lin, mode, calib=calib))
        return probe.eval()

    def cosine_of(assignment: dict[str, LayerMode]) -> float:
        nonlocal probes
        probes += 1
        with torch.no_grad():
            return float(measure_agreement(t_logits, build(assignment)(calib_inputs)).cosine)

    current = cosine_of(chosen)
    # Each layer can be relaxed at most twice (binary→ternary→skip).
    budget = 2 * len(order) if max_relaxations is None else max_relaxations

    while current < quality_floor and budget > 0:
        # Which single relaxation buys the most quality right now?
        best_gain, best_name, best_mode, best_cos = 0.0, None, None, current
        for name in order:
            nxt = _relax(chosen[name])
            if nxt is None:
                continue
            trial = dict(chosen)
            trial[name] = nxt
            cos = cosine_of(trial)
            if cos - current > best_gain:
                best_gain, best_name, best_mode, best_cos = cos - current, name, nxt, cos
        if best_name is None or best_mode is None or best_gain <= 0.0:
            break  # nothing left that helps — report honestly below
        reasons[best_name] = (
            f"relaxed to {best_mode}: cosine {current:.4f} -> {best_cos:.4f}"
        )
        chosen[best_name] = best_mode
        current = best_cos
        budget -= 1

    assignments = [
        SearchAssignment(
            name=name,
            mode=chosen[name],
            cosine=current,
            weight_elems=elems[name],
            packed_bytes=elems[name] * _BYTES_PER_ELEM[chosen[name]],
            reason=reasons.get(name, "kept most aggressive mode (floor already met)"),
        )
        for name in order
    ]
    return ModeSearchReport(
        baseline_cosine=baseline_cos,
        final_cosine=current,
        quality_floor=quality_floor,
        met_floor=current >= quality_floor,
        assignments=assignments,
        probes=probes,
    )


def _relax(mode: LayerMode) -> LayerMode | None:
    """Next-less-aggressive mode, or None when already skipped."""
    if mode == "binary_xnor":
        return "ternary_weight_only"
    if mode == "ternary_weight_only":
        return "skip"
    return None


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
