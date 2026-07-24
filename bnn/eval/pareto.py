"""Pareto report: accuracy / compression / latency / energy-proxy (W7.T03).

Dual-metric rule
----------------
``compression`` is theoretical pack ratio. ``latency_ms`` / ``energy_proxy`` are
wall-clock / estimate — never conflate with 32× theory.
"""

from __future__ import annotations

import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

PARETO_SCHEMA_ID = "bnn_pareto_report_v1"
PARETO_SCHEMA_VERSION = 1

REQUIRED_POINT_KEYS = frozenset(
    {
        "name",
        "accuracy",
        "compression",
        "latency_ms",
        "energy_proxy",
    }
)


@dataclass
class ParetoPoint:
    """One optimiser / baseline configuration on the fair protocol."""

    name: str
    accuracy: float | None
    compression: float
    latency_ms: float | None
    energy_proxy: float | None = None
    notes: str = ""
    # Honesty tags
    accuracy_metric: str = "cosine_or_top1"  # document which
    compression_is_theory: bool = True
    latency_is_wall_clock: bool = True
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        extra = d.pop("extra", {}) or {}
        d.update(extra)
        return d


def _machine_meta() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def build_pareto_report(
    points: list[ParetoPoint] | list[dict[str, Any]],
    *,
    protocol: str = "docs/FAIR_EVAL_PROTOCOL.md",
    bench_shapes_ref: str = "docs/BENCH_SHAPES.md",
    warmup: int | None = None,
    threads: int | None = None,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a versioned Pareto JSON payload."""
    serialised: list[dict[str, Any]] = []
    for p in points:
        if isinstance(p, ParetoPoint):
            serialised.append(p.to_dict())
        else:
            serialised.append(dict(p))

    out: dict[str, Any] = {
        "schema": PARETO_SCHEMA_ID,
        "schema_version": PARETO_SCHEMA_VERSION,
        "protocol": protocol,
        "bench_shapes_ref": bench_shapes_ref,
        "thesis_note": (
            "Dual-metric: compression is theory (pack ratio); latency_ms and "
            "energy_proxy are wall-clock / estimate. Never claim GPU 32× from sign()/STE."
        ),
        "warmup": warmup,
        "threads": threads,
        "machine": _machine_meta(),
        "points": serialised,
    }
    if meta:
        out["meta"] = meta
    return out


def validate_pareto_report(payload: dict[str, Any]) -> list[str]:
    """Return validation errors (empty ⇒ OK)."""
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["report must be a dict"]
    if payload.get("schema") != PARETO_SCHEMA_ID:
        errors.append(f"schema must be {PARETO_SCHEMA_ID!r}")
    if payload.get("schema_version") != PARETO_SCHEMA_VERSION:
        errors.append(f"schema_version must be {PARETO_SCHEMA_VERSION}")
    points = payload.get("points")
    if not isinstance(points, list) or not points:
        errors.append("points must be a non-empty list")
        return errors
    for i, pt in enumerate(points):
        if not isinstance(pt, dict):
            errors.append(f"points[{i}] must be a dict")
            continue
        for key in REQUIRED_POINT_KEYS:
            if key not in pt:
                errors.append(f"points[{i}] missing {key}")
        # Soft honesty: very high compression + latency labeled as speedup elsewhere
        note = str(payload.get("thesis_note") or "")
        if "dual-metric" not in note.lower() and "theory" not in note.lower():
            errors.append("thesis_note must mention theory vs wall-clock / dual-metric")
            break
    return errors


def demo_points() -> list[ParetoPoint]:
    """Tiny synthetic points for CI / schema smoke (not golden floors)."""
    return [
        ParetoPoint(
            name="fp32_baseline",
            accuracy=1.0,
            compression=1.0,
            latency_ms=10.0,
            energy_proxy=1.0,
            notes="Reference FP32; compression=1 means no pack",
            accuracy_metric="relative_ref",
        ),
        ParetoPoint(
            name="binary_xnor_packed",
            accuracy=0.92,
            compression=32.0,
            latency_ms=4.0,
            energy_proxy=0.4,
            notes="Illustrative dual-metric point — not a published golden",
            accuracy_metric="cosine",
        ),
        ParetoPoint(
            name="ternary_weight_only",
            accuracy=0.97,
            compression=16.0,
            latency_ms=7.0,
            energy_proxy=0.7,
            notes="Size win; GEMM still FP — honesty",
            accuracy_metric="cosine",
        ),
    ]
