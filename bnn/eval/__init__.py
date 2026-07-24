"""Evaluation helpers: Pareto dual-metric reports (W7)."""

from __future__ import annotations

from .pareto import (
    PARETO_SCHEMA_ID,
    PARETO_SCHEMA_VERSION,
    ParetoPoint,
    build_pareto_report,
    validate_pareto_report,
)

__all__ = [
    "PARETO_SCHEMA_ID",
    "PARETO_SCHEMA_VERSION",
    "ParetoPoint",
    "build_pareto_report",
    "validate_pareto_report",
]
