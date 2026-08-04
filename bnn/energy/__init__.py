"""Energy measurement helpers — RAPL where OS allows, honest proxy otherwise.

Dual-metric rule
----------------
Compression 32× is **theory** (uint64 pack). Energy figures are either:

* **Measured** package Joules via Linux powercap RAPL, or
* **Proxy** ``E = P_assumed × t_measured`` with status ``CLOSED-BY-PROXY``.

Never conflate arithmetic-energy literature tables with board Joules.
"""

from __future__ import annotations

from .bound import (
    EnergyBoundResult,
    build_energy_bound,
    energy_proxy_relative,
    write_energy_bound,
)
from .proxy import closed_by_proxy_status, estimate_energy, proxy_status_windows
from .rapl import (
    RAPLDomain,
    RAPLMeter,
    RAPLUnavailable,
    detect_rapl,
    measure_rapl_joules,
)

__all__ = [
    "EnergyBoundResult",
    "RAPLDomain",
    "RAPLMeter",
    "RAPLUnavailable",
    "build_energy_bound",
    "closed_by_proxy_status",
    "detect_rapl",
    "energy_proxy_relative",
    "estimate_energy",
    "measure_rapl_joules",
    "proxy_status_windows",
    "write_energy_bound",
]
