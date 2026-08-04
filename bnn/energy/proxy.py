"""Order-of-magnitude energy proxy: ``E = P_avg × t_infer``.

Used when RAPL / board Joules are unavailable (typical on Windows).
"""

from __future__ import annotations

import platform
from typing import Any


def estimate_energy(
    *,
    latency_s: float,
    power_w: float,
    baseline_latency_s: float | None = None,
    baseline_power_w: float | None = None,
) -> dict[str, Any]:
    """Return a JSON-serialisable energy row (same shape as ``energy_estimate`` CLI)."""
    e_j = power_w * latency_s
    row: dict[str, Any] = {
        "latency_s": latency_s,
        "power_w": power_w,
        "energy_j": e_j,
        "energy_mj": e_j * 1e3,
        "method": "proxy_E_eq_P_times_t",
    }
    if baseline_latency_s is not None and baseline_power_w is not None:
        e0 = baseline_power_w * baseline_latency_s
        row["baseline_energy_j"] = e0
        row["energy_reduction_factor"] = (e0 / e_j) if e_j else None
        row["note"] = (
            "If binary lowers both P and t, E drops multiplicatively. "
            "BitNet.cpp reports ~55–82% energy reduction on CPU (literature)."
        )
    return row


def proxy_status_windows() -> str:
    """Honest status string for hosts without portable RAPL."""
    system = platform.system()
    if system == "Windows":
        return (
            "CLOSED-BY-PROXY: Windows has no portable RAPL in stdlib; "
            "E=P*t with measured t + assumed P brackets + literature anchors. "
            "Sufficient for decision thesis."
        )
    if system == "Linux":
        return (
            "CLOSED-BY-PROXY: Linux host but RAPL powercap unavailable/unreadable; "
            "E=P*t with measured t + assumed P brackets + literature anchors."
        )
    return (
        f"CLOSED-BY-PROXY: OS={system}; no RAPL path used; "
        "E=P*t with measured t + assumed P brackets + literature anchors."
    )
