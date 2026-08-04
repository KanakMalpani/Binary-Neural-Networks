"""Build energy-bound payloads for ``results/energy_bound.json`` + Pareto fields."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bnn.paths import repo_relative

from .proxy import estimate_energy, proxy_status_windows
from .rapl import detect_rapl


LITERATURE_ANCHORS = {
    "bitnet_cpp_cpu_energy_reduction_pct": "55–82% (Microsoft bitnet.cpp reports)",
    "bitnet_b158_arith_energy_7nm": "~71× lower matmul arithmetic energy vs FP16 (paper model)",
    "finn_fpga_fps_per_w": "FINN MNIST prototypes: very high FPS/W (FPGA'17)",
}


@dataclass
class EnergyBoundResult:
    """Structured energy-bound + Pareto energy-proxy fields."""

    payload: dict[str, Any]

    @property
    def energy_proxy_fp(self) -> float:
        return float(self.payload["energy_proxy"]["fp"])

    @property
    def energy_proxy_binary(self) -> float:
        return float(self.payload["energy_proxy"]["binary"])

    @property
    def board_joules_status(self) -> str:
        return str(self.payload["board_joules_status"])


def energy_proxy_relative(*, energy_j_fp: float, energy_j_binary: float) -> dict[str, Any]:
    """Normalise energies so FP baseline = 1.0 (Pareto ``energy_proxy`` field)."""
    if energy_j_fp <= 0:
        return {
            "fp": 1.0,
            "binary": None,
            "unit": "relative_to_fp",
            "note": "fp energy non-positive; cannot normalise",
        }
    return {
        "fp": 1.0,
        "binary": energy_j_binary / energy_j_fp,
        "unit": "relative_to_fp",
        "note": (
            "Pareto energy_proxy: FP=1.0 reference; binary is E_bin/E_fp. "
            "Not a GPU claim; wall-clock/proxy only."
        ),
    }


def build_energy_bound(
    *,
    t_fp_s: float,
    t_bin_s: float,
    power_w_fp: float = 35.0,
    power_w_binary: float = 25.0,
    source_latency: str | Path | None = None,
    prefer_rapl: bool = True,
) -> EnergyBoundResult:
    """Bind measured latencies to energy (RAPL metadata when present, else proxy).

    On Linux with readable powercap, status becomes ``RAPL_AVAILABLE`` and the
    payload records domains — wrap e2e Joules still use the latency×P proxy unless
    a separate RAPL spike timed the same workload (see ``scripts/energy_rapl_spike.py``).
    """
    e_fp_row = estimate_energy(latency_s=t_fp_s, power_w=power_w_fp)
    e_bin_row = estimate_energy(latency_s=t_bin_s, power_w=power_w_binary)
    e_fp = float(e_fp_row["energy_j"])
    e_bin = float(e_bin_row["energy_j"])

    rapl_domains = detect_rapl() if prefer_rapl else []
    if rapl_domains:
        status = (
            "RAPL_AVAILABLE: powercap domains readable; "
            "wrap energy_j still from E=P*t (measured latency × assumed P). "
            "Use scripts/energy_rapl_spike.py for timed package Joules on a kernel loop."
        )
        measurement_method = "proxy_E_eq_P_times_t_with_rapl_probe"
        rapl_meta: dict[str, Any] = {
            "available": True,
            "domains": [
                {"name": d.name, "path": str(d.path), "max_energy_uj": d.max_energy_uj}
                for d in rapl_domains
            ],
            "backend": "linux_powercap",
        }
    else:
        status = proxy_status_windows()
        measurement_method = "proxy_E_eq_P_times_t"
        rapl_meta = {
            "available": False,
            "domains": [],
            "backend": None,
            "reason": "non-Linux or powercap missing/unreadable",
        }

    proxy = energy_proxy_relative(energy_j_fp=e_fp, energy_j_binary=e_bin)
    payload: dict[str, Any] = {
        "source_latency": repo_relative(source_latency) if source_latency else None,
        "measured_latency_s": {"fp": t_fp_s, "binary_wrap": t_bin_s},
        "assumed_power_w": {"fp": power_w_fp, "binary": power_w_binary},
        "energy_j": {"fp": e_fp, "binary": e_bin},
        "energy_reduction_factor_if_power_as_assumed": (e_fp / e_bin) if e_bin else None,
        "energy_reduction_latency_only_same_power": (t_fp_s / t_bin_s) if t_bin_s else None,
        "energy_proxy": proxy,
        "measurement_method": measurement_method,
        "board_joules_status": status,
        "rapl": rapl_meta,
        "literature": dict(LITERATURE_ANCHORS),
        "thesis_note": (
            "Dual-metric: do not conflate pack 32× theory with energy_proxy / RAPL Joules."
        ),
    }
    return EnergyBoundResult(payload=payload)


def write_energy_bound(result: EnergyBoundResult, out: Path) -> Path:
    """Write JSON + companion markdown; return JSON path."""
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = result.payload
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    t_fp = float(payload["measured_latency_s"]["fp"])
    t_bin = float(payload["measured_latency_s"]["binary_wrap"])
    e_fp = float(payload["energy_j"]["fp"])
    e_bin = float(payload["energy_j"]["binary"])
    p_fp = float(payload["assumed_power_w"]["fp"])
    p_bin = float(payload["assumed_power_w"]["binary"])
    red_p = payload["energy_reduction_factor_if_power_as_assumed"]
    red_t = payload["energy_reduction_latency_only_same_power"]
    proxy = payload["energy_proxy"]

    md_lines = [
        "# Energy bound to measured latency",
        "",
        f"- FP latency: {t_fp * 1e3:.2f} ms → E≈{e_fp * 1e3:.1f} mJ @ {p_fp} W",
        f"- Binary wrap: {t_bin * 1e3:.2f} ms → E≈{e_bin * 1e3:.1f} mJ @ {p_bin} W",
        f"- Reduction (assumed P): **{red_p:.2f}×**" if red_p is not None else "- Reduction (assumed P): n/a",
        f"- Reduction (latency-only, same P): **{red_t:.2f}×**" if red_t is not None else "- Reduction (latency-only): n/a",
        f"- Pareto energy_proxy (FP=1): binary={proxy.get('binary')}",
        f"- {payload['board_joules_status']}",
        "",
    ]
    md = out.with_suffix(".md")
    md.write_text("\n".join(md_lines), encoding="utf-8")
    return out
