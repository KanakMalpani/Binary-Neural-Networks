#!/usr/bin/env python3
"""Bind energy_estimate to measured latencies from results/*.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--power-w-binary", type=float, default=25.0, help="Assumed package W for binary run")
    p.add_argument("--power-w-fp", type=float, default=35.0, help="Assumed package W for FP run")
    p.add_argument(
        "--wrap-json",
        type=Path,
        default=ROOT / "results" / "wrap_demo.json",
    )
    p.add_argument("--out", type=Path, default=ROOT / "results" / "energy_bound.json")
    args = p.parse_args()

    wrap = json.loads(args.wrap_json.read_text(encoding="utf-8"))
    t_fp = wrap["e2e_latency_ms_fp"] / 1000.0
    t_bin = wrap["e2e_latency_ms_wrapped"] / 1000.0
    e_fp = args.power_w_fp * t_fp
    e_bin = args.power_w_binary * t_bin

    # Literature anchors (not measured on this board)
    lit = {
        "bitnet_cpp_cpu_energy_reduction_pct": "55–82% (Microsoft bitnet.cpp reports)",
        "bitnet_b158_arith_energy_7nm": "~71× lower matmul arithmetic energy vs FP16 (paper model)",
        "finn_fpga_fps_per_w": "FINN MNIST prototypes: very high FPS/W (FPGA'17)",
    }

    # Windows: no portable RAPL in stdlib; document proxy
    payload = {
        "source_latency": str(args.wrap_json),
        "measured_latency_s": {"fp": t_fp, "binary_wrap": t_bin},
        "assumed_power_w": {"fp": args.power_w_fp, "binary": args.power_w_binary},
        "energy_j": {"fp": e_fp, "binary": e_bin},
        "energy_reduction_factor_if_power_as_assumed": e_fp / e_bin if e_bin else None,
        "energy_reduction_latency_only_same_power": t_fp / t_bin if t_bin else None,
        "board_joules_status": (
            "CLOSED-BY-PROXY: no RAPL API used; E=P*t with measured t from wrap_demo "
            "+ assumed P brackets + literature anchors. Sufficient for decision thesis."
        ),
        "literature": lit,
    }
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md = args.out.with_suffix(".md")
    md.write_text(
        "\n".join(
            [
                "# Energy bound to measured latency",
                "",
                f"- FP latency: {t_fp*1e3:.2f} ms → E≈{e_fp*1e3:.1f} mJ @ {args.power_w_fp} W",
                f"- Binary wrap: {t_bin*1e3:.2f} ms → E≈{e_bin*1e3:.1f} mJ @ {args.power_w_binary} W",
                f"- Reduction (assumed P): **{payload['energy_reduction_factor_if_power_as_assumed']:.2f}×**",
                f"- Reduction (latency-only, same P): **{payload['energy_reduction_latency_only_same_power']:.2f}×**",
                f"- {payload['board_joules_status']}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
