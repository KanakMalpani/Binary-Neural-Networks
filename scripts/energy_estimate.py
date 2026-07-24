#!/usr/bin/env python3
"""Order-of-magnitude energy estimate: E = P_avg * t_infer.

Does not read RAPL/SMI — pass measured latency and an assumed or metered power.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--latency-s", type=float, required=True, help="Seconds per inference")
    p.add_argument("--power-w", type=float, required=True, help="Average package power (W)")
    p.add_argument("--baseline-latency-s", type=float, default=None)
    p.add_argument("--baseline-power-w", type=float, default=None)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    e_j = args.power_w * args.latency_s
    row = {
        "latency_s": args.latency_s,
        "power_w": args.power_w,
        "energy_j": e_j,
        "energy_mj": e_j * 1e3,
    }
    if args.baseline_latency_s and args.baseline_power_w:
        e0 = args.baseline_power_w * args.baseline_latency_s
        row["baseline_energy_j"] = e0
        row["energy_reduction_factor"] = e0 / e_j if e_j else None
        row["note"] = (
            "If binary lowers both P and t, E drops multiplicatively. "
            "BitNet.cpp reports ~55–82% energy reduction on CPU (literature)."
        )
    print(json.dumps(row, indent=2))
    if args.out:
        args.out.write_text(json.dumps(row, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
