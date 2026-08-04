#!/usr/bin/env python3
"""Bind energy_estimate to measured latencies from results/*.json.

Uses ``bnn.energy``: Linux RAPL is probed when readable; wrap Joules remain
``E=P*t`` unless a RAPL spike timed the same workload. Windows stays
CLOSED-BY-PROXY with Pareto ``energy_proxy`` fields (FP=1 reference).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bnn.energy import build_energy_bound, write_energy_bound  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--power-w-binary", type=float, default=25.0, help="Assumed package W for binary run")
    p.add_argument("--power-w-fp", type=float, default=35.0, help="Assumed package W for FP run")
    p.add_argument(
        "--wrap-json",
        type=Path,
        default=ROOT / "results" / "wrap_demo.json",
    )
    p.add_argument("--out", type=Path, default=ROOT / "results" / "energy_bound.json")
    p.add_argument(
        "--no-rapl-probe",
        action="store_true",
        help="Skip Linux powercap probe (force proxy-only metadata)",
    )
    args = p.parse_args()

    wrap = json.loads(args.wrap_json.read_text(encoding="utf-8"))
    t_fp = wrap["e2e_latency_ms_fp"] / 1000.0
    t_bin = wrap["e2e_latency_ms_wrapped"] / 1000.0

    result = build_energy_bound(
        t_fp_s=t_fp,
        t_bin_s=t_bin,
        power_w_fp=args.power_w_fp,
        power_w_binary=args.power_w_binary,
        source_latency=args.wrap_json,
        prefer_rapl=not args.no_rapl_probe,
    )
    write_energy_bound(result, args.out)
    print(json.dumps(result.payload, indent=2))


if __name__ == "__main__":
    main()
