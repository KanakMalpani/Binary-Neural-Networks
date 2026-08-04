#!/usr/bin/env python3
"""Order-of-magnitude energy estimate: E = P_avg * t_infer.

Does not require RAPL/SMI — pass measured latency and an assumed or metered power.
On Linux, ``--probe-rapl`` records whether powercap is readable (does not replace P).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bnn.energy import detect_rapl, estimate_energy  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--latency-s", type=float, required=True, help="Seconds per inference")
    p.add_argument("--power-w", type=float, required=True, help="Average package power (W)")
    p.add_argument("--baseline-latency-s", type=float, default=None)
    p.add_argument("--baseline-power-w", type=float, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument(
        "--probe-rapl",
        action="store_true",
        help="Attach Linux RAPL domain list when readable",
    )
    args = p.parse_args()

    row = estimate_energy(
        latency_s=args.latency_s,
        power_w=args.power_w,
        baseline_latency_s=args.baseline_latency_s,
        baseline_power_w=args.baseline_power_w,
    )
    if args.probe_rapl:
        domains = detect_rapl()
        row["rapl"] = {
            "available": bool(domains),
            "domains": [d.name for d in domains],
        }
    print(json.dumps(row, indent=2))
    if args.out:
        args.out.write_text(json.dumps(row, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
