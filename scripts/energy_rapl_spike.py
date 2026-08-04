#!/usr/bin/env python3
"""RAPL / energy-proxy spike (moonshot M5).

* **Linux + readable powercap:** times a short busy loop under RAPL and writes
  measured package Joules to ``results/energy_rapl_spike.json``.
* **Windows / no RAPL:** writes ``CLOSED-BY-PROXY`` status and exits 0 (honest
  residual — not a failed gate).

Also refreshes Pareto-shaped ``energy_proxy`` fields from wrap latencies when
``--from-bound`` is set.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bnn.energy import (  # noqa: E402
    RAPLUnavailable,
    build_energy_bound,
    detect_rapl,
    measure_rapl_joules,
)
from bnn.paths import repo_relative  # noqa: E402


def _busy_xnor_loop(seconds: float = 0.5) -> None:
    """CPU burn so RAPL counters move; not a published golden shape."""
    import numpy as np

    rng = np.random.default_rng(0)
    a = rng.integers(0, 2**63, size=(1024, 1024), dtype=np.int64)
    b = rng.integers(0, 2**63, size=(1024, 1024), dtype=np.int64)
    t_end = time.perf_counter() + seconds
    acc = np.int64(0)
    while time.perf_counter() < t_end:
        acc ^= np.bitwise_xor(a, b).sum()
    _ = int(acc)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=ROOT / "results" / "energy_rapl_spike.json")
    p.add_argument("--sleep-s", type=float, default=0.75, help="Busy-loop duration for RAPL delta")
    p.add_argument(
        "--from-bound",
        type=Path,
        default=None,
        help="Optional wrap JSON to also emit energy_bound-style energy_proxy fields",
    )
    p.add_argument("--power-w-binary", type=float, default=25.0)
    p.add_argument("--power-w-fp", type=float, default=35.0)
    args = p.parse_args()

    payload: dict = {
        "schema": "bnn_energy_rapl_spike_v1",
        "os": platform.system(),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }

    domains = detect_rapl()
    if domains:
        try:
            measured = measure_rapl_joules(lambda: _busy_xnor_loop(args.sleep_s))
            payload.update(
                {
                    "board_joules_status": "MEASURED_RAPL",
                    "rapl_domains": [d.name for d in domains],
                    "measurement": measured,
                    "note": (
                        "Timed busy loop — pedagogy spike, not a golden floor. "
                        "Dual-metric: do not claim GPU 32× from sign()/STE."
                    ),
                }
            )
        except RAPLUnavailable as exc:
            payload.update(
                {
                    "board_joules_status": "CLOSED-BY-PROXY",
                    "rapl_error": str(exc),
                    "note": "Domains listed but measure failed; use energy_bound proxy.",
                }
            )
    else:
        payload.update(
            {
                "board_joules_status": (
                    "CLOSED-BY-PROXY: no readable RAPL powercap on this host; "
                    "energy-bound proxy remains the default path."
                ),
                "rapl_domains": [],
                "note": (
                    "Windows and locked-down Linux keep E=P*t. "
                    "See docs/spikes/RAPL_ENERGY_SPIKE.md."
                ),
            }
        )

    if args.from_bound is not None and args.from_bound.is_file():
        wrap = json.loads(args.from_bound.read_text(encoding="utf-8"))
        bound = build_energy_bound(
            t_fp_s=wrap["e2e_latency_ms_fp"] / 1000.0,
            t_bin_s=wrap["e2e_latency_ms_wrapped"] / 1000.0,
            power_w_fp=args.power_w_fp,
            power_w_binary=args.power_w_binary,
            source_latency=args.from_bound,
        )
        payload["energy_proxy"] = bound.payload["energy_proxy"]
        payload["energy_bound_excerpt"] = {
            "energy_j": bound.payload["energy_j"],
            "board_joules_status": bound.payload["board_joules_status"],
            "source_latency": repo_relative(args.from_bound),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
