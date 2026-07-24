#!/usr/bin/env python3
"""Build a Pareto dual-metric JSON report (W7.T03 / W7.T04).

Examples::

  python scripts/pareto_report.py --demo --out results/pareto_demo.json
  python scripts/pareto_report.py --from-optimise results/optimise_report.json \\
      --out results/pareto.json --plot results/pareto.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bnn.eval.pareto import (  # noqa: E402
    ParetoPoint,
    build_pareto_report,
    demo_points,
    validate_pareto_report,
)


def _load_optimise_point(path: Path) -> ParetoPoint:
    data = json.loads(path.read_text(encoding="utf-8"))
    eff = data.get("effectiveness") or {}
    acc = eff.get("cosine")
    if acc is None:
        acc = eff.get("top1_agreement")
    lat = data.get("e2e_latency_ms_wrapped")
    return ParetoPoint(
        name=f"optimise:{data.get('policy')}/{data.get('mode')}",
        accuracy=float(acc) if acc is not None else None,
        compression=float(data.get("compression_replaced_weights") or 0.0),
        latency_ms=float(lat) if lat is not None else None,
        energy_proxy=data.get("energy_proxy"),
        notes=str(data.get("status") or ""),
        accuracy_metric="cosine" if eff.get("cosine") is not None else "top1_or_unknown",
        extra={"source": str(path), "schema": data.get("schema")},
    )


def _try_plot(report: dict, out_png: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skip plot (optional W7.T04)", flush=True)
        return False

    xs, ys, labels = [], [], []
    for pt in report["points"]:
        lat = pt.get("latency_ms")
        acc = pt.get("accuracy")
        if lat is None or acc is None:
            continue
        xs.append(lat)
        ys.append(acc)
        labels.append(pt.get("name", "?"))
    if not xs:
        print("no plottable points (need latency_ms + accuracy)", flush=True)
        return False

    fig, ax = plt.subplots(figsize=(6, 4))
    sizes = [max(20.0, float(pt.get("compression") or 1) * 3) for pt in report["points"] if pt.get("latency_ms") is not None and pt.get("accuracy") is not None]
    ax.scatter(xs, ys, s=sizes, alpha=0.75)
    for x, y, lab in zip(xs, ys, labels):
        ax.annotate(lab, (x, y), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("latency_ms (wall-clock)")
    ax.set_ylabel("accuracy (cosine/top1)")
    ax.set_title("BNN Pareto (theory compression = marker size)")
    ax.grid(True, alpha=0.3)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"Wrote plot {out_png}", flush=True)
    return True


def main() -> int:
    p = argparse.ArgumentParser(description="Pareto dual-metric report")
    p.add_argument("--demo", action="store_true", help="Emit synthetic schema smoke points")
    p.add_argument("--from-optimise", type=Path, action="append", default=[], help="optimise JSON(s)")
    p.add_argument("--out", type=Path, default=ROOT / "results" / "pareto_report.json")
    p.add_argument("--plot", type=Path, default=None, help="Optional PNG (needs matplotlib)")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--threads", type=int, default=None)
    args = p.parse_args()

    points: list[ParetoPoint] = []
    if args.demo:
        points.extend(demo_points())
    for path in args.from_optimise:
        points.append(_load_optimise_point(path))
    if not points:
        print("No points — use --demo or --from-optimise", file=sys.stderr)
        return 2

    report = build_pareto_report(
        points,
        warmup=args.warmup,
        threads=args.threads,
    )
    errs = validate_pareto_report(report)
    if errs:
        print("schema errors:", errs, file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {args.out} ({len(points)} points)", flush=True)
    if args.plot is not None:
        _try_plot(report, args.plot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
