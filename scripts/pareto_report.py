#!/usr/bin/env python3
"""Build a Pareto dual-metric JSON report (W7.T03 / W7.T04 / W12.T03).

Examples::

  python scripts/pareto_report.py --demo --out results/pareto_demo.json
  python scripts/pareto_report.py --from-optimise results/optimise_report.json \\
      --out results/pareto.json --plot results/pareto.png
  python scripts/pareto_report.py --from-results --out results/pareto_from_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

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


def _acc_from_results(blob: Any, model_name: str) -> float | None:
    if isinstance(blob, list):
        for row in blob:
            if isinstance(row, dict) and row.get("model") == model_name:
                acc = row.get("test_acc")
                return float(acc) if acc is not None else None
        return None
    if isinstance(blob, dict):
        for row in blob.get("results") or []:
            if isinstance(row, dict) and row.get("model") == model_name:
                acc = row.get("test_acc")
                return float(acc) if acc is not None else None
    return None


def _load_committed_result_points() -> list[ParetoPoint]:
    """Dual-metric points from committed ``results/*.json`` only (no invented shapes)."""
    points: list[ParetoPoint] = []

    points.append(
        ParetoPoint(
            name="fp32_reference",
            accuracy=1.0,
            compression=1.0,
            latency_ms=None,
            energy_proxy=None,
            notes="Reference; compression=1 means no pack",
            accuracy_metric="relative_ref",
            extra={"source": "synthetic_ref"},
        )
    )

    wrap = ROOT / "results" / "wrap_demo.json"
    if wrap.is_file():
        data = json.loads(wrap.read_text(encoding="utf-8"))
        comp = float(
            data.get("weight_compression_replaced_layers")
            or data.get("weight_compression")
            or data.get("compression_replaced_weights")
            or 0.0
        )
        lat = data.get("e2e_latency_ms_wrapped") or data.get("latency_ms_wrapped")
        cos = data.get("output_cosine_vs_fp") or data.get("cosine")
        points.append(
            ParetoPoint(
                name="wrap_demo",
                accuracy=float(cos) if cos is not None else None,
                compression=comp,
                latency_ms=float(lat) if lat is not None else None,
                energy_proxy=data.get("energy_proxy"),
                notes="Committed wrap_demo.json",
                accuracy_metric="cosine" if cos is not None else "unknown",
                extra={"source": str(wrap)},
            )
        )

    hybrid = ROOT / "results" / "hybrid_ffn_wrap.json"
    if hybrid.is_file():
        data = json.loads(hybrid.read_text(encoding="utf-8"))
        points.append(
            ParetoPoint(
                name="hybrid_ffn_wrap",
                accuracy=None,
                compression=float(data.get("compression_replaced_weights") or 0.0),
                latency_ms=None,
                energy_proxy=None,
                notes=str(data.get("verdict") or "hybrid FFN wrap"),
                accuracy_metric="n/a",
                extra={"source": str(hybrid)},
            )
        )

    ultra = ROOT / "results" / "ultra_wrap.json"
    if ultra.is_file():
        data = json.loads(ultra.read_text(encoding="utf-8"))
        for key in ("primary", "ternary", "aggressive"):
            block = data.get(key)
            if not isinstance(block, dict):
                continue
            cos = (block.get("effectiveness") or {}).get("cosine") or block.get("cosine")
            lat = block.get("e2e_latency_ms_wrapped") or block.get("latency_ms")
            comp = block.get("compression_replaced_weights") or block.get("compression")
            if comp is None:
                continue
            points.append(
                ParetoPoint(
                    name=f"ultra_wrap:{key}",
                    accuracy=float(cos) if cos is not None else None,
                    compression=float(comp),
                    latency_ms=float(lat) if lat is not None else None,
                    energy_proxy=block.get("energy_proxy"),
                    notes="Committed ultra_wrap.json",
                    accuracy_metric="cosine" if cos is not None else "unknown",
                    extra={"source": str(ultra)},
                )
            )

    train = ROOT / "results" / "train_results.json"
    if train.is_file():
        data = json.loads(train.read_text(encoding="utf-8"))
        bin_acc = _acc_from_results(data, "binary_mlp")
        if bin_acc is not None:
            points.append(
                ParetoPoint(
                    name="mnist_binary_mlp",
                    accuracy=bin_acc / 100.0 if bin_acc > 1.5 else bin_acc,
                    compression=32.0,
                    latency_ms=None,
                    energy_proxy=None,
                    notes="MNIST canary (acc scaled to 0–1 if percent)",
                    accuracy_metric="top1",
                    extra={"source": str(train)},
                )
            )

    bench = ROOT / "results" / "benchmark.json"
    if bench.is_file():
        data = json.loads(bench.read_text(encoding="utf-8"))
        for row in data.get("results") or []:
            shape = row.get("shape") or {}
            sec = (row.get("seconds") or {}).get("binary_compute_only_prepacked")
            theory = (row.get("theoretical") or {}).get("weight_compression")
            if sec is None or theory is None:
                continue
            name = (
                f"bench_{shape.get('batch')}x"
                f"{shape.get('in_features')}x{shape.get('out_features')}"
            )
            points.append(
                ParetoPoint(
                    name=name,
                    accuracy=None,
                    compression=float(theory),
                    latency_ms=float(sec) * 1000.0,
                    energy_proxy=None,
                    notes="Committed benchmark.json compute-only wall-clock",
                    accuracy_metric="n/a",
                    extra={
                        "source": str(bench),
                        "speedup_compute_vs_numpy_fp32": row.get(
                            "speedup_compute_vs_numpy_fp32"
                        ),
                        "max_abs_error_vs_fp32": row.get("max_abs_error_vs_fp32"),
                    },
                )
            )

    return points


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
    sizes = [
        max(20.0, float(pt.get("compression") or 1) * 3)
        for pt in report["points"]
        if pt.get("latency_ms") is not None and pt.get("accuracy") is not None
    ]
    ax.scatter(xs, ys, s=sizes, alpha=0.75)
    for x, y, lab in zip(xs, ys, labels, strict=True):
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


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Pareto dual-metric report")
    p.add_argument("--demo", action="store_true", help="Emit synthetic schema smoke points")
    p.add_argument("--from-optimise", type=Path, action="append", default=[], help="optimise JSON(s)")
    p.add_argument(
        "--from-results",
        action="store_true",
        help="Load points from committed results/*.json (W12.T03; no invented shapes)",
    )
    p.add_argument("--out", type=Path, default=ROOT / "results" / "pareto_report.json")
    p.add_argument("--plot", type=Path, default=None, help="Optional PNG (needs matplotlib)")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--threads", type=int, default=None)
    args = p.parse_args(argv)

    points: list[ParetoPoint] = []
    if args.demo:
        points.extend(demo_points())
    for path in args.from_optimise:
        points.append(_load_optimise_point(path))
    if args.from_results:
        points.extend(_load_committed_result_points())
    if not points:
        print(
            "No points — use --demo, --from-optimise, or --from-results",
            file=sys.stderr,
        )
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
