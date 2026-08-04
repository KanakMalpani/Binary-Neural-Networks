#!/usr/bin/env python3
"""W12.T03 — figure / claims pipeline from committed ``results/*.json``.

Does **not** invent golden shapes. Reads published floors + committed results
only, emits a claims checklist + optional PNG plots for the paper vault.

Examples::

  python scripts/figure_from_results.py
  python scripts/figure_from_results.py --plot-dir results/figures
  bnn bridge figures --plot-dir results/figures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

# Committed sources only — never invent alternate bench shapes.
SOURCES = {
    "floors": ROOT / "tests" / "golden_floors.json",
    "benchmark": ROOT / "results" / "benchmark.json",
    "train": ROOT / "results" / "train_results.json",
    "image": ROOT / "results" / "image_cifar.json",
    "audio": ROOT / "results" / "audio_synth.json",
    "wrap": ROOT / "results" / "wrap_demo.json",
    "hybrid": ROOT / "results" / "hybrid_ffn_wrap.json",
}


def _load(path: Path) -> Any:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _claim(
    claim_id: str,
    text: str,
    *,
    allowed: bool,
    evidence: str,
    value: Any = None,
    floor: Any = None,
) -> dict[str, Any]:
    return {
        "id": claim_id,
        "text": text,
        "allowed": allowed,
        "evidence": evidence,
        "value": value,
        "floor": floor,
    }


def _acc_from_model_list(blob: Any, model_name: str) -> float | None:
    """Extract test_acc from list- or dict-shaped committed result JSON."""
    if isinstance(blob, list):
        for row in blob:
            if isinstance(row, dict) and row.get("model") == model_name:
                acc = row.get("test_acc")
                return float(acc) if acc is not None else None
        return None
    if not isinstance(blob, dict):
        return None
    for row in blob.get("results") or []:
        if isinstance(row, dict) and row.get("model") == model_name:
            acc = row.get("test_acc")
            return float(acc) if acc is not None else None
    direct = blob.get(model_name)
    if isinstance(direct, dict):
        acc = direct.get("test_acc") or direct.get("acc")
        return float(acc) if acc is not None else None
    if isinstance(direct, (int, float)):
        return float(direct)
    return None


def build_manifest() -> dict[str, Any]:
    floors = _load(SOURCES["floors"]) or {}
    bench = _load(SOURCES["benchmark"]) or {}
    train = _load(SOURCES["train"])
    image = _load(SOURCES["image"]) or {}
    audio = _load(SOURCES["audio"]) or {}
    wrap = _load(SOURCES["wrap"]) or {}
    hybrid = _load(SOURCES["hybrid"]) or {}

    claims: list[dict[str, Any]] = []

    comp_target = floors.get("compression_exact_when_uint64_pack", 32.0)
    wrap_comp = wrap.get("weight_compression_replaced_layers") or wrap.get(
        "weight_compression"
    )
    hybrid_comp = hybrid.get("compression_replaced_weights")
    bench_comps = [
        float(r.get("theoretical", {}).get("weight_compression") or 0)
        for r in bench.get("results") or []
        if r.get("theoretical")
    ]
    observed_comp = wrap_comp or hybrid_comp or (bench_comps[0] if bench_comps else None)
    claims.append(
        _claim(
            "C1_compression_32x",
            "Aligned uint64 binary pack compression is 32.00× (theory, not latency)",
            allowed=observed_comp is not None
            and abs(float(observed_comp) - float(comp_target)) < 1e-6,
            evidence="results/wrap_demo.json | hybrid_ffn_wrap.json | benchmark.json theoretical",
            value=observed_comp,
            floor=comp_target,
        )
    )

    errs = [r.get("max_abs_error_vs_fp32") for r in bench.get("results") or []]
    errs = [e for e in errs if e is not None]
    native_ok = bool(errs) and all(float(e) == 0.0 for e in errs)
    claims.append(
        _claim(
            "C2_native_err0",
            "Native XNOR-popcount GEMM err = 0 vs ±1 FP when kernel present",
            allowed=native_ok,
            evidence="results/benchmark.json max_abs_error_vs_fp32",
            value=errs,
            floor=floors.get("native_err_max", 0.0),
        )
    )

    claims.append(
        _claim(
            "C3_dual_metric",
            "Dual-metric culture: theory compression vs wall-clock latency; never GPU 32× from sign()",
            allowed=True,
            evidence="tests/golden_floors.json notes + docs/PUBLICATION_PLAN.md",
            value="policy",
            floor=None,
        )
    )

    mnist_floor = floors.get("mnist") or {}
    mnist_rec = mnist_floor.get("recorded") or {}
    bin_acc = _acc_from_model_list(train, "binary_mlp")
    if bin_acc is None:
        bin_acc = mnist_rec.get("binary_mlp")
    claims.append(
        _claim(
            "C4_mnist_canary",
            "MNIST binary MLP accuracy within golden floors",
            allowed=bin_acc is not None
            and float(bin_acc) >= float(mnist_floor.get("binary_mlp_min_acc", 95.0)),
            evidence="results/train_results.json vs tests/golden_floors.json mnist",
            value=bin_acc,
            floor=mnist_floor.get("binary_mlp_min_acc"),
        )
    )

    cifar_floor = floors.get("image_cifar") or {}
    cifar_rec = cifar_floor.get("recorded") or {}
    bireal = _acc_from_model_list(image, "binary_cifar_bireal")
    if bireal is None:
        bireal = cifar_rec.get("binary_cifar_bireal")
    claims.append(
        _claim(
            "C5_cifar_canary",
            "CIFAR Bi-Real proxy accuracy within golden floors",
            allowed=bireal is not None
            and float(bireal) >= float(cifar_floor.get("binary_bireal_min_acc", 55.0)),
            evidence="results/image_cifar.json vs tests/golden_floors.json image_cifar",
            value=bireal,
            floor=cifar_floor.get("binary_bireal_min_acc"),
        )
    )

    audio_floor = floors.get("audio_synth") or {}
    audio_rec = audio_floor.get("recorded") or {}
    aud_bin = _acc_from_model_list(audio, "binary_cnn")
    if aud_bin is None:
        aud_bin = audio_rec.get("binary_cnn")
    claims.append(
        _claim(
            "C6_audio_canary",
            "Audio synth binary CNN accuracy within golden floors",
            allowed=aud_bin is not None
            and float(aud_bin) >= float(audio_floor.get("binary_cnn_min_acc", 85.0)),
            evidence="results/audio_synth.json vs tests/golden_floors.json audio_synth",
            value=aud_bin,
            floor=audio_floor.get("binary_cnn_min_acc"),
        )
    )

    speed_rows = []
    for r in bench.get("results") or []:
        shape = r.get("shape") or {}
        speed_rows.append(
            {
                "shape": shape,
                "speedup_compute_vs_numpy_fp32": r.get("speedup_compute_vs_numpy_fp32"),
                "speedup_e2e_vs_numpy_fp32": r.get("speedup_e2e_vs_numpy_fp32"),
                "weight_compression_theory": (r.get("theoretical") or {}).get(
                    "weight_compression"
                ),
                "max_abs_error_vs_fp32": r.get("max_abs_error_vs_fp32"),
            }
        )

    allowed = [c for c in claims if c["allowed"]]
    blocked = [c for c in claims if not c["allowed"]]

    return {
        "schema": "bnn_figures_manifest_v1",
        "schema_version": 1,
        "thesis_note": (
            "Claims whitelist tied to tests/golden_floors.json + committed results/*.json. "
            "Compression is theory; latency/speedup are wall-clock. No invented goldens."
        ),
        "sources": {
            k: str(v.relative_to(ROOT)).replace("\\", "/") for k, v in SOURCES.items()
        },
        "claims": claims,
        "claims_allowed_count": len(allowed),
        "claims_blocked_count": len(blocked),
        "benchmark_dual_metric": speed_rows,
        "forbidden_reminders": [
            "GPU e2e 32× from STE/sign()",
            "Invented bench shapes as the golden",
            "Bit-identical floats across machines as pass criterion",
            "Production ASR / full ImageNet SOTA as delivered",
        ],
        "paper_vault": "C:\\00 Research Papers (see docs/32_NOVEL_PAPER_CANDIDATES.md)",
        "figure_cmds": [
            "bnn bridge figures --plot-dir results/figures",
            "bnn pareto --from-results --out results/pareto_from_results.json "
            "--plot results/pareto_from_results.png",
        ],
    }


def _try_plots(manifest: dict[str, Any], plot_dir: Path) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skip PNG plots", flush=True)
        return []

    plot_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    rows = manifest.get("benchmark_dual_metric") or []
    labels, compute, e2e = [], [], []
    for r in rows:
        sh = r.get("shape") or {}
        lab = f"{sh.get('batch')}×{sh.get('in_features')}×{sh.get('out_features')}"
        if r.get("speedup_compute_vs_numpy_fp32") is None:
            continue
        labels.append(lab)
        compute.append(float(r["speedup_compute_vs_numpy_fp32"]))
        e2e.append(float(r.get("speedup_e2e_vs_numpy_fp32") or 0))
    if labels:
        fig, ax = plt.subplots(figsize=(7, 4))
        x = range(len(labels))
        ax.bar([i - 0.2 for i in x], compute, width=0.4, label="compute vs NumPy FP32")
        ax.bar([i + 0.2 for i in x], e2e, width=0.4, label="e2e (+act pack) vs NumPy FP32")
        ax.axhline(
            32.0,
            color="gray",
            linestyle="--",
            linewidth=1,
            label="theory compression 32× (not latency)",
        )
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylabel("speedup (wall-clock)")
        ax.set_title("Dual-metric: packed CPU speedup ≠ theory 32×")
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        out = plot_dir / "dual_metric_speedup.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        written.append(str(out))

    canaries = [
        c
        for c in manifest["claims"]
        if c["id"] in {"C4_mnist_canary", "C5_cifar_canary", "C6_audio_canary"}
        and c.get("value") is not None
    ]
    names = [c["id"].removeprefix("C").split("_", 1)[-1] for c in canaries]
    vals = [float(c["value"]) for c in canaries]
    floors_v = [float(c["floor"]) for c in canaries if c.get("floor") is not None]
    if names and vals:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.bar(names, vals, label="recorded / committed")
        if floors_v and len(floors_v) == len(names):
            ax.plot(names, floors_v, "r--", marker="o", label="floor min")
        ax.set_ylabel("accuracy %")
        ax.set_title("Canaries vs golden floors (no invented shapes)")
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        out = plot_dir / "canary_vs_floors.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        written.append(str(out))

    return written


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Figure / claims pipeline from committed results")
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "figures_manifest.json",
    )
    p.add_argument("--plot-dir", type=Path, default=None, help="Optional PNG output dir")
    args = p.parse_args(argv)

    missing = [str(v) for v in SOURCES.values() if not v.is_file()]
    if missing:
        print("WARN missing sources (claims may block):", *missing, sep="\n  ", flush=True)

    manifest = build_manifest()
    if args.plot_dir is not None:
        plots = _try_plots(manifest, args.plot_dir)
        manifest["plots"] = plots

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"Wrote {args.out} claims_ok={manifest['claims_allowed_count']} "
        f"blocked={manifest['claims_blocked_count']}",
        flush=True,
    )
    if manifest["claims_blocked_count"]:
        for c in manifest["claims"]:
            if not c["allowed"]:
                print(f"  BLOCKED {c['id']}: {c['text']} (value={c['value']})", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
