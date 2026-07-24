"""Aggregate results/*.json into results/SUMMARY.md."""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def _load(name: str) -> dict[str, Any] | None:
    p = RESULTS / name
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def machine_card() -> dict[str, Any]:
    import torch

    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }


def render_summary(results_dir: Path | None = None) -> str:
    global RESULTS
    if results_dir is not None:
        RESULTS = Path(results_dir)

    bench_raw = _load("benchmark.json")
    train_raw = _load("train_results.json")
    cifar = _load("cifar10_proxy.json") or {}
    wrap = _load("wrap_demo.json") or {}
    energy = _load("energy_bound.json") or {}
    fgsm = _load("robustness_fgsm.json") or {}
    card = machine_card()

    lines = [
        "# Results summary (this workspace)",
        "",
        f"_Regenerated: {card['generated_utc']}_",
        f"_Machine: {card['platform']} | torch {card['torch']} | CUDA={card['cuda']}_",
        "",
        "## Kernel (CPU packed XNOR)",
        "",
    ]

    bench = bench_raw if isinstance(bench_raw, dict) else {}
    rows = bench.get("rows") or bench.get("results") or []
    if isinstance(bench.get("benchmarks"), list):
        rows = bench["benchmarks"]
    if rows:
        lines += [
            "| Shape | S vs NumPy FP32 | S vs Torch FP32 | Err |",
            "|-------|----------------:|----------------:|----:|",
        ]
        for r in rows:
            sh = r.get("shape")
            if isinstance(sh, dict):
                shape = f"{sh.get('batch')}×{sh.get('in_features')}×{sh.get('out_features')}"
            else:
                shape = sh or f"{r.get('batch')}×{r.get('n')}×{r.get('m')}"
            s_np = (
                r.get("speedup_compute_vs_numpy_fp32")
                or r.get("speedup_vs_numpy_fp32")
                or r.get("speedup_numpy")
            )
            s_t = (
                r.get("speedup_compute_vs_torch_fp32")
                or r.get("speedup_vs_torch_fp32")
                or r.get("speedup_torch")
            )
            err = r.get("max_abs_error_vs_fp32") or r.get("max_err") or r.get("err") or 0
            s_np_s = f"{s_np:.2f}" if isinstance(s_np, (int, float)) else "—"
            s_t_s = f"{s_t:.2f}" if isinstance(s_t, (int, float)) else "—"
            lines.append(f"| {shape} | {s_np_s} | {s_t_s} | {err} |")
        comp = None
        if rows and isinstance(rows[0].get("theoretical"), dict):
            comp = rows[0]["theoretical"].get("weight_compression")
        comp = bench.get("compression") or bench.get("weight_compression") or comp
        if comp:
            lines.append("")
            lines.append(f"Compression: **{comp}×**. Source: `benchmark.json`.")
    else:
        lines.append("_No benchmark.json rows found — run `bnn bench`._")

    lines += ["", "## MNIST", ""]
    if isinstance(train_raw, list):
        models = train_raw
    elif isinstance(train_raw, dict):
        models = train_raw.get("results") or train_raw.get("models") or []
    else:
        models = []
    if models:
        lines += ["| Model | Acc % |", "|-------|------:|"]
        for m in models:
            name = m.get("model") or m.get("name")
            acc = m.get("test_acc") or m.get("acc")
            lines.append(f"| {name} | {acc} |")
        lines.append("")
        lines.append("Source: `train_results.json`.")
    else:
        lines.append("_No train_results.json — run `bnn train`._")

    lines += ["", "## CIFAR-10 proxy", ""]
    if cifar:
        cres = cifar.get("results") or []
        fp = next((r for r in cres if "fp32" in r.get("model", "")), None)
        bn = next((r for r in cres if "binary" in r.get("model", "")), None)
        if fp and bn:
            lines.append(f"- FP32: **{fp['test_acc']:.2f}%**")
            lines.append(f"- Binary Bi-Real: **{bn['test_acc']:.2f}%**")
            lines.append(f"- Gap: **{cifar.get('acc_gap_pp', fp['test_acc']-bn['test_acc']):.2f} pp**")
        lines.append("Source: `cifar10_proxy.json`.")
    else:
        lines.append("_No cifar10_proxy.json — run `bnn train-cifar`._")

    lines += ["", "## Wrap / energy / robustness", ""]
    if wrap:
        lines.append(
            f"- Wrap e2e: FP {wrap.get('e2e_latency_ms_fp')} ms → "
            f"binary {wrap.get('e2e_latency_ms_wrapped')} ms "
            f"(compression {wrap.get('compression')})"
        )
    if energy:
        er = energy.get("energy_reduction_latency_only_same_power")
        lines.append(f"- Energy latency-only reduction: **{er}×** (`energy_bound.json`)")
    if fgsm:
        for r in fgsm.get("results", []):
            lines.append(
                f"- FGSM {r.get('model')}: clean {r.get('clean_acc')}% → "
                f"{r.get('fgsm_acc')}% (drop {r.get('drop_pp')} pp)"
            )

    lines += [
        "",
        "## Formula reminder",
        "",
        r"\[",
        r"S_{e2e}=\frac{1}{(1-f)+f/S_{kernel}},\quad R_{arith}\approx 64,\quad compress=32\times",
        r"\]",
        "",
        "Do not advertise \\(R_{arith}\\) as wall-clock.",
        "",
        "Gap closure: `docs/19_GAP_CLOSURE_REPORT.md`. Completion: `docs/22_COMPLETION_REPORT.md`.",
        "",
    ]
    return "\n".join(lines)


def write_summary(out: Path | None = None, results_dir: Path | None = None) -> Path:
    text = render_summary(results_dir)
    out = Path(out) if out else (results_dir or ROOT / "results") / "SUMMARY.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return out
