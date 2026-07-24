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

    image = _load("image_cifar.json") or cifar
    audio = _load("audio_synth.json") or {}

    lines += ["", "## Image (CIFAR-10 Bi-Real)", ""]
    if image:
        cres = image.get("results") or []
        fp = next((r for r in cres if "fp32" in str(r.get("model", "")).lower()), None)
        bn = next(
            (
                r
                for r in cres
                if "binary" in str(r.get("model", "")).lower()
                and "vit" not in str(r.get("model", "")).lower()
            ),
            None,
        )
        if fp and bn:
            lines.append(f"- FP32 CNN: **{fp['test_acc']:.2f}%**")
            lines.append(f"- Binary Bi-Real: **{bn['test_acc']:.2f}%**")
            gap = image.get("acc_gap_pp_fp_vs_binary_cnn", image.get("acc_gap_pp"))
            if gap is None:
                gap = fp["test_acc"] - bn["test_acc"]
            lines.append(f"- Gap: **{gap:.2f} pp**")
        src = "image_cifar.json" if (_load("image_cifar.json")) else "cifar10_proxy.json"
        lines.append(f"Source: `{src}`. Tutorial: `docs/tutorials/04_image_cifar.md`.")
    else:
        lines.append("_No image results — run `bnn train-image`._")

    lines += ["", "## Audio (synthetic tones)", ""]
    if audio:
        ares = audio.get("results") or []
        fp = next((r for r in ares if "fp32" in str(r.get("model", "")).lower()), None)
        bn = next((r for r in ares if "binary" in str(r.get("model", "")).lower()), None)
        if fp and bn:
            lines.append(f"- FP32 CNN: **{fp['test_acc']:.2f}%**")
            lines.append(f"- Binary CNN: **{bn['test_acc']:.2f}%**")
            lines.append(f"- Gap: **{audio.get('acc_gap_pp', fp['test_acc']-bn['test_acc']):.2f} pp**")
        lines.append(
            "Source: `audio_synth.json`. "
            "**Not production ASR** — INT8 Whisper/ORT for real speech. "
            "Tutorial: `docs/tutorials/05_audio.md`."
        )
    else:
        lines.append("_No audio_synth.json — run `bnn train-audio`._")

    lines += ["", "## Wrap / energy / robustness", ""]
    if wrap:
        e2e_fp = wrap.get("e2e_latency_ms_fp")
        e2e_w = wrap.get("e2e_latency_ms_wrapped")
        e2e_s = wrap.get("e2e_speedup")
        comp = wrap.get("weight_compression_replaced_layers") or wrap.get("compression")
        cos = wrap.get("output_cosine_vs_fp")
        gemm = (wrap.get("layer_microbench") or {}).get("speedup_gemm_only_vs_torch_linear")
        lines.append(
            f"- Wrap e2e latency: FP **{e2e_fp:.2f}** ms → wrapped **{e2e_w:.2f}** ms "
            f"(e2e **{e2e_s:.2f}×**)"
            if isinstance(e2e_fp, (int, float)) and isinstance(e2e_w, (int, float))
            else f"- Wrap e2e: {e2e_fp} → {e2e_w} ms"
        )
        if comp is not None:
            lines.append(f"- Weight compression (replaced layers): **{comp}×** (exact bit-pack)")
        if gemm is not None:
            lines.append(f"- Layer gemm_only vs torch Linear: **{gemm:.2f}×** (kernel ROI)")
        if cos is not None:
            lines.append(
                f"- Output cosine vs FP: **{cos:.3f}** "
                "(low without QAT is expected — not a transparent wrap)"
            )
    if energy:
        er = energy.get("energy_reduction_latency_only_same_power")
        if isinstance(er, (int, float)):
            lines.append(
                f"- Energy (latency-only, same power proxy): **{er:.2f}×** (`energy_bound.json`)"
            )
        else:
            lines.append(f"- Energy latency-only reduction: **{er}×** (`energy_bound.json`)")
    if fgsm:
        for r in fgsm.get("results", []):
            lines.append(
                f"- FGSM {r.get('model')}: clean {r.get('clean_acc')}% → "
                f"{r.get('fgsm_acc')}% (drop {r.get('drop_pp')} pp)"
            )

    lines += [
        "",
        "## Honesty / dual reporting",
        "",
        "| Quantity | Meaning | Do not claim as |",
        "|----------|---------|-----------------|",
        "| Weight compression **32×** | Bit-pack bytes | e2e latency |",
        "| Theoretical word reduction ~64× | XNOR-popcount ops | wall-clock |",
        "| Kernel speedup (bench) | Prepacked GEMM vs NumPy/Torch FP | full-model FPS |",
        "| E2E wrap speedup | Whole forward | quality-preserving wrap |",
        "",
        r"Amdahl: \(S_{e2e}=\frac{1}{(1-f)+f/S_{kernel}}\). "
        "Fake `sign()`+torch Linear is often **slower** than FP32 on GPU.",
        "",
        "Repro gates: `tests/golden_floors.json` · `bnn repro` · "
        "[`REPRODUCIBILITY.md`](../REPRODUCIBILITY.md).",
        "",
        "More: `docs/19_GAP_CLOSURE_REPORT.md`, `docs/28_IMAGE_AUDIO_COMPLETION.md`, "
        "`docs/29_FINAL_COMPLETION.md`, `docs/31_QUALITY_UPGRADE.md`.",
        "",
    ]
    return "\n".join(lines)


def write_summary(out: Path | None = None, results_dir: Path | None = None) -> Path:
    text = render_summary(results_dir)
    out = Path(out) if out else (results_dir or ROOT / "results") / "SUMMARY.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return out
