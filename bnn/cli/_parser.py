"""Argparse surface for ``bnn <command>``. Shape is locked by ``tests/test_cli_surface.py``."""

from __future__ import annotations

import argparse
from pathlib import Path

from bnn._version import __version__

from ._commands import (
    cmd_bench,
    cmd_bridge,
    cmd_compile_native,
    cmd_decode,
    cmd_encode,
    cmd_energy_bound,
    cmd_eval_suite,
    cmd_export_check,
    cmd_kg,
    cmd_memory,
    cmd_optimise,
    cmd_pareto,
    cmd_profile,
    cmd_recommend,
    cmd_repro,
    cmd_train,
    cmd_train_audio,
    cmd_train_cifar,
    cmd_train_image,
    cmd_train_seq2seq,
    cmd_validate_native,
    cmd_version,
    cmd_wrap,
    cmd_wrap_transformer,
)
from ._dispatch import BRIDGE_ALIASES, BRIDGE_RECIPES, EPILOG, ROOT


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bnn",
        description=(
            "Extreme low-bit inference lab. Training is STE/simulation (not faster). "
            "Inference wins need packed kernels on CPU/edge — not sign() on GPU."
        ),
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"bnn {__version__}",
    )
    sub = p.add_subparsers(dest="command", required=True)

    cn = sub.add_parser("compile-native", help="Build MSVC x64 popcount DLL (Windows)")
    cn.add_argument("--force", action="store_true", help="Rebuild even if DLL exists")
    cn.set_defaults(func=cmd_compile_native)

    sub.add_parser(
        "validate-native",
        help="Assert native GEMM err=0 vs ±1 FP (fails loudly if DLL missing)",
    ).set_defaults(func=cmd_validate_native)

    b = sub.add_parser("bench", help="Kernel microbench (theory vs wall-clock)")
    b.add_argument("--reps", type=int, default=None)
    b.add_argument("--warmup", type=int, default=None)
    b.add_argument(
        "--threads",
        type=str,
        default=None,
        help="Comma list for OpenMP scaling, e.g. 1,2,4,8",
    )
    b.set_defaults(func=cmd_bench)

    sub.add_parser("export-check", help="Assert ~32× weight pack compression").set_defaults(
        func=cmd_export_check
    )

    t = sub.add_parser("train", help="MNIST STE train (simulation — not a throughput win)")
    t.add_argument("--epochs", type=int, default=3)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--model", type=str, default=None, help="Optional single model name")
    t.add_argument("--threads", type=int, default=None)
    t.set_defaults(func=cmd_train)

    c = sub.add_parser("train-cifar", help="CIFAR-10 Bi-Real proxy (legacy script)")
    c.add_argument("--epochs", type=int, default=5)
    c.add_argument("--subset", type=int, default=20000)
    c.add_argument("--batch-size", type=int, default=128)
    c.set_defaults(func=cmd_train_cifar)

    img = sub.add_parser(
        "train-image",
        help="Image lane: CIFAR-10 FP vs Bi-Real (+ optional ViT / ResNet-BiReal)",
    )
    img.add_argument("--epochs", type=int, default=8)
    img.add_argument("--subset", type=int, default=30000, help="0 = full 50k")
    img.add_argument("--batch-size", type=int, default=128)
    img.add_argument("--channels", type=int, default=64)
    img.add_argument("--seed", type=int, default=0)
    img.add_argument("--approx-sign", action="store_true")
    img.add_argument("--include-vit", action="store_true")
    img.add_argument(
        "--include-resnet",
        action="store_true",
        help="Also train ResNet-BiReal CIFAR reference (W4.T05)",
    )
    img.add_argument("--resnet-width", type=int, default=16)
    img.add_argument("--out", type=Path, default=None)
    img.set_defaults(func=cmd_train_image)

    aud = sub.add_parser(
        "train-audio",
        help="Audio lane: synthetic tone spectrograms (not production ASR)",
    )
    aud.add_argument("--epochs", type=int, default=5)
    aud.add_argument("--batch-size", type=int, default=64)
    aud.add_argument("--n-train", type=int, default=800)
    aud.add_argument("--n-test", type=int, default=200)
    aud.add_argument("--n-classes", type=int, default=8)
    aud.add_argument("--channels", type=int, default=32)
    aud.add_argument("--seed", type=int, default=0)
    aud.add_argument("--approx-sign", action="store_true")
    aud.add_argument("--out", type=Path, default=None)
    aud.set_defaults(func=cmd_train_audio)

    rp = sub.add_parser(
        "repro",
        help="Verify published goldens (fast) or run full smoke regen — exit 0 on PASS",
    )
    rp.add_argument(
        "--mode",
        choices=("verify", "full"),
        default="verify",
        help="verify = few-min gates; full = +short smoke trains",
    )
    rp.add_argument(
        "--overwrite-goldens",
        action="store_true",
        help="With --mode full, overwrite results/*.json (off by default)",
    )
    rp.add_argument("--skip-compile", action="store_true")
    rp.add_argument("--skip-pytest", action="store_true")
    rp.set_defaults(func=cmd_repro)

    w = sub.add_parser(
        "wrap",
        help="Wrap demo MLP (legacy) or --ultra hybrid/calib/ternary path",
    )
    w.add_argument(
        "--mode",
        default="binary_xnor",
        choices=[
            "binary_xnor",
            "ternary_weight_only",
            "binary_weight_only_dequant",
            "auto",
        ],
    )
    w.add_argument(
        "--policy",
        default="hybrid_ffn",
        choices=["hybrid_ffn", "aggressive", "ternary_wo", "auto", "default"],
    )
    w.add_argument("--hidden", type=int, default=4096)
    w.add_argument("--batch", type=int, default=32)
    w.add_argument("--ultra", action="store_true", help="Run ultra wrap demo")
    w.add_argument("--d-model", type=int, default=512)
    w.add_argument("--ff", type=int, default=2048)
    w.add_argument("--calib-batches", type=int, default=4)
    w.add_argument("--min-width", type=int, default=64)
    w.add_argument("--qat-steps", type=int, default=0)
    w.add_argument("--drop-in-threshold", type=float, default=0.85)
    w.add_argument("--force", action="store_true", help="Allow drop-in claim below threshold")
    w.add_argument("--report", type=Path, default=None, help="JSON report path (ultra)")
    w.add_argument("--compare-baseline", action="store_true")
    w.set_defaults(func=cmd_wrap)

    opt = sub.add_parser(
        "optimise",
        help="Optimise demo model: calibrate → policy → wrap → report (+ optional .bnnpack)",
    )
    opt.add_argument(
        "--mode",
        default="auto",
        choices=[
            "binary_xnor",
            "ternary_weight_only",
            "binary_weight_only_dequant",
            "auto",
        ],
    )
    opt.add_argument(
        "--policy",
        default="auto",
        choices=["hybrid_ffn", "aggressive", "ternary_wo", "auto", "default"],
    )
    opt.add_argument("--batch", type=int, default=32)
    opt.add_argument("--d-model", type=int, default=512)
    opt.add_argument("--ff", type=int, default=2048)
    opt.add_argument("--calib-batches", type=int, default=4)
    opt.add_argument("--min-width", type=int, default=64)
    opt.add_argument("--qat-steps", type=int, default=0)
    opt.add_argument("--drop-in-threshold", type=float, default=0.85)
    opt.add_argument("--force", action="store_true")
    opt.add_argument(
        "--report",
        type=Path,
        default=ROOT / "results" / "optimise_report.json",
        help="JSON report path (schema bnn_optimise_report_v1)",
    )
    opt.add_argument("--compare-baseline", action="store_true")
    opt.add_argument(
        "--pack",
        type=Path,
        default=None,
        help="Also write a toy MLP .bnnpack via encode (portable artifact smoke)",
    )
    opt.add_argument("--pack-hidden", type=int, default=256)
    opt.set_defaults(func=cmd_optimise)

    sub.add_parser("energy-bound", help="Bind energy estimate to wrap latencies").set_defaults(
        func=cmd_energy_bound
    )

    e = sub.add_parser("eval-suite", help="Run gates + regenerate SUMMARY.md")
    e.add_argument("--out", type=Path, default=ROOT / "results" / "SUMMARY.md")
    e.add_argument("--full", action="store_true", help="Include short image/audio smokes")
    e.add_argument("--skip-pytest", action="store_true")
    e.add_argument(
        "--strict-budgets",
        action="store_true",
        help="Fail when soft latency budgets are exceeded (W13.T03; default warn-only)",
    )
    e.set_defaults(func=cmd_eval_suite)

    pa = sub.add_parser(
        "pareto",
        help="Dual-metric Pareto JSON (accuracy / compression / latency / energy-proxy)",
    )
    pa.add_argument("--demo", action="store_true", help="Synthetic schema smoke points")
    pa.add_argument(
        "--from-optimise",
        type=Path,
        action="append",
        default=[],
        help="Load point(s) from bnn_optimise_report_v1 JSON",
    )
    pa.add_argument(
        "--from-results",
        action="store_true",
        help="Load dual-metric points from committed results/*.json (no invented shapes)",
    )
    pa.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "pareto_report.json",
    )
    pa.add_argument("--plot", type=Path, default=None, help="Optional PNG (matplotlib)")
    pa.add_argument("--warmup", type=int, default=3)
    pa.add_argument("--threads", type=int, default=None)
    pa.set_defaults(func=cmd_pareto)

    br = sub.add_parser(
        "bridge",
        help="Production bridges (GPU INT4/FP8, CPU LLM / bitnet.cpp) — not classic BNN",
    )
    br_sub = br.add_subparsers(dest="bridge_action", required=True)
    br_sub.add_parser("list", help="List bridge recipes").set_defaults(func=cmd_bridge)
    for key, meta in BRIDGE_RECIPES.items():
        bp = br_sub.add_parser(key, help=meta["summary"])
        bp.add_argument("--out", type=Path, default=None, help="Write recipe JSON")
        if key == "gpu":
            bp.add_argument(
                "--probe",
                action="store_true",
                help="Check whether torchao is importable",
            )
        bp.set_defaults(func=cmd_bridge)
    for alias, canon in BRIDGE_ALIASES.items():
        ap = br_sub.add_parser(alias, help=f"Alias for `bnn bridge {canon}`")
        ap.add_argument("--out", type=Path, default=None)
        if canon == "gpu":
            ap.add_argument("--probe", action="store_true")
        ap.set_defaults(func=cmd_bridge)
    fig = br_sub.add_parser(
        "figures",
        help="Build figure manifest / plots from committed results/*.json (W12.T03)",
    )
    fig.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "figures_manifest.json",
    )
    fig.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help="Optional directory for PNG plots (matplotlib)",
    )
    fig.set_defaults(func=cmd_bridge)

    r = sub.add_parser("recommend", help="Recommend stack for a deployment goal")
    r.add_argument(
        "--goal",
        required=True,
        choices=["gpu-server", "cpu-llm", "edge-vision", "npu-phone", "research-xnor", "diffusion"],
    )
    r.set_defaults(func=cmd_recommend)

    enc = sub.add_parser("encode", help="Encode Linear weights → portable .bnnpack")
    enc.add_argument("--source", choices=("mlp", "random"), default="mlp")
    enc.add_argument("--out", type=Path, default=ROOT / "results" / "model.bnnpack")
    enc.add_argument("--hidden", type=int, default=256)
    enc.add_argument("--min-width", type=int, default=1)
    enc.add_argument("--in-features", type=int, default=512)
    enc.add_argument("--out-features", type=int, default=512)
    enc.set_defaults(func=cmd_encode)

    dec = sub.add_parser("decode", help="Load .bnnpack and verify GEMM round-trip err=0")
    dec.add_argument("--pack", type=Path, required=True)
    dec.set_defaults(func=cmd_decode)

    pr = sub.add_parser(
        "profile",
        help="Pack / GEMM / overhead breakdown vs torch FP32 + INT8-WO baselines",
    )
    pr.add_argument("--batch", type=int, default=64)
    pr.add_argument("--in-features", type=int, default=4096)
    pr.add_argument("--out-features", type=int, default=4096)
    pr.add_argument(
        "--no-baselines",
        action="store_true",
        help="Skip FP32/INT8-WO baseline timings (W13.T06 compare is on by default)",
    )
    pr.add_argument("--reps", type=int, default=20)
    pr.add_argument("--warmup", type=int, default=5)
    pr.add_argument("--out", type=Path, default=None)
    pr.set_defaults(func=cmd_profile)

    mem = sub.add_parser("memory", help="Weight footprint: FP32 vs wrapped (measured + theoretical)")
    mem.add_argument("--dim", type=int, default=1024)
    mem.add_argument("--ff", type=int, default=4096)
    mem.add_argument("--batch", type=int, default=64)
    mem.add_argument("--mode", default="binary_xnor",
                     choices=["binary_xnor", "ternary_weight_only", "binary_weight_only_dequant"])
    mem.add_argument("--out", type=Path, default=None)
    mem.set_defaults(func=cmd_memory)

    s2s = sub.add_parser(
        "train-seq2seq",
        help="Binary Encoder–Decoder reverse task (+ optional autoencoder)",
    )
    s2s.add_argument("--task", choices=("seq2seq", "ae", "both"), default="seq2seq")
    s2s.add_argument("--ffn", choices=("binary", "ternary", "fp"), default="binary")
    s2s.add_argument("--steps", type=int, default=80)
    s2s.add_argument("--batch", type=int, default=32)
    s2s.add_argument("--seq-len", type=int, default=8)
    s2s.add_argument("--dim", type=int, default=64)
    s2s.add_argument("--seed", type=int, default=0)
    s2s.add_argument("--out", type=Path, default=None)
    s2s.set_defaults(func=cmd_train_seq2seq)

    wt = sub.add_parser(
        "wrap-transformer",
        help="Tiny Transformer hybrid_ffn wrap + QAT + metrics JSON",
    )
    wt.add_argument("--d-model", type=int, default=128)
    wt.add_argument("--ff", type=int, default=512)
    wt.add_argument("--depth", type=int, default=2)
    wt.add_argument("--batch", type=int, default=32)
    wt.add_argument("--qat-steps", type=int, default=40)
    wt.add_argument("--policy", default="hybrid_ffn")
    wt.add_argument("--out", type=Path, default=None)
    wt.set_defaults(func=cmd_wrap_transformer)

    ver = sub.add_parser("version", help="Print package version")
    ver.set_defaults(func=cmd_version)

    kg = sub.add_parser(
        "kg",
        help="Knowledge graph validate/summary (thesis + lab map)",
    )
    kg.add_argument(
        "action",
        nargs="?",
        default="summary",
        choices=("summary", "validate"),
        help="summary (default) or validate",
    )
    kg.set_defaults(func=cmd_kg)

    return p
