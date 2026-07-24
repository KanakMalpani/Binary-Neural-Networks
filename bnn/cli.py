"""Console entry point: ``bnn <command> ...``."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from bnn._version import __version__

ROOT = Path(__file__).resolve().parents[1]

EPILOG = """
Thesis: packed binary/ternary kernels for CPU/edge inference.
Training (STE) is simulation — not a GPU 32× claim.
Reproduce:  bnn repro
Agents:     see AGENTS.md
Docs:       REPRODUCIBILITY.md
""".strip()


def _run_script(script: str, extra: list[str] | None = None) -> int:
    cmd = [sys.executable, str(ROOT / "scripts" / script), *(extra or [])]
    print(">", " ".join(cmd), flush=True)
    return int(subprocess.call(cmd, cwd=str(ROOT)))


def cmd_compile_native(args: argparse.Namespace) -> int:
    from bnn.kernels.compile_native import main as compile_main

    extra = ["--force"] if getattr(args, "force", False) else []
    return int(compile_main(extra))


def cmd_validate_native(_: argparse.Namespace) -> int:
    return _run_script("validate_native.py")


def cmd_bench(args: argparse.Namespace) -> int:
    extra: list[str] = []
    if args.reps:
        extra += ["--reps", str(args.reps)]
    if getattr(args, "threads", None):
        extra += ["--threads", str(args.threads)]
    if getattr(args, "warmup", None):
        extra += ["--warmup", str(args.warmup)]
    return _run_script("benchmark.py", extra)


def cmd_export_check(_: argparse.Namespace) -> int:
    return _run_script("export_check.py")


def cmd_train(args: argparse.Namespace) -> int:
    extra = ["--epochs", str(args.epochs), "--seed", str(args.seed)]
    if args.model:
        extra += ["--models", args.model]
    if args.threads:
        extra += ["--threads", str(args.threads)]
    return _run_script("train.py", extra)


def cmd_train_cifar(args: argparse.Namespace) -> int:
    extra = [
        "--epochs",
        str(args.epochs),
        "--train-subset",
        str(args.subset),
        "--batch-size",
        str(args.batch_size),
    ]
    return _run_script("train_cifar10_proxy.py", extra)


def cmd_train_image(args: argparse.Namespace) -> int:
    extra = [
        "--epochs",
        str(args.epochs),
        "--train-subset",
        str(args.subset),
        "--batch-size",
        str(args.batch_size),
        "--channels",
        str(args.channels),
        "--seed",
        str(args.seed),
    ]
    if args.approx_sign:
        extra.append("--approx-sign")
    if args.include_vit:
        extra.append("--include-vit")
    if args.out:
        extra += ["--out", str(args.out)]
    return _run_script("train_image.py", extra)


def cmd_train_audio(args: argparse.Namespace) -> int:
    extra = [
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--n-train",
        str(args.n_train),
        "--n-test",
        str(args.n_test),
        "--n-classes",
        str(args.n_classes),
        "--channels",
        str(args.channels),
        "--seed",
        str(args.seed),
    ]
    if args.approx_sign:
        extra.append("--approx-sign")
    if args.out:
        extra += ["--out", str(args.out)]
    return _run_script("train_audio.py", extra)


def cmd_repro(args: argparse.Namespace) -> int:
    extra = ["--mode", args.mode]
    if args.overwrite_goldens:
        extra.append("--overwrite-goldens")
    if args.skip_compile:
        extra.append("--skip-compile")
    if args.skip_pytest:
        extra.append("--skip-pytest")
    return _run_script("repro_all.py", extra)


def cmd_wrap(args: argparse.Namespace) -> int:
    """Legacy wide-MLP wrap, or ``--ultra`` for hybrid/calib/ternary demo."""
    if getattr(args, "ultra", False):
        extra: list[str] = [
            "--policy",
            args.policy,
            "--batch",
            str(args.batch),
            "--d-model",
            str(getattr(args, "d_model", 512)),
            "--ff",
            str(getattr(args, "ff", 2048)),
            "--calib-batches",
            str(args.calib_batches),
            "--min-width",
            str(args.min_width),
            "--qat-steps",
            str(args.qat_steps),
            "--drop-in-threshold",
            str(args.drop_in_threshold),
        ]
        # Only forward --mode when explicit; let policy=auto pick mode.
        if args.mode and args.mode != "auto" and args.policy != "auto":
            extra += ["--mode", args.mode]
        elif args.mode == "auto" or args.policy == "auto":
            extra += ["--mode", "auto"]
        elif args.mode:
            extra += ["--mode", args.mode]
        if args.force:
            extra.append("--force")
        if args.report:
            extra += ["--report", str(args.report)]
        if args.compare_baseline:
            extra.append("--compare-baseline")
        return _run_script("ultra_wrap_demo.py", extra)

    extra = [
        "--mode",
        args.mode if args.mode != "auto" else "binary_xnor",
        "--hidden",
        str(args.hidden),
        "--batch",
        str(args.batch),
    ]
    return _run_script("wrap_existing_demo.py", extra)


def cmd_energy_bound(_: argparse.Namespace) -> int:
    return _run_script("energy_bound_measured.py")


def cmd_eval_suite(args: argparse.Namespace) -> int:
    extra = ["--out", str(args.out)]
    if args.full:
        extra.append("--full")
    if args.skip_pytest:
        extra.append("--skip-pytest")
    return _run_script("run_eval_suite.py", extra)


def cmd_recommend(args: argparse.Namespace) -> int:
    return _run_script("recommend_stack.py", ["--goal", args.goal])


def cmd_version(_: argparse.Namespace) -> int:
    print(f"bnn {__version__}")
    return 0


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

    img = sub.add_parser("train-image", help="Image lane: CIFAR-10 FP vs Bi-Real (+ optional ViT)")
    img.add_argument("--epochs", type=int, default=8)
    img.add_argument("--subset", type=int, default=30000, help="0 = full 50k")
    img.add_argument("--batch-size", type=int, default=128)
    img.add_argument("--channels", type=int, default=64)
    img.add_argument("--seed", type=int, default=0)
    img.add_argument("--approx-sign", action="store_true")
    img.add_argument("--include-vit", action="store_true")
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

    sub.add_parser("energy-bound", help="Bind energy estimate to wrap latencies").set_defaults(
        func=cmd_energy_bound
    )

    e = sub.add_parser("eval-suite", help="Run gates + regenerate SUMMARY.md")
    e.add_argument("--out", type=Path, default=ROOT / "results" / "SUMMARY.md")
    e.add_argument("--full", action="store_true", help="Include short image/audio smokes")
    e.add_argument("--skip-pytest", action="store_true")
    e.set_defaults(func=cmd_eval_suite)

    r = sub.add_parser("recommend", help="Recommend stack for a deployment goal")
    r.add_argument(
        "--goal",
        required=True,
        choices=["gpu-server", "cpu-llm", "edge-vision", "npu-phone", "research-xnor", "diffusion"],
    )
    r.set_defaults(func=cmd_recommend)

    ver = sub.add_parser("version", help="Print package version")
    ver.set_defaults(func=cmd_version)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        # argparse already printed help/errors; normalize to int exit
        code = exc.code
        return int(code) if isinstance(code, int) else (1 if code else 0)
    try:
        return int(args.func(args))
    except FileNotFoundError as exc:
        print(f"ERROR missing file: {exc}", file=sys.stderr, flush=True)
        return 2
    except RuntimeError as exc:
        print(f"ERROR {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
