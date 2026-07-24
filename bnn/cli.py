"""Console entry point: ``bnn <command> ...``."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run_script(script: str, extra: list[str] | None = None) -> int:
    cmd = [sys.executable, str(ROOT / "scripts" / script), *(extra or [])]
    print(">", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(ROOT))


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


def cmd_wrap(args: argparse.Namespace) -> int:
    extra = [
        "--mode",
        args.mode,
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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bnn",
        description=(
            "Extreme low-bit inference lab. Training is STE/simulation (not faster). "
            "Inference wins need packed kernels on CPU/edge — not sign() on GPU."
        ),
    )
    sub = p.add_subparsers(dest="command", required=True)

    cn = sub.add_parser("compile-native", help="Build MSVC x64 popcount DLL")
    cn.add_argument("--force", action="store_true")
    cn.set_defaults(func=cmd_compile_native)
    sub.add_parser("validate-native", help="err=0 vs ±1 FP GEMM").set_defaults(
        func=cmd_validate_native
    )
    b = sub.add_parser("bench", help="Kernel benchmark")
    b.add_argument("--reps", type=int, default=None)
    b.set_defaults(func=cmd_bench)

    sub.add_parser("export-check", help="Compression ~32× assert").set_defaults(
        func=cmd_export_check
    )

    t = sub.add_parser("train", help="MNIST train (STE sim — not a throughput win)")
    t.add_argument("--epochs", type=int, default=3)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--model", type=str, default=None, help="Optional single model name")
    t.add_argument("--threads", type=int, default=None)
    t.set_defaults(func=cmd_train)

    c = sub.add_parser("train-cifar", help="CIFAR-10 Bi-Real proxy")
    c.add_argument("--epochs", type=int, default=5)
    c.add_argument("--subset", type=int, default=20000)
    c.add_argument("--batch-size", type=int, default=128)
    c.set_defaults(func=cmd_train_cifar)

    w = sub.add_parser("wrap", help="Wrap demo MLP with packed Linears")
    w.add_argument(
        "--mode",
        default="binary_xnor",
        choices=["binary_xnor", "ternary_weight_only", "binary_weight_only_dequant"],
    )
    w.add_argument("--hidden", type=int, default=4096)
    w.add_argument("--batch", type=int, default=32)
    w.set_defaults(func=cmd_wrap)

    sub.add_parser("energy-bound", help="Bind energy estimate to wrap latencies").set_defaults(
        func=cmd_energy_bound
    )

    e = sub.add_parser("eval-suite", help="Run gates + regenerate SUMMARY.md")
    e.add_argument("--out", type=Path, default=ROOT / "results" / "SUMMARY.md")
    e.add_argument("--full", action="store_true", help="Include CIFAR proxy")
    e.add_argument("--skip-pytest", action="store_true")
    e.set_defaults(func=cmd_eval_suite)

    r = sub.add_parser("recommend", help="Recommend stack for a goal")
    r.add_argument(
        "--goal",
        required=True,
        choices=["gpu-server", "cpu-llm", "edge-vision", "npu-phone", "research-xnor", "diffusion"],
    )
    r.set_defaults(func=cmd_recommend)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
