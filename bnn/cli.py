"""Console entry point: ``bnn <command> ...``."""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType

from bnn._version import __version__

ROOT = Path(__file__).resolve().parents[1]

EPILOG = """
Thesis: packed binary/ternary kernels for CPU/edge inference.
Training (STE) is simulation — not a GPU 32× claim.
Reproduce:  bnn repro
Agents:     see AGENTS.md
Docs:       REPRODUCIBILITY.md
""".strip()


def _script_path(script: str) -> Path:
    """Resolve a bare ``*.py`` name under ``scripts/`` (no path separators)."""
    if Path(script).name != script or not script.endswith(".py"):
        raise FileNotFoundError(f"refusing non-basename script: {script!r}")
    scripts_root = (ROOT / "scripts").resolve()
    path = (scripts_root / script).resolve()
    if not path.is_relative_to(scripts_root) or not path.is_file():
        raise FileNotFoundError(path)
    return path


def _load_script_module(path: Path) -> ModuleType:
    """Import ``path`` once as ``bnn._scripts.<stem>``; undo cache on failure."""
    mod_name = f"bnn._scripts.{path.stem}"
    cached = sys.modules.get(mod_name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load script {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(mod_name, None)
        raise
    return module


def _exit_code(result: object) -> int:
    """Map ``main()`` returns / ``SystemExit.code`` to a process exit status."""
    if result is None or result is True:
        return 0
    if result is False:
        return 1
    if isinstance(result, int):
        return result
    return 1 if result else 0


@contextmanager
def _temporary_argv(argv: list[str]) -> Iterator[None]:
    previous = sys.argv
    sys.argv = argv
    try:
        yield
    finally:
        sys.argv = previous


def _call_script_main(main: Callable[..., object], path: Path, argv: list[str]) -> int:
    """Invoke ``main``, supporting either ``main(argv)`` or ``parse_args()``-style mains."""
    if "argv" in inspect.signature(main).parameters:
        return _exit_code(main(argv))
    with _temporary_argv([str(path), *argv]):
        return _exit_code(main())


def _run_script(script: str, extra: list[str] | None = None) -> int:
    """Run ``scripts/<script>`` in-process (testable; same exit-code contract as subprocess).

    ``main(argv)`` gets ``extra`` directly. Otherwise ``sys.argv`` is rewritten for
    scripts that call ``argparse.parse_args()`` with no arguments.
    """
    argv = list(extra or [])
    path = _script_path(script)
    print(f"> scripts/{path.name}", *argv, flush=True)
    try:
        main = getattr(_load_script_module(path), "main", None)
        if not callable(main):
            raise RuntimeError(f"{path.name} has no callable main()")
        return _call_script_main(main, path, argv)
    except SystemExit as exc:
        return _exit_code(exc.code)
    except Exception as exc:
        print(f"ERROR {path.name}: {exc}", file=sys.stderr, flush=True)
        return 1


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


def _ultra_wrap_extra(args: argparse.Namespace) -> list[str]:
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
        str(getattr(args, "calib_batches", 4)),
        "--min-width",
        str(args.min_width),
        "--qat-steps",
        str(args.qat_steps),
        "--drop-in-threshold",
        str(args.drop_in_threshold),
    ]
    # Forward mode as chosen on the CLI. Never let policy=auto clobber an
    # explicit non-auto mode (matches scripts/ultra_wrap_demo.py).
    extra += ["--mode", args.mode]
    if args.force:
        extra.append("--force")
    if getattr(args, "report", None):
        extra += ["--report", str(args.report)]
    if getattr(args, "compare_baseline", False):
        extra.append("--compare-baseline")
    return extra


def cmd_optimise(args: argparse.Namespace) -> int:
    """Product verb: ultra wrap (+ optional encode). Preferred over ``bnn wrap --ultra``."""
    report = args.report or (ROOT / "results" / "optimise_report.json")
    args.report = Path(report)
    code = _run_script("ultra_wrap_demo.py", _ultra_wrap_extra(args))
    if code != 0:
        return code
    if getattr(args, "pack", None):
        enc_ns = argparse.Namespace(
            source="mlp",
            out=Path(args.pack),
            hidden=getattr(args, "pack_hidden", 256),
            min_width=args.min_width,
            in_features=512,
            out_features=512,
        )
        return cmd_encode(enc_ns)
    return 0


def cmd_wrap(args: argparse.Namespace) -> int:
    """Legacy wide-MLP wrap, or ``--ultra`` for hybrid/calib/ternary demo."""
    if getattr(args, "ultra", False):
        return _run_script("ultra_wrap_demo.py", _ultra_wrap_extra(args))

    import warnings

    warnings.warn(
        "bnn wrap without --ultra is the legacy demo; prefer `bnn optimise` "
        "(see docs/adr/0001_public_optimiser_api.md).",
        DeprecationWarning,
        stacklevel=2,
    )
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
    if getattr(args, "strict_budgets", False):
        extra.append("--strict-budgets")
    return _run_script("run_eval_suite.py", extra)


def cmd_recommend(args: argparse.Namespace) -> int:
    return _run_script("recommend_stack.py", ["--goal", args.goal])


def cmd_encode(args: argparse.Namespace) -> int:
    """Encode toy MLP BinaryLinear layers or a random Linear into ``.bnnpack``."""
    import torch

    from bnn.codec import encode_file, encode_linear_state, save_bnnpack
    from bnn.models import build_model

    out = Path(args.out)
    meta = {"source": args.source, "cli": "bnn encode"}
    if args.source == "mlp":
        model = build_model("binary_mlp", hidden=args.hidden)
        # Only BinaryLinear (not FP stem/head) — thesis-aligned
        encode_file(
            model,
            out,
            meta=meta,
            min_in_features=args.min_width,
            include_binary_linear=True,
            include_fp_linear=False,
            include_packed=True,
        )
    elif args.source == "random":
        w = torch.randn(args.out_features, args.in_features)
        blob = encode_linear_state(w, name="linear")
        save_bnnpack({"linear": blob}, out, meta=meta)
    else:
        print(f"ERROR unknown source {args.source}", file=sys.stderr)
        return 2
    from bnn.codec import load_bnnpack

    payload = load_bnnpack(out)
    fp = sum(int(b["fp32_bytes"]) for b in payload["layers"].values())
    pk = sum(int(b["packed_bytes"]) for b in payload["layers"].values())
    print(
        f"Wrote {out} layers={len(payload['layers'])} "
        f"fp32_bytes={fp} packed_bytes={pk} compression={fp / max(pk, 1):.2f}x"
    )
    return 0


def cmd_decode(args: argparse.Namespace) -> int:
    """Load ``.bnnpack`` and verify layers (GEMM err=0 for binary_xnor only)."""
    import torch

    from bnn.codec import decode_file, packed_module_fp_err
    from bnn.wrap.packed_linear import (
        PackedBinaryConv2d,
        PackedBinaryXNORLinear,
        TernaryWeightOnlyLinear,
    )

    path = Path(args.pack)
    modules, meta = decode_file(path)
    print(f"Loaded {path} layers={list(modules)} meta_keys={list(meta)}")
    max_err = 0.0
    for name, mod in modules.items():
        if isinstance(mod, PackedBinaryXNORLinear):
            err = packed_module_fp_err(mod, batch=4, seed=0)
            comp = (mod.in_features * mod.out_features * 4) / max(
                mod.packed_weight_bytes(), 1
            )
            print(f"  {name}: kind=binary_xnor fp_err={err} compression~{comp:.2f}x")
            max_err = max(max_err, err)
            continue
        if isinstance(mod, TernaryWeightOnlyLinear):
            x = torch.randn(2, mod.in_features)
            y = mod(x)
            ok = bool(torch.isfinite(y).all()) and y.shape == (2, mod.out_features)
            comp = (mod.in_features * mod.out_features * 4) / max(
                mod.packed_weight_bytes(), 1
            )
            print(
                f"  {name}: kind=ternary_weight_only "
                f"forward_ok={ok} compression~{comp:.2f}x "
                f"(theoretical_2bit; GEMM err=0 N/A)"
            )
            if not ok:
                print(f"ERROR ternary forward check failed for {name}", file=sys.stderr)
                return 1
            continue
        if isinstance(mod, PackedBinaryConv2d):
            h = w = 8
            x = torch.randn(1, mod.in_channels, h, w)
            y = mod(x)
            ok = bool(torch.isfinite(y).all()) and y.shape[1] == mod.out_channels
            kh, kw = mod.kernel_size
            fp = mod.in_channels * mod.out_channels * kh * kw * 4
            comp = fp / max(mod.packed_weight_bytes(), 1)
            print(
                f"  {name}: kind=binary_conv_packed "
                f"forward_ok={ok} compression~{comp:.2f}x "
                f"(size/dequant path; GEMM err=0 N/A)"
            )
            if not ok:
                print(f"ERROR conv forward check failed for {name}", file=sys.stderr)
                return 1
            continue
        print(
            f"ERROR {name}: unsupported module type {type(mod).__name__} "
            "(decode expects binary_xnor / ternary_weight_only / binary_conv_packed)",
            file=sys.stderr,
        )
        return 1
    if max_err > 0:
        print(f"ERROR non-zero packed vs FP err={max_err}", file=sys.stderr)
        return 1
    print("DECODE: PASS")
    return 0


def cmd_profile(args: argparse.Namespace) -> int:
    import json

    from bnn.profile import check_soft_budgets, profile_packed_linear

    br = profile_packed_linear(
        m=args.batch,
        n=args.in_features,
        k=args.out_features,
        reps=args.reps,
        warmup=args.warmup,
        compare_baselines=not getattr(args, "no_baselines", False),
    )
    d = br.to_dict()
    soft = check_soft_budgets(br)
    d["soft_budget_ok"] = not soft
    if soft:
        d["soft_budget_violations"] = soft
        print("WARN soft latency budget:", "; ".join(soft), file=sys.stderr)
    print(json.dumps(d, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(d, indent=2), encoding="utf-8")
    return 0


def cmd_memory(args: argparse.Namespace) -> int:
    """Weight footprint of a demo model, FP32 vs wrapped (W13.T05)."""
    import json

    import torch.nn as nn

    from bnn.memory import forward_transient_bytes, memory_report
    from bnn.wrap import wrap_model

    dim, ff = args.dim, args.ff

    def build() -> nn.Module:
        return nn.Sequential(
            nn.Linear(dim, ff),
            nn.ReLU(),
            nn.Linear(ff, dim),
        )

    fp32 = memory_report(build()).to_dict()
    wrapped, _report = wrap_model(build(), mode=args.mode, policy="all_large_linear")
    packed = memory_report(wrapped).to_dict()

    out = {
        "schema": "bnn_memory_report_v1",
        "shape": {"dim": dim, "ff": ff, "batch": args.batch},
        "mode": args.mode,
        "fp32": fp32,
        "wrapped": packed,
        "transient_per_forward": forward_transient_bytes(args.batch, dim, ff),
        "thesis_note": (
            "Weight bytes are measured from real buffers; theoretical_* is the "
            "encoding pack ratio. Neither is a latency claim."
        ),
    }
    print(json.dumps(out, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0


def cmd_train_seq2seq(args: argparse.Namespace) -> int:
    extra = [
        "--task",
        args.task,
        "--ffn",
        args.ffn,
        "--steps",
        str(args.steps),
        "--batch",
        str(args.batch),
        "--seq-len",
        str(args.seq_len),
        "--dim",
        str(args.dim),
        "--seed",
        str(args.seed),
    ]
    if args.out:
        extra += ["--out", str(args.out)]
    return _run_script("train_seq2seq.py", extra)


def cmd_wrap_transformer(args: argparse.Namespace) -> int:
    extra = [
        "--d-model",
        str(args.d_model),
        "--ff",
        str(args.ff),
        "--depth",
        str(args.depth),
        "--batch",
        str(args.batch),
        "--qat-steps",
        str(args.qat_steps),
        "--policy",
        args.policy,
    ]
    if args.out:
        extra += ["--out", str(args.out)]
    return _run_script("tiny_transformer_wrap_demo.py", extra)


def cmd_version(_: argparse.Namespace) -> int:
    print(f"bnn {__version__}")
    return 0


def cmd_kg(args: argparse.Namespace) -> int:
    """Validate or summarize the lab knowledge graph."""
    from bnn.kg import clear_kg_cache, load_kg, nodes_by_type, validate_graph

    clear_kg_cache()
    g = load_kg()
    if args.action == "validate":
        errs = validate_graph(g)
        meta = g.get("meta", {})
        print(
            f"KG: {meta.get('node_count', len(g['nodes']))} nodes, "
            f"{meta.get('edge_count', len(g['edges']))} edges"
        )
        if errs:
            print("FAIL:")
            for e in errs:
                print(f" - {e}")
            return 1
        print("KG: PASS")
        return 0

    # summary
    meta = g.get("meta", {})
    print(
        f"BNN knowledge graph v{meta.get('version', '?')}: "
        f"{meta.get('node_count', len(g['nodes']))} nodes / "
        f"{meta.get('edge_count', len(g['edges']))} edges"
    )
    print("View: knowledge_graph/VIEW.md  ·  docs: docs/44_KNOWLEDGE_GRAPH.md")
    print("Recommend: bnn recommend --goal <gpu-server|cpu-llm|edge-vision|…>")
    print("Eval:      bnn eval-suite [--skip-pytest]")
    gaps = nodes_by_type(g, "OpenGap")
    openish = [n for n in gaps if str(n.get("status", "")).startswith(("open", "deferred"))]
    print(f"OpenGaps: {len(gaps)} ({len(openish)} open/deferred/open_pr)")
    for n in sorted(openish, key=lambda x: x["id"])[:8]:
        print(f"  - {n['id']} [{n.get('status')}]")
    if len(openish) > 8:
        print(f"  … +{len(openish) - 8} more")
    return 0


def cmd_pareto(args: argparse.Namespace) -> int:
    """Emit dual-metric Pareto JSON (W7.T03)."""
    extra = ["--out", str(args.out)]
    if args.demo:
        extra.append("--demo")
    for path in args.from_optimise or []:
        extra += ["--from-optimise", str(path)]
    if args.plot:
        extra += ["--plot", str(args.plot)]
    if args.warmup is not None:
        extra += ["--warmup", str(args.warmup)]
    if args.threads is not None:
        extra += ["--threads", str(args.threads)]
    if not args.demo and not (args.from_optimise or []):
        extra.append("--demo")
    return _run_script("pareto_report.py", extra)


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
        "--out",
        type=Path,
        default=ROOT / "results" / "pareto_report.json",
    )
    pa.add_argument("--plot", type=Path, default=None, help="Optional PNG (matplotlib)")
    pa.add_argument("--warmup", type=int, default=3)
    pa.add_argument("--threads", type=int, default=None)
    pa.set_defaults(func=cmd_pareto)

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

def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        # argparse already printed help/errors; normalize to int exit
        return _exit_code(exc.code)
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
