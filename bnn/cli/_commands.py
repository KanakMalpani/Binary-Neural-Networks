"""CLI command handlers. Thin adapters over library APIs and ``scripts/``.

``_run_script`` / ``_run_bridge`` late-bind through ``bnn.cli`` so tests that
patch ``cli._run_script`` still intercept handler dispatch after the split.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import ModuleType

from bnn._version import __version__

from ._dispatch import BRIDGE_ALIASES, BRIDGE_RECIPES


def _pkg() -> ModuleType:
    """Late-bind the public CLI package (preserves test monkeypatches)."""
    import bnn.cli as cli

    return cli


def _run_script(script: str, extra: list[str] | None = None) -> int:
    return _pkg()._run_script(script, extra)


def _run_bridge(script: str, extra: list[str] | None = None) -> int:
    return _pkg()._run_bridge(script, extra)


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
    if getattr(args, "include_resnet", False):
        extra.append("--include-resnet")
        extra += ["--resnet-width", str(args.resnet_width)]
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
    report = args.report or (_pkg().ROOT / "results" / "optimise_report.json")
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
        return _pkg().cmd_encode(enc_ns)
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
    """Emit dual-metric Pareto JSON (W7.T03 / W12.T03)."""
    extra = ["--out", str(args.out)]
    if args.demo:
        extra.append("--demo")
    for path in args.from_optimise or []:
        extra += ["--from-optimise", str(path)]
    if getattr(args, "from_results", False):
        extra.append("--from-results")
    if args.plot:
        extra += ["--plot", str(args.plot)]
    if args.warmup is not None:
        extra += ["--warmup", str(args.warmup)]
    if args.threads is not None:
        extra += ["--threads", str(args.threads)]
    has_points = (
        args.demo
        or bool(args.from_optimise or [])
        or bool(getattr(args, "from_results", False))
    )
    if not has_points:
        extra.append("--demo")
    return _run_script("pareto_report.py", extra)


def cmd_bridge(args: argparse.Namespace) -> int:
    """First-class production bridges over ``scripts/bridges/*`` (W12)."""
    action = getattr(args, "bridge_action", None)
    if action == "list":
        rows = []
        for key, meta in BRIDGE_RECIPES.items():
            aliases = [a for a, canon in BRIDGE_ALIASES.items() if canon == key]
            alias_note = f" (aliases: {', '.join(aliases)})" if aliases else ""
            rows.append(
                f"  {key:<8} → scripts/bridges/{meta['script']}  "
                f"[{meta['lane']}] {meta['summary']}{alias_note}\n"
                f"           doc: {meta['doc']}"
            )
        print("bnn bridge recipes:\n" + "\n".join(rows))
        print("\nAlso: bnn recommend --goal {gpu-server,cpu-llm,…}")
        return 0

    if action == "figures":
        fig_extra: list[str] = ["--out", str(args.out)] if args.out else []
        if args.plot_dir:
            fig_extra += ["--plot-dir", str(args.plot_dir)]
        return _run_script("figure_from_results.py", fig_extra)

    name = BRIDGE_ALIASES.get(action or "", action or "")
    recipe = BRIDGE_RECIPES.get(name or "")
    if recipe is None:
        print(f"ERROR unknown bridge {action!r}; try: bnn bridge list", file=sys.stderr)
        return 2
    bridge_extra: list[str] = []
    if getattr(args, "probe", False) and name == "gpu":
        bridge_extra.append("--probe")
    if getattr(args, "out", None):
        bridge_extra += ["--out", str(args.out)]
    return _run_bridge(recipe["script"], bridge_extra)

