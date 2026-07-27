#!/usr/bin/env python3
"""Wrap an existing FP MLP: size, e2e latency, and isolated Linear microbench."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bnn.wrapper import (  # noqa: E402
    BinaryWeightOnlyDequantLinear,
    PackedBinaryXNORLinear,
    TernaryWeightOnlyLinear,
    WrapReport,
    model_param_bytes,
)


def time_fn(fn, warmup: int = 5, reps: int = 30) -> float:
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps


def make_wide_mlp(hidden: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, hidden),  # 1 stem — keep FP
        nn.ReLU(inplace=True),
        nn.Linear(hidden, hidden),  # 3
        nn.ReLU(inplace=True),
        nn.Linear(hidden, hidden),  # 5
        nn.ReLU(inplace=True),
        nn.Linear(hidden, 10),  # 7 head — keep FP
    )


def replace_middles(model: nn.Sequential, mode: str) -> WrapReport:
    report = WrapReport(mode=mode)  # type: ignore[arg-type]
    for idx in (3, 5):
        lin = model[idx]
        assert isinstance(lin, nn.Linear)
        w, b = lin.weight.data, lin.bias.data if lin.bias is not None else None
        fp_bytes = int(w.numel() * 4)
        if mode == "binary_xnor":
            new = PackedBinaryXNORLinear(w, b)
            packed = new.packed_weight_bytes()
            report.native_kernel = new.uses_native
        elif mode == "ternary_weight_only":
            new = TernaryWeightOnlyLinear(w, b)
            packed = new.packed_weight_bytes()
        else:
            new = BinaryWeightOnlyDequantLinear(w, b)
            packed = new.packed_weight_bytes()
        model[idx] = new
        report.replaced.append(str(idx))
        report.fp32_weight_bytes_replaced += fp_bytes
        report.packed_weight_bytes += packed
    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--mode",
        choices=["binary_xnor", "ternary_weight_only", "binary_weight_only_dequant"],
        default="binary_xnor",
    )
    p.add_argument("--hidden", type=int, default=4096)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--out", type=Path, default=ROOT / "results" / "wrap_demo.json")
    args = p.parse_args()

    from bnn.determinism import set_repro_seed

    set_repro_seed(0, deterministic=True, force_cpu=True)
    fp_model = make_wide_mlp(args.hidden)
    wrapped = copy.deepcopy(fp_model)
    report = replace_middles(wrapped, args.mode)

    x = torch.randn(args.batch, 1, 28, 28)
    t_fp = time_fn(lambda: fp_model(x))
    t_wrap = time_fn(lambda: wrapped(x))

    # Isolated middle Linear: torch FP vs packed (fair kernel comparison)
    lin_fp = fp_model[3]
    assert isinstance(lin_fp, nn.Linear)
    h = torch.randn(args.batch, args.hidden)
    t_lin_fp = time_fn(lambda: lin_fp(h))
    layer_micro = {}
    if args.mode == "binary_xnor":
        lin_w = wrapped[3]
        assert isinstance(lin_w, PackedBinaryXNORLinear)
        h_pm1 = np.where(h.numpy() >= 0, 1.0, -1.0).astype(np.float32)
        # Include act pack + gemm (realistic layer)
        t_lin_wrap = time_fn(lambda: lin_w(h))
        t_gemm_only = time_fn(lambda: lin_w.gemm_only(h_pm1))
        layer_micro = {
            "linear_fp_ms": t_lin_fp * 1e3,
            "linear_wrapped_forward_ms": t_lin_wrap * 1e3,
            "linear_gemm_only_prepacked_acts_ms": t_gemm_only * 1e3,
            "speedup_gemm_only_vs_torch_linear": t_lin_fp / t_gemm_only if t_gemm_only else None,
            "speedup_wrapped_linear_vs_torch": t_lin_fp / t_lin_wrap if t_lin_wrap else None,
        }
    else:
        lin_w = wrapped[3]
        t_lin_wrap = time_fn(lambda: lin_w(h))
        layer_micro = {
            "linear_fp_ms": t_lin_fp * 1e3,
            "linear_wrapped_forward_ms": t_lin_wrap * 1e3,
            "speedup_wrapped_linear_vs_torch": t_lin_fp / t_lin_wrap if t_lin_wrap else None,
        }

    with torch.no_grad():
        cos = float(
            torch.nn.functional.cosine_similarity(
                fp_model(x).flatten(), wrapped(x).float().cpu().flatten(), dim=0
            ).item()
        )

    size_fp = model_param_bytes(fp_model)
    size_w = model_param_bytes(wrapped)
    result = {
        "mode": args.mode,
        "hidden": args.hidden,
        "batch": args.batch,
        "replaced_layers": report.replaced,
        "weight_compression_replaced_layers": report.compression,
        "fp32_weight_bytes_replaced": report.fp32_weight_bytes_replaced,
        "packed_weight_bytes_reported": report.packed_weight_bytes,
        "model_total_bytes_fp": size_fp["total_bytes"],
        "model_total_bytes_wrapped": size_w["total_bytes"],
        "e2e_latency_ms_fp": t_fp * 1e3,
        "e2e_latency_ms_wrapped": t_wrap * 1e3,
        "e2e_speedup": (t_fp / t_wrap) if t_wrap else None,
        "layer_microbench": layer_micro,
        "output_cosine_vs_fp": cos,
        "native_kernel": report.native_kernel,
        "interpretation": (
            "E2E may lose to torch when stem/head/ReLU dominate or Python pack overhead "
            "is large; layer gemm_only shows true kernel ROI. Cosine<<1 for binary_xnor "
            "without QAT is expected (not a transparent accuracy-preserving wrap)."
        ),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    lm = layer_micro
    md_lines = [
        "# Wrap existing model demo",
        "",
        f"- Mode: `{args.mode}` hidden={args.hidden} batch={args.batch}",
        f"- Replaced: {report.replaced}",
        f"- Weight compression (replaced): **{report.compression:.2f}×**",
        f"- Model bytes: {size_fp['total_bytes']} → {size_w['total_bytes']}",
        f"- E2E latency: {t_fp*1e3:.2f} ms → {t_wrap*1e3:.2f} ms "
        f"(**{(t_fp/t_wrap) if t_wrap else 0:.2f}×**)",
        f"- Output cosine vs FP: **{cos:.4f}**",
    ]
    if "speedup_gemm_only_vs_torch_linear" in lm:
        md_lines += [
            f"- Layer micro: torch Linear {lm['linear_fp_ms']:.2f} ms | "
            f"wrapped fwd {lm['linear_wrapped_forward_ms']:.2f} ms | "
            f"gemm_only {lm['linear_gemm_only_prepacked_acts_ms']:.2f} ms",
            f"- **Kernel speedup (gemm_only vs torch Linear): "
            f"{lm['speedup_gemm_only_vs_torch_linear']:.2f}×**",
        ]
    md_lines += ["", result["interpretation"], ""]
    args.out.with_suffix(".md").write_text("\n".join(md_lines), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
