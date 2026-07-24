#!/usr/bin/env python3
"""Benchmark FP32 GEMM vs packed binary XNOR+popcount (CPU).

Fair inference-style protocol:
  - Weights pre-packed once (deploy-time)
  - Activations packed per forward (included in 'packed_e2e')
  - Also report compute-only native time (prepacked X and W)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bnn.kernels.packed import (  # noqa: E402
    binary_gemm_native_prepacked,
    binary_gemm_numpy_prepacked,
    fp32_gemm,
    native_kernel_available,
    pack_binary_pm1,
    theoretical_ops,
)


def time_fn(fn, warmup: int = 3, reps: int = 10) -> float:
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps


def bench_size(batch: int, n: int, m: int, reps: int) -> dict:
    rng = np.random.default_rng(0)
    x = rng.choice([-1.0, 1.0], size=(batch, n)).astype(np.float32)
    w = rng.choice([-1.0, 1.0], size=(m, n)).astype(np.float32)

    wp, n_dim = pack_binary_pm1(w, axis=1)
    xp, _ = pack_binary_pm1(x, axis=1)

    y_fp = fp32_gemm(x, w)
    use_native = native_kernel_available()
    if use_native:
        y_bin = binary_gemm_native_prepacked(xp, wp, n_dim)
        assert y_bin is not None
    else:
        y_bin = binary_gemm_numpy_prepacked(xp, wp, n_dim)
    max_err = float(np.max(np.abs(y_fp - y_bin)))

    t_fp = time_fn(lambda: fp32_gemm(x, w), reps=reps)

    def compute_only():
        if use_native:
            return binary_gemm_native_prepacked(xp, wp, n_dim)
        return binary_gemm_numpy_prepacked(xp, wp, n_dim)

    t_compute = time_fn(compute_only, reps=reps)

    def e2e_act_pack():
        xpl, _ = pack_binary_pm1(x, axis=1)
        if use_native:
            return binary_gemm_native_prepacked(xpl, wp, n_dim)
        return binary_gemm_numpy_prepacked(xpl, wp, n_dim)

    t_e2e = time_fn(e2e_act_pack, reps=reps)

    xt = torch.from_numpy(x)
    wt = torch.from_numpy(w)

    def fake_binary():
        return torch.nn.functional.linear(xt.sign(), wt.sign())

    def torch_fp():
        return torch.nn.functional.linear(xt, wt)

    t_fake = time_fn(fake_binary, reps=reps)
    t_torch = time_fn(torch_fp, reps=reps)

    ops = theoretical_ops(batch, n, m)
    return {
        "shape": {"batch": batch, "in_features": n, "out_features": m},
        "native_kernel": use_native,
        "max_abs_error_vs_fp32": max_err,
        "seconds": {
            "numpy_fp32_gemm": t_fp,
            "binary_compute_only_prepacked": t_compute,
            "binary_e2e_with_act_pack": t_e2e,
            "torch_fp32_linear": t_torch,
            "torch_fake_binary_sign_linear": t_fake,
        },
        "speedup_compute_vs_numpy_fp32": t_fp / t_compute if t_compute else None,
        "speedup_e2e_vs_numpy_fp32": t_fp / t_e2e if t_e2e else None,
        "speedup_compute_vs_torch_fp32": t_torch / t_compute if t_compute else None,
        "fake_binary_vs_torch_fp32": t_fake / t_torch if t_torch else None,
        "theoretical": ops,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--reps", type=int, default=8)
    p.add_argument(
        "--sizes",
        nargs="+",
        default=["128x2048x2048", "64x4096x4096", "32x8192x8192"],
    )
    p.add_argument("--out", type=Path, default=ROOT / "results" / "benchmark.json")
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    print("native_kernel_available:", native_kernel_available())
    results = []
    for spec in args.sizes:
        b, n, m = map(int, spec.lower().split("x"))
        print(f"Benchmarking batch={b} in={n} out={m} ...")
        row = bench_size(b, n, m, args.reps)
        results.append(row)
        sec = row["seconds"]
        print(
            f"  compute {sec['binary_compute_only_prepacked']*1e3:.2f} ms | "
            f"e2e+actpack {sec['binary_e2e_with_act_pack']*1e3:.2f} ms | "
            f"numpy FP32 {sec['numpy_fp32_gemm']*1e3:.2f} ms | "
            f"S_compute {row['speedup_compute_vs_numpy_fp32']:.2f}x | "
            f"err {row['max_abs_error_vs_fp32']:.3g}"
        )
        print(
            f"  torch FP32 {sec['torch_fp32_linear']*1e3:.2f} ms | "
            f"fake-binary {sec['torch_fake_binary_sign_linear']*1e3:.2f} ms | "
            f"fake/fp {row['fake_binary_vs_torch_fp32']:.2f}"
        )

    payload = {
        "device_note": "CPU; weights pre-packed (inference). Native MSVC popcount when available.",
        "results": results,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    md = args.out.with_suffix(".md")
    lines = [
        "# Kernel benchmark (CPU, fair protocol)",
        "",
        "Weights pre-packed once. `compute` = GEMM only; `e2e` = pack activations + GEMM.",
        "",
        "| Shape | Native? | Compute ms | E2E ms | NumPy FP32 | Torch FP32 | Fake-bin | "
        "S_compute | S_e2e | Fake/FP | Theory↓ | Err |",
        "|-------|---------|------------|--------|------------|------------|----------|"
        "----------|-------|---------|---------|-----|",
    ]
    for r in results:
        s = r["shape"]
        shape = f"{s['batch']}×{s['in_features']}×{s['out_features']}"
        sec = r["seconds"]
        lines.append(
            f"| {shape} | {r['native_kernel']} | "
            f"{sec['binary_compute_only_prepacked']*1e3:.2f} | "
            f"{sec['binary_e2e_with_act_pack']*1e3:.2f} | "
            f"{sec['numpy_fp32_gemm']*1e3:.2f} | "
            f"{sec['torch_fp32_linear']*1e3:.2f} | "
            f"{sec['torch_fake_binary_sign_linear']*1e3:.2f} | "
            f"{r['speedup_compute_vs_numpy_fp32']:.2f}× | "
            f"{r['speedup_e2e_vs_numpy_fp32']:.2f}× | "
            f"{r['fake_binary_vs_torch_fp32']:.2f} | "
            f"{r['theoretical']['theoretical_word_reduction']:.0f}× | "
            f"{r['max_abs_error_vs_fp32']:.3g} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- **S_compute** is the honest kernel win with deployed packed weights.",
        "- **S_e2e** includes activation packing (still usually >1× at large N).",
        "- **Fake-binary > 1** means `sign`+FP GEMM is slower — simulation ≠ acceleration.",
        "- Theory ~64× is word-op reduction, not wall-clock.",
    ]
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.out} and {md}")


if __name__ == "__main__":
    main()
