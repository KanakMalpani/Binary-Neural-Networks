#!/usr/bin/env python3
"""Benchmark FP32 GEMM vs packed binary XNOR+popcount (CPU).

Fair inference-style protocol:
  - Weights pre-packed once (deploy-time)
  - Activations packed per forward (included in 'packed_e2e')
  - Also report compute-only native time (prepacked X and W)
  - Optional thread scaling curve (1/2/4/8)
  - Pack-once timing vs compute
"""

from __future__ import annotations

import argparse
import json
import os
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
    get_num_threads,
    native_kernel_available,
    openmp_enabled,
    pack_binary_pm1,
    set_num_threads,
    theoretical_ops,
)
from bnn.kernels.ternary_gemm import (  # noqa: E402
    ternary_bitplane_gemm_native,
    ternary_bitplane_gemm_numpy,
    ternary_dequant_gemm,
)
from bnn.kernels.ternary_pack import (  # noqa: E402
    pack_ternary_bitplanes,
    precompute_bitplane_pops,
)


def time_fn(fn, warmup: int = 5, reps: int = 10) -> float:
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps


def _parse_threads(spec: str) -> list[int]:
    return [int(x) for x in spec.split(",") if x.strip()]


def bench_size(
    batch: int,
    n: int,
    m: int,
    reps: int,
    warmup: int,
    thread_list: list[int] | None,
) -> dict:
    rng = np.random.default_rng(0)
    x = rng.choice([-1.0, 1.0], size=(batch, n)).astype(np.float32)
    w = rng.choice([-1.0, 1.0], size=(m, n)).astype(np.float32)

    # Prefer explicit BNN_NUM_THREADS; else a moderate default (avoid oversubscription).
    env_threads = os.environ.get("BNN_NUM_THREADS")
    if env_threads and env_threads.strip().isdigit():
        default_threads = max(1, int(env_threads))
    else:
        default_threads = min(8, max(1, (os.cpu_count() or 4) // 2))
    if native_kernel_available():
        set_num_threads(default_threads)

    t_pack_w = time_fn(lambda: pack_binary_pm1(w, axis=1), warmup=2, reps=max(3, reps // 2))
    wp, n_dim = pack_binary_pm1(w, axis=1)
    t_pack_x = time_fn(lambda: pack_binary_pm1(x, axis=1), warmup=2, reps=max(3, reps // 2))
    xp, _ = pack_binary_pm1(x, axis=1)

    y_fp = fp32_gemm(x, w)
    use_native = native_kernel_available()
    if use_native:
        y_bin = binary_gemm_native_prepacked(xp, wp, n_dim)
        assert y_bin is not None
    else:
        y_bin = binary_gemm_numpy_prepacked(xp, wp, n_dim)
    max_err = float(np.max(np.abs(y_fp - y_bin)))

    t_fp = time_fn(lambda: fp32_gemm(x, w), warmup=warmup, reps=reps)

    def compute_only():
        if use_native:
            return binary_gemm_native_prepacked(xp, wp, n_dim)
        return binary_gemm_numpy_prepacked(xp, wp, n_dim)

    t_compute = time_fn(compute_only, warmup=warmup, reps=reps)

    def e2e_act_pack():
        xpl, _ = pack_binary_pm1(x, axis=1)
        if use_native:
            return binary_gemm_native_prepacked(xpl, wp, n_dim)
        return binary_gemm_numpy_prepacked(xpl, wp, n_dim)

    t_e2e = time_fn(e2e_act_pack, warmup=warmup, reps=reps)

    xt = torch.from_numpy(x)
    wt = torch.from_numpy(w)

    def fake_binary():
        return torch.nn.functional.linear(xt.sign(), wt.sign())

    def torch_fp():
        return torch.nn.functional.linear(xt, wt)

    t_fake = time_fn(fake_binary, warmup=warmup, reps=reps)
    t_torch = time_fn(torch_fp, warmup=warmup, reps=reps)

    # Ternary microbench (same shape): bitplane vs dequant FP
    q = rng.integers(-1, 2, size=(m, n), dtype=np.int8)
    scale = 0.5
    twp, twn, _ = pack_ternary_bitplanes(q)
    pop_p, pop_n = precompute_bitplane_pops(twp, twn)
    y_t_ref = ternary_dequant_gemm(x, q, scale)
    y_t_np = ternary_bitplane_gemm_numpy(xp, twp, twn, scale, pop_p, pop_n)
    y_t_nat = ternary_bitplane_gemm_native(xp, twp, twn, scale, pop_p, pop_n)
    t_err = float(np.max(np.abs(y_t_ref - y_t_np)))
    if y_t_nat is not None:
        t_err = max(t_err, float(np.max(np.abs(y_t_ref - y_t_nat))))

    def tern_fast():
        if y_t_nat is not None:
            return ternary_bitplane_gemm_native(xp, twp, twn, scale, pop_p, pop_n)
        return ternary_bitplane_gemm_numpy(xp, twp, twn, scale, pop_p, pop_n)

    t_tern_fast = time_fn(tern_fast, warmup=warmup, reps=reps)
    t_tern_fp = time_fn(lambda: ternary_dequant_gemm(x, q, scale), warmup=warmup, reps=reps)

    thread_scaling = []
    if thread_list and use_native:
        for nth in thread_list:
            set_num_threads(nth)
            t_nth = time_fn(compute_only, warmup=warmup, reps=reps)
            thread_scaling.append(
                {
                    "threads": nth,
                    "seconds_compute": t_nth,
                    "speedup_vs_1thread": None,  # filled below
                    "reported_get_num_threads": get_num_threads(),
                }
            )
        # Restore moderate default after scaling sweep
        set_num_threads(default_threads)
        if thread_scaling:
            base = thread_scaling[0]["seconds_compute"]
            for row in thread_scaling:
                row["speedup_vs_1thread"] = (
                    base / row["seconds_compute"] if row["seconds_compute"] else None
                )

    ops = theoretical_ops(batch, n, m)
    return {
        "shape": {"batch": batch, "in_features": n, "out_features": m},
        "native_kernel": use_native,
        "openmp": openmp_enabled(),
        "bench_threads": default_threads if use_native else 1,
        "max_abs_error_vs_fp32": max_err,
        "seconds": {
            "numpy_fp32_gemm": t_fp,
            "binary_compute_only_prepacked": t_compute,
            "binary_e2e_with_act_pack": t_e2e,
            "pack_weights_once": t_pack_w,
            "pack_activations": t_pack_x,
            "torch_fp32_linear": t_torch,
            "torch_fake_binary_sign_linear": t_fake,
            "ternary_bitplane_compute": t_tern_fast,
            "ternary_dequant_fp_gemm": t_tern_fp,
        },
        "ternary_max_abs_error_vs_dequant": t_err,
        "speedup_compute_vs_numpy_fp32": t_fp / t_compute if t_compute else None,
        "speedup_e2e_vs_numpy_fp32": t_fp / t_e2e if t_e2e else None,
        "speedup_compute_vs_torch_fp32": t_torch / t_compute if t_compute else None,
        "fake_binary_vs_torch_fp32": t_fake / t_torch if t_torch else None,
        "speedup_ternary_bitplane_vs_dequant": t_tern_fp / t_tern_fast if t_tern_fast else None,
        "thread_scaling": thread_scaling,
        "theoretical": ops,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument(
        "--sizes",
        nargs="+",
        default=["128x2048x2048", "64x4096x4096", "32x8192x8192"],
    )
    p.add_argument(
        "--threads",
        type=str,
        default="1,2,4,8",
        help="Comma list for thread scaling (empty to skip)",
    )
    p.add_argument("--out", type=Path, default=ROOT / "results" / "benchmark.json")
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    thread_list = _parse_threads(args.threads) if args.threads.strip() else None
    print("native_kernel_available:", native_kernel_available())
    print("openmp_enabled:", openmp_enabled())
    print("thread_scaling:", thread_list)

    results = []
    for spec in args.sizes:
        b, n, m = map(int, spec.lower().split("x"))
        print(f"Benchmarking batch={b} in={n} out={m} ...")
        row = bench_size(b, n, m, args.reps, args.warmup, thread_list)
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
            f"  pack_W {sec['pack_weights_once']*1e3:.2f} ms | "
            f"pack_X {sec['pack_activations']*1e3:.2f} ms | "
            f"ternary bitplane {sec['ternary_bitplane_compute']*1e3:.2f} ms "
            f"(vs dequant {sec['ternary_dequant_fp_gemm']*1e3:.2f} ms, "
            f"S={row['speedup_ternary_bitplane_vs_dequant']:.2f}x, "
            f"err={row['ternary_max_abs_error_vs_dequant']:.3g})"
        )
        if row["thread_scaling"]:
            bits = ", ".join(
                f"{t['threads']}→{t['seconds_compute']*1e3:.2f}ms"
                f"({t['speedup_vs_1thread']:.2f}x)"
                for t in row["thread_scaling"]
            )
            print(f"  threads: {bits}")
        print(
            f"  torch FP32 {sec['torch_fp32_linear']*1e3:.2f} ms | "
            f"fake-binary {sec['torch_fake_binary_sign_linear']*1e3:.2f} ms | "
            f"fake/fp {row['fake_binary_vs_torch_fp32']:.2f}"
        )

    payload = {
        "device_note": (
            "CPU; weights pre-packed (inference). Native MSVC/GCC popcount + OpenMP when available. "
            "Thread scaling via binary_gemm_set_num_threads / BNN_NUM_THREADS."
        ),
        "openmp": openmp_enabled(),
        "warmup": args.warmup,
        "reps": args.reps,
        "results": results,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    md = args.out.with_suffix(".md")
    lines = [
        "# Kernel benchmark (CPU, fair protocol)",
        "",
        "Weights pre-packed once. `compute` = GEMM only; `e2e` = pack activations + GEMM.",
        f"Warmup={args.warmup}, reps={args.reps}, OpenMP={openmp_enabled()}.",
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

    lines += ["", "## Thread scaling (compute-only, native)", ""]
    for r in results:
        if not r.get("thread_scaling"):
            continue
        s = r["shape"]
        shape = f"{s['batch']}×{s['in_features']}×{s['out_features']}"
        lines.append(f"### {shape}")
        lines.append("")
        lines.append("| Threads | Compute ms | vs 1-thread |")
        lines.append("|--------:|-----------:|------------:|")
        for t in r["thread_scaling"]:
            lines.append(
                f"| {t['threads']} | {t['seconds_compute']*1e3:.2f} | "
                f"{t['speedup_vs_1thread']:.2f}× |"
            )
        lines.append("")

    lines += [
        "## Pack vs compute",
        "",
        "| Shape | Pack W ms | Pack X ms | Compute ms |",
        "|-------|----------:|----------:|-----------:|",
    ]
    for r in results:
        s = r["shape"]
        shape = f"{s['batch']}×{s['in_features']}×{s['out_features']}"
        sec = r["seconds"]
        lines.append(
            f"| {shape} | {sec['pack_weights_once']*1e3:.2f} | "
            f"{sec['pack_activations']*1e3:.2f} | "
            f"{sec['binary_compute_only_prepacked']*1e3:.2f} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "- **S_compute** is the honest kernel win with deployed packed weights.",
        "- **S_e2e** includes activation packing (still usually >1× at large N).",
        "- **Fake-binary > 1** means `sign`+FP GEMM is slower — simulation ≠ acceleration.",
        "- Theory ~64× is word-op reduction, not wall-clock.",
        "- Thread scaling uses OpenMP over output rows; memory-bound shapes may plateau early.",
        "- Ternary bitplane path is for ±1 activations; full-precision X still uses dequant FP.",
    ]
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.out} and {md}")


if __name__ == "__main__":
    main()
