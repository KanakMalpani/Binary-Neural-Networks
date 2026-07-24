"""Ops / byte effectiveness calculators (honest, not marketing).

These numbers feed docs and repro scripts.  They never claim wall-clock 32×
from ``sign()`` alone — that is the thesis lock.
"""

from __future__ import annotations

import math
from typing import Any


def effective_ops_per_mac(
    *,
    k: int,
    word_bits: int = 64,
) -> dict[str, float]:
    """Arithmetic intensity model for one output MAC-equivalent along K.

    For FP32: ``k`` MACs.
    For packed binary: ``ceil(k / word_bits)`` XOR+popcount word ops
    (plus a cheap scale outside the loop).

    Returns ratios as *theoretical word-op reduction*, not latency.
    """
    if k < 0:
        raise ValueError("k must be >= 0")
    if word_bits <= 0:
        raise ValueError("word_bits must be > 0")
    fp_macs = float(k)
    bin_word_ops = float(math.ceil(k / word_bits)) if k else 0.0
    reduction = fp_macs / bin_word_ops if bin_word_ops else float("inf")
    return {
        "k": float(k),
        "word_bits": float(word_bits),
        "fp32_macs": fp_macs,
        "binary_word_ops": bin_word_ops,
        "theoretical_word_reduction": reduction,
        # Rough: each word op ≈ XOR + popcnt ≈ 2 uops vs 1 FMA
        "uop_proxy_fp32": fp_macs,  # 1 FMA ≈ 1 MAC
        "uop_proxy_binary": bin_word_ops * 2.0,
        "uop_proxy_reduction": fp_macs / max(bin_word_ops * 2.0, 1e-12),
    }


def bytes_per_mac(
    *,
    k: int,
    batch: int = 1,
    out_features: int = 1,
    act_dtype_bytes: int = 4,
    weight_bits: int = 1,
) -> dict[str, float]:
    """DRAM bytes touched per output element (batch-1 weight-stream model).

    For inference ``y = W x`` with ``W`` of shape ``(M, K)``, batch 1:
    weight bytes dominate when activations are tiny.
    """
    if k < 0 or batch < 1 or out_features < 1:
        raise ValueError("invalid shape")
    m = out_features
    # Weight footprint for M rows
    w_bytes_fp32 = float(m * k * 4)
    w_bytes_bin = float(m * math.ceil(k * weight_bits / 8))
    # Activations once
    a_bytes = float(batch * k * act_dtype_bytes)
    # Per-output (amortize activations across M)
    per_out_fp = (w_bytes_fp32 + a_bytes) / m
    per_out_bin = (w_bytes_bin + a_bytes) / m
    return {
        "k": float(k),
        "m": float(m),
        "batch": float(batch),
        "weight_bytes_fp32": w_bytes_fp32,
        "weight_bytes_binary": w_bytes_bin,
        "activation_bytes": a_bytes,
        "bytes_per_out_fp32": per_out_fp,
        "bytes_per_out_binary": per_out_bin,
        "weight_compression": w_bytes_fp32 / max(w_bytes_bin, 1e-12),
        "bytes_per_mac_fp32": 4.0,  # one weight byte-load proxy per MAC
        "bytes_per_mac_binary": weight_bits / 8.0,
    }


def amdahl_speedup(f: float, s_kernel: float) -> float:
    """End-to-end speedup when fraction ``f`` of runtime is sped up by ``s_kernel``."""
    if not 0.0 <= f <= 1.0:
        raise ValueError("f must be in [0, 1]")
    if s_kernel <= 0:
        raise ValueError("s_kernel must be > 0")
    return 1.0 / ((1.0 - f) + f / s_kernel)


def when_binary_less_effective(
    *,
    k: int,
    has_softmax: bool = False,
    has_layernorm: bool = False,
    on_gpu_tensor_cores: bool = False,
    m: int = 64,
) -> dict[str, Any]:
    """Quantitative *when not* thresholds (heuristics, documented in docs/35).

    Binary math loses when:
    - K is tiny (packing overhead dominates)
    - Softmax / LayerNorm / attention dominate (Amdahl)
    - GPU Tensor Cores already give dense FP16/INT8 throughput
    """
    ops = effective_ops_per_mac(k=k)
    bw = bytes_per_mac(k=k, out_features=m)
    # Overhead proxy: at least ~1 scale + pack cost; require K >= 256 to win on CPU
    k_threshold_cpu = 256
    pack_overhead_dominates = k < k_threshold_cpu
    # Softmax is O(seq^2) FP — binary FFN cannot erase it
    non_matmul_tax = 0.0
    if has_softmax:
        non_matmul_tax += 0.35
    if has_layernorm:
        non_matmul_tax += 0.10
    f_matmul = max(0.05, 1.0 - non_matmul_tax)
    # On GPU TC, FP16 matmul is so fast that bit kernels rarely win
    gpu_disadvantage = on_gpu_tensor_cores
    s_kernel_assumed = 4.0 if not gpu_disadvantage else 1.2
    e2e = amdahl_speedup(f_matmul, s_kernel_assumed)
    less_effective = bool(
        pack_overhead_dominates
        or e2e < 1.15
        or gpu_disadvantage
        or (has_softmax and k < 1024)
    )
    return {
        "k": k,
        "less_effective": less_effective,
        "reasons": [
            r
            for r, cond in (
                (f"K={k} < CPU packing threshold {k_threshold_cpu}", pack_overhead_dominates),
                ("softmax/attention non-matmul tax", has_softmax),
                ("layernorm FP tax", has_layernorm),
                ("GPU Tensor Core regime (prefer INT8/FP16)", gpu_disadvantage),
                (f"Amdahl e2e<{1.15:.2f} with assumed S={s_kernel_assumed}", e2e < 1.15),
            )
            if cond
        ],
        "assumed_matmul_fraction": f_matmul,
        "assumed_kernel_speedup": s_kernel_assumed,
        "amdahl_e2e": e2e,
        "theoretical_word_reduction": ops["theoretical_word_reduction"],
        "weight_compression": bw["weight_compression"],
    }


def effectiveness_report(
    *,
    k: int = 4096,
    m: int = 4096,
    f_matmul: float = 0.70,
    s_kernel: float = 4.0,
) -> dict[str, Any]:
    """Bundle arith + memory + Amdahl for docs/repro."""
    ops = effective_ops_per_mac(k=k)
    bw = bytes_per_mac(k=k, out_features=m)
    return {
        "ops": ops,
        "bytes": bw,
        "amdahl": {
            "f_matmul": f_matmul,
            "s_kernel": s_kernel,
            "s_e2e": amdahl_speedup(f_matmul, s_kernel),
        },
        "thesis_lock": (
            "32× is exact weight compression (uint64 pack) and a bandwidth upper "
            "bound — not a guaranteed GPU or wall-clock claim from sign()."
        ),
    }
