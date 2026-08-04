"""Scalar pedagogy GEMM used for WASM parity tests (W2.T06).

Implements the same contract as ``wasm/binary_gemm_wasm.c`` without requiring
Emscripten/clang/Rust. Optional SIMD128 is represented only as a label for
documentation parity with the WASM build — Python always uses the scalar
popcount path (NumPy ``bitwise_count``), which is the correctness oracle.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from bnn.kernels.packed import binary_gemm_numpy_prepacked, pack_binary_pm1
from bnn.kernels.popcount import bitwise_count

KERNEL_SCALAR = 0
KERNEL_SIMD128 = 1

_KernelName = Literal["scalar", "simd128"]
_g_kernel: int = KERNEL_SCALAR


def set_kernel(name: str | None) -> str:
    """Select pedagogy path label. ``simd128`` still uses scalar math in Python."""
    global _g_kernel
    if name is None or name in ("auto", "scalar"):
        _g_kernel = KERNEL_SCALAR
    elif name in ("simd128", "wasm_simd128", "simd"):
        # Label only — Python cannot execute WASM SIMD; math stays scalar.
        _g_kernel = KERNEL_SIMD128
    else:
        raise ValueError(f"unknown wasm pedagogy kernel: {name!r}")
    return kernel_name()


def kernel_name() -> str:
    return "simd128" if _g_kernel == KERNEL_SIMD128 else "scalar"


def binary_gemm_wasm_prepacked(
    xp: np.ndarray, wp: np.ndarray, n: int
) -> np.ndarray:
    """Y from pre-packed uint64 mats — identical math to NumPy packed GEMM."""
    if xp.dtype != np.uint64 or wp.dtype != np.uint64:
        raise TypeError("prepacked matrices must be uint64")
    if xp.ndim != 2 or wp.ndim != 2:
        raise ValueError("expected 2D packed mats")
    if xp.shape[1] != wp.shape[1]:
        raise ValueError("packed word mismatch")
    words = int(xp.shape[1])
    expected = (n + 63) // 64
    if words != expected:
        raise ValueError(f"n={n} implies {expected} words, got {words}")
    B, M = int(xp.shape[0]), int(wp.shape[0])
    out = np.empty((B, M), dtype=np.float32)
    for b in range(B):
        xor = np.bitwise_xor(xp[b : b + 1], wp)
        dist = bitwise_count(xor).sum(axis=1).astype(np.int32)
        out[b] = n - 2 * dist
    return out


def binary_gemm_wasm_numpy(x_pm1: np.ndarray, w_pm1: np.ndarray) -> np.ndarray:
    """Pack ±1 mats and run pedagogy GEMM."""
    xp, n = pack_binary_pm1(np.asarray(x_pm1), axis=1)
    wp, n2 = pack_binary_pm1(np.asarray(w_pm1), axis=1)
    if n != n2:
        raise ValueError(f"packed n mismatch: {n} vs {n2}")
    y = binary_gemm_wasm_prepacked(xp, wp, n)
    y_ref = binary_gemm_numpy_prepacked(xp, wp, n)
    if float(np.max(np.abs(y - y_ref))) != 0.0:
        raise AssertionError("wasm pedagogy path drifted from NumPy packed GEMM")
    return y
