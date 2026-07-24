"""Ternary GEMM: fast bitplane path + FP dequant reference.

Fast path (±1 activations, ternary weights):
  y = scale * (|Wp| - 2 pop(X&Wp) - |Wn| + 2 pop(X&Wn))
Native C when DLL exposes ternary_gemm_u64; else NumPy bitplane.
Dequant FP GEMM remains for full-precision activations (pedagogy).
"""

from __future__ import annotations

import ctypes

import numpy as np

from .packed import (
    _try_load_native,
    ensure_native_threads,
    native_kernel_available,
    pack_binary_pm1,
    ternary_native_available,
)
from .popcount import bitwise_count
from .ternary_pack import (
    pack_ternary_2bit,
    pack_ternary_bitplanes,
    precompute_bitplane_pops,
    unpack_ternary_2bit,
)


def ternary_dequant_gemm(
    x: np.ndarray,
    q: np.ndarray,
    scale: float,
) -> np.ndarray:
    """Y = X @ (Q.T * scale) with Q in {-1,0,+1}. Correct but not faster than BLAS FP."""
    w = q.astype(np.float32) * float(scale)
    return x.astype(np.float32) @ w.T


def ternary_bitplane_gemm_numpy(
    xp: np.ndarray,
    wp: np.ndarray,
    wn: np.ndarray,
    scale: float,
    pop_p: np.ndarray | None = None,
    pop_n: np.ndarray | None = None,
) -> np.ndarray:
    """NumPy ternary GEMM from pre-packed activation + weight bitplanes."""
    if xp.dtype != np.uint64 or wp.dtype != np.uint64 or wn.dtype != np.uint64:
        raise TypeError("bitplanes must be uint64")
    if xp.ndim != 2 or wp.ndim != 2 or wn.ndim != 2:
        raise ValueError("expected 2D packed mats")
    if xp.shape[1] != wp.shape[1] or wp.shape != wn.shape:
        raise ValueError("bitplane shape mismatch")
    B, words = xp.shape
    M = wp.shape[0]
    if pop_p is None or pop_n is None:
        pop_p, pop_n = precompute_bitplane_pops(wp, wn)
    out = np.empty((B, M), dtype=np.float32)
    for b in range(B):
        and_p = bitwise_count(xp[b : b + 1] & wp).sum(axis=1).astype(np.int32)
        and_n = bitwise_count(xp[b : b + 1] & wn).sum(axis=1).astype(np.int32)
        out[b] = scale * (pop_p - 2 * and_p - pop_n + 2 * and_n).astype(np.float32)
    return out


def ternary_bitplane_gemm_native(
    xp: np.ndarray,
    wp: np.ndarray,
    wn: np.ndarray,
    scale: float,
    pop_p: np.ndarray | None = None,
    pop_n: np.ndarray | None = None,
) -> np.ndarray | None:
    lib = _try_load_native()
    if not lib or not hasattr(lib, "ternary_gemm_u64"):
        return None
    ensure_native_threads()
    if xp.shape[1] != wp.shape[1] or wp.shape != wn.shape:
        raise ValueError("bitplane shape mismatch")
    B, words = int(xp.shape[0]), int(xp.shape[1])
    M = int(wp.shape[0])
    if pop_p is None or pop_n is None:
        pop_p, pop_n = precompute_bitplane_pops(wp, wn)
    pop_p = np.ascontiguousarray(pop_p, dtype=np.int32)
    pop_n = np.ascontiguousarray(pop_n, dtype=np.int32)
    out = np.empty((B, M), dtype=np.float32)
    lib.ternary_gemm_u64(
        xp.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        wp.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        wn.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        pop_p.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        pop_n.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        float(scale),
        B,
        M,
        words,
    )
    return out


def ternary_gemm_pm1_x(
    x_pm1: np.ndarray,
    q: np.ndarray,
    scale: float = 1.0,
    *,
    prepacked: (
        tuple[np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]] | None
    ) = None,
) -> np.ndarray:
    """Y = scale * X @ Q.T for X in ±1 and Q in {-1,0,+1} via bitplanes.

    ``prepacked`` optional deploy tuple: ``(xp, wp, wn, (pop_p, pop_n))``.
    """
    if prepacked is None:
        xp, _n = pack_binary_pm1(np.asarray(x_pm1), axis=1)
        wp, wn, _ = pack_ternary_bitplanes(q)
        pop_p, pop_n = precompute_bitplane_pops(wp, wn)
    else:
        xp, wp, wn, (pop_p, pop_n) = prepacked

    native = ternary_bitplane_gemm_native(xp, wp, wn, scale, pop_p, pop_n)
    if native is not None:
        return native
    return ternary_bitplane_gemm_numpy(xp, wp, wn, scale, pop_p, pop_n)


def pack_and_dequant_roundtrip_gemm(
    x: np.ndarray,
    q: np.ndarray,
    scale: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    packed = pack_ternary_2bit(q)
    q2 = unpack_ternary_2bit(packed, q.shape[0], q.shape[1])
    err = int(np.sum(q != q2))
    y_ref = ternary_dequant_gemm(x, q, scale)
    y = ternary_dequant_gemm(x, q2, scale)
    return y_ref, y, err


def ternary_fast_matches_dequant(
    x_pm1: np.ndarray,
    q: np.ndarray,
    scale: float = 1.0,
) -> float:
    """Max abs error of bitplane path vs FP dequant (for tests)."""
    y_fast = ternary_gemm_pm1_x(x_pm1, q, scale)
    y_ref = ternary_dequant_gemm(x_pm1, q, scale)
    return float(np.max(np.abs(y_fast - y_ref)))


__all__ = [
    "ternary_dequant_gemm",
    "ternary_bitplane_gemm_numpy",
    "ternary_bitplane_gemm_native",
    "ternary_gemm_pm1_x",
    "pack_and_dequant_roundtrip_gemm",
    "ternary_fast_matches_dequant",
    "ternary_native_available",
    "native_kernel_available",
]
