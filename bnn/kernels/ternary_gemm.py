"""2-bit ternary store + FP dequant GEMM reference (pedagogy — not a speed path)."""

from __future__ import annotations

import numpy as np

from .ternary_pack import pack_ternary_2bit, unpack_ternary_2bit


def ternary_dequant_gemm(
    x: np.ndarray,
    q: np.ndarray,
    scale: float,
) -> np.ndarray:
    """Y = X @ (Q.T * scale) with Q in {-1,0,+1}. Correct but not faster than BLAS FP."""
    w = q.astype(np.float32) * float(scale)
    return x.astype(np.float32) @ w.T


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
