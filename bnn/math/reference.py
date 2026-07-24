"""Pure-Python / NumPy reference kernels for identity proofs (not for speed)."""

from __future__ import annotations

import numpy as np

from .identity import xor_popcount_dot
from .packing import pack_pm1_uint64


def binary_dot_ref(x_pm1: np.ndarray, w_pm1: np.ndarray) -> float:
    """Reference ±1 dot via XOR–popcount (math path)."""
    x = np.asarray(x_pm1, dtype=np.float64).ravel()
    w = np.asarray(w_pm1, dtype=np.float64).ravel()
    xp, n = pack_pm1_uint64(x)
    wp, n2 = pack_pm1_uint64(w)
    if n != n2:
        raise ValueError("n mismatch")
    return xor_popcount_dot(xp, wp, n)


def binary_gemm_ref(x_pm1: np.ndarray, w_pm1: np.ndarray) -> np.ndarray:
    """Y[b, m] = <x[b], w[m]> via packed identity (slow, exact for ±1)."""
    x = np.asarray(x_pm1, dtype=np.float64)
    w = np.asarray(w_pm1, dtype=np.float64)
    if x.ndim != 2 or w.ndim != 2:
        raise ValueError("expected 2-D")
    if x.shape[1] != w.shape[1]:
        raise ValueError("in_features mismatch")
    b, m = x.shape[0], w.shape[0]
    out = np.empty((b, m), dtype=np.float64)
    for i in range(b):
        for j in range(m):
            out[i, j] = binary_dot_ref(x[i], w[j])
    return out
