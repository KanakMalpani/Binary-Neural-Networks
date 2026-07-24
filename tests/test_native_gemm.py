"""Native / NumPy binary GEMM correctness."""

from __future__ import annotations

import numpy as np
import pytest

from bnn.kernels.packed import (
    binary_gemm_native_prepacked,
    binary_gemm_numpy_prepacked,
    binary_gemm_packed,
    fp32_gemm,
    native_kernel_available,
    pack_binary_pm1,
)


@pytest.mark.parametrize(
    "B,N,M",
    [
        (4, 63, 7),
        (8, 64, 16),
        (4, 65, 9),
        (16, 128, 64),
        (8, 512, 256),
    ],
)
def test_numpy_packed_matches_fp(B, N, M):
    rng = np.random.default_rng(0)
    x = rng.choice([-1.0, 1.0], size=(B, N)).astype(np.float32)
    w = rng.choice([-1.0, 1.0], size=(M, N)).astype(np.float32)
    y_fp = fp32_gemm(x, w)
    y = binary_gemm_packed(x, w)
    assert float(np.max(np.abs(y_fp - y))) == 0.0


def test_native_matches_fp_when_available():
    if not native_kernel_available():
        pytest.skip("native DLL not available")
    rng = np.random.default_rng(1)
    B, N, M = 32, 1024, 512
    x = rng.choice([-1.0, 1.0], size=(B, N)).astype(np.float32)
    w = rng.choice([-1.0, 1.0], size=(M, N)).astype(np.float32)
    xp, n = pack_binary_pm1(x, 1)
    wp, _ = pack_binary_pm1(w, 1)
    y_fp = fp32_gemm(x, w)
    y_nat = binary_gemm_native_prepacked(xp, wp, n)
    y_np = binary_gemm_numpy_prepacked(xp, wp, n)
    assert y_nat is not None
    assert float(np.max(np.abs(y_fp - y_nat))) == 0.0
    assert float(np.max(np.abs(y_nat - y_np))) == 0.0
