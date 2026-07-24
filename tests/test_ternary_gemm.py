"""Ternary bitplane GEMM correctness (+ native when available)."""

from __future__ import annotations

import numpy as np
import pytest

from bnn.kernels.ternary_gemm import (
    ternary_bitplane_gemm_native,
    ternary_bitplane_gemm_numpy,
    ternary_dequant_gemm,
    ternary_fast_matches_dequant,
    ternary_gemm_pm1_x,
)
from bnn.kernels.ternary_pack import pack_ternary_bitplanes, precompute_bitplane_pops
from bnn.kernels.packed import pack_binary_pm1, ternary_native_available


@pytest.mark.parametrize("B,N,M", [(4, 64, 32), (8, 128, 64), (2, 65, 17)])
def test_ternary_bitplane_matches_dequant(B, N, M):
    rng = np.random.default_rng(0)
    x = rng.choice([-1.0, 1.0], size=(B, N)).astype(np.float32)
    q = rng.integers(-1, 2, size=(M, N), dtype=np.int8)
    err = ternary_fast_matches_dequant(x, q, scale=0.75)
    assert err == 0.0


def test_ternary_native_matches_numpy_when_available():
    if not ternary_native_available():
        pytest.skip("ternary native not in DLL")
    rng = np.random.default_rng(2)
    B, N, M = 16, 512, 256
    x = rng.choice([-1.0, 1.0], size=(B, N)).astype(np.float32)
    q = rng.integers(-1, 2, size=(M, N), dtype=np.int8)
    xp, _ = pack_binary_pm1(x, 1)
    wp, wn, _ = pack_ternary_bitplanes(q)
    pop_p, pop_n = precompute_bitplane_pops(wp, wn)
    scale = 0.5
    y_np = ternary_bitplane_gemm_numpy(xp, wp, wn, scale, pop_p, pop_n)
    y_nat = ternary_bitplane_gemm_native(xp, wp, wn, scale, pop_p, pop_n)
    y_ref = ternary_dequant_gemm(x, q, scale)
    assert y_nat is not None
    assert float(np.max(np.abs(y_np - y_nat))) == 0.0
    assert float(np.max(np.abs(y_ref - y_nat))) == 0.0


def test_ternary_gemm_pm1_api():
    rng = np.random.default_rng(3)
    x = rng.choice([-1.0, 1.0], size=(4, 128)).astype(np.float32)
    q = rng.integers(-1, 2, size=(16, 128), dtype=np.int8)
    y = ternary_gemm_pm1_x(x, q, 1.0)
    assert y.shape == (4, 16)
    assert float(np.max(np.abs(y - ternary_dequant_gemm(x, q, 1.0)))) == 0.0
