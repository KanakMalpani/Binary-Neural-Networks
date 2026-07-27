"""Ternary pack fuzz + dequant GEMM."""

from __future__ import annotations

import numpy as np

from bnn.kernels.ternary_gemm import pack_and_dequant_roundtrip_gemm
from bnn.kernels.ternary_pack import pack_ternary_2bit, ternary_bytes, unpack_ternary_2bit


def test_ternary_fuzz_roundtrip():
    rng = np.random.default_rng(0)
    for rows, cols in [(7, 13), (64, 128), (33, 99), (1024, 64)]:
        q = rng.integers(-1, 2, size=(rows, cols), dtype=np.int8)
        packed = pack_ternary_2bit(q)
        q2 = unpack_ternary_2bit(packed, rows, cols)
        assert int(np.sum(q != q2)) == 0
        assert ternary_bytes(rows, cols) == (rows * cols * 2 + 7) // 8


def test_ternary_dequant_gemm_err0():
    rng = np.random.default_rng(1)
    q = rng.integers(-1, 2, size=(32, 64), dtype=np.int8)
    x = rng.normal(size=(4, 64)).astype(np.float32)
    y_ref, y, err = pack_and_dequant_roundtrip_gemm(x, q, 0.5)
    assert err == 0
    assert float(np.max(np.abs(y_ref - y))) == 0.0
