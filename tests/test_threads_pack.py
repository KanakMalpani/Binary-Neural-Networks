"""Native thread control + packbits encoding smoke."""

from __future__ import annotations

import numpy as np
import pytest

from bnn.kernels.packed import (
    binary_gemm_native_prepacked,
    get_num_threads,
    native_kernel_available,
    openmp_enabled,
    pack_binary_pm1,
    set_num_threads,
)


def test_packbits_matches_legacy_multiply_sum():
    rng = np.random.default_rng(0)
    x = rng.choice([-1.0, 1.0], size=(5, 130)).astype(np.float32)
    packed, n = pack_binary_pm1(x, axis=1)
    # Legacy reference
    bits = (x <= 0).astype(np.uint8)
    pad = (-x.shape[1]) % 64
    if pad:
        bits = np.pad(bits, [(0, 0), (0, pad)], constant_values=0)
    bits = bits.reshape(x.shape[0], -1, 64)
    weights = np.uint64(1) << np.arange(64, dtype=np.uint64)
    legacy = (bits.astype(np.uint64) * weights).sum(axis=-1)
    assert n == x.shape[1]
    assert np.array_equal(packed, legacy)


def test_set_num_threads_when_native():
    if not native_kernel_available():
        pytest.skip("native DLL not available")
    set_num_threads(1)
    assert get_num_threads() >= 1
    y_shape_ok = False
    rng = np.random.default_rng(0)
    x = rng.choice([-1.0, 1.0], size=(8, 256)).astype(np.float32)
    w = rng.choice([-1.0, 1.0], size=(64, 256)).astype(np.float32)
    xp, n = pack_binary_pm1(x, 1)
    wp, _ = pack_binary_pm1(w, 1)
    set_num_threads(2 if openmp_enabled() else 1)
    y = binary_gemm_native_prepacked(xp, wp, n)
    assert y is not None and y.shape == (8, 64)
    y_shape_ok = True
    set_num_threads(0)  # restore default
    assert y_shape_ok
