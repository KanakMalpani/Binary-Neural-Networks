"""Popcount helper — NumPy 1.x LUT vs NumPy 2 bitwise_count."""

from __future__ import annotations

import numpy as np

from bnn.kernels import popcount as pc


def test_bitwise_count_matches_int_bit_count():
    rng = np.random.default_rng(0)
    x = rng.integers(0, 2**64, size=(7, 3), dtype=np.uint64)
    got = pc.bitwise_count(x)
    for i in range(x.size):
        assert int(got.ravel()[i]) == int(x.ravel()[i]).bit_count()


def test_lut_fallback_forced(monkeypatch):
    """Force LUT path even if NumPy 2 is installed."""
    real_hasattr = hasattr

    def fake_hasattr(obj, name):
        if obj is np and name == "bitwise_count":
            return False
        return real_hasattr(obj, name)

    monkeypatch.setattr("builtins.hasattr", fake_hasattr)
    x = np.array([0, 1, 0xFFFFFFFFFFFFFFFF], dtype=np.uint64)
    got = pc.bitwise_count(x)
    assert list(map(int, got)) == [0, 1, 64]
