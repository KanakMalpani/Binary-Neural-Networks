"""Binary pack / compression tests."""

from __future__ import annotations

import numpy as np

from bnn.kernels.packed import pack_binary_pm1, theoretical_ops


def test_pack_roundtrip_length():
    rng = np.random.default_rng(0)
    for n in (63, 64, 65, 128, 1000):
        w = rng.choice([-1.0, 1.0], size=(8, n)).astype(np.float32)
        packed, n2 = pack_binary_pm1(w, axis=1)
        assert n2 == n
        assert packed.shape[0] == 8
        assert packed.shape[1] == (n + 63) // 64


def test_compression_theoretical_32x():
    t = theoretical_ops(1, 1024, 1024)
    assert abs(t["weight_compression"] - 32.0) < 1e-6
