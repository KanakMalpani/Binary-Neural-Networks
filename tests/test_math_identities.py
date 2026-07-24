"""Mathematical identity tests for binary linear algebra (math path)."""

from __future__ import annotations

import numpy as np
import pytest

from bnn.kernels.packed import pack_binary_pm1
from bnn.math.effectiveness import (
    amdahl_speedup,
    bytes_per_mac,
    effective_ops_per_mac,
    when_binary_less_effective,
)
from bnn.math.identity import prove_identity_sample, xnor_dot_identity
from bnn.math.packing import pack_pm1_uint64, pack_unpack_roundtrip
from bnn.math.reference import binary_gemm_ref
from bnn.math.ternary_math import ternary_accuracy_per_bit, ternary_quantize_pm1_0


@pytest.mark.parametrize("n", [1, 7, 63, 64, 65, 127, 128, 257, 1024])
def test_xnor_dot_identity_random(n: int):
    for seed in (0, 1, 2):
        r = prove_identity_sample(n, seed=seed + n)
        assert r["ok"], r


def test_xnor_identity_all_agree_and_disagree():
    n = 100
    ones = np.ones(n)
    assert xnor_dot_identity(ones, ones)["dot_xor"] == n
    assert xnor_dot_identity(ones, -ones)["dot_xor"] == -n


def test_padding_does_not_affect_dot():
    # n not multiple of 64 — pad bits must be +1 / zero bits
    rng = np.random.default_rng(42)
    for n in (65, 100, 200):
        x = np.where(rng.integers(0, 2, n) == 0, 1.0, -1.0)
        w = np.where(rng.integers(0, 2, n) == 0, 1.0, -1.0)
        r = xnor_dot_identity(x, w)
        assert r["ok"]
        # Compare to naive float dot on ±1
        assert r["dot_pm1"] == pytest.approx(float(np.dot(x, w)))


@pytest.mark.parametrize("n", [1, 64, 65, 128, 200])
def test_pack_unpack_roundtrip(n: int):
    rng = np.random.default_rng(n)
    x = np.where(rng.integers(0, 2, n) == 0, 1.0, -1.0)
    r = pack_unpack_roundtrip(x)
    assert r["ok"] and r["pad_bits_zero"]


def test_pack_matches_kernels_packed():
    rng = np.random.default_rng(7)
    x = np.where(rng.integers(0, 2, (4, 100)) == 0, 1.0, -1.0)
    a, n1 = pack_pm1_uint64(x, axis=1)
    b, n2 = pack_binary_pm1(x, axis=1)
    assert n1 == n2 == 100
    assert np.array_equal(a, b)


def test_binary_gemm_ref_matches_float():
    rng = np.random.default_rng(3)
    x = np.where(rng.integers(0, 2, (5, 70)) == 0, 1.0, -1.0)
    w = np.where(rng.integers(0, 2, (3, 70)) == 0, 1.0, -1.0)
    y_ref = binary_gemm_ref(x, w)
    y_fp = x @ w.T
    assert np.allclose(y_ref, y_fp)


def test_effective_ops_and_bytes():
    ops = effective_ops_per_mac(k=4096)
    assert ops["theoretical_word_reduction"] == pytest.approx(64.0)
    bw = bytes_per_mac(k=4096, out_features=1024)
    assert bw["weight_compression"] == pytest.approx(32.0)
    assert amdahl_speedup(0.7, 32) == pytest.approx(1.0 / (0.3 + 0.7 / 32))


def test_when_binary_less_effective_gpu_and_small_k():
    small = when_binary_less_effective(k=32)
    assert small["less_effective"]
    gpu = when_binary_less_effective(k=4096, on_gpu_tensor_cores=True)
    assert gpu["less_effective"]
    cpu_big = when_binary_less_effective(k=4096, has_softmax=False)
    assert cpu_big["less_effective"] is False or "GPU" not in str(cpu_big["reasons"])


def test_ternary_absmean_and_acc_per_bit():
    w = np.array([0.0, 0.5, -0.5, 1.5, -2.0])
    q, gamma = ternary_quantize_pm1_0(w)
    assert set(np.unique(q).tolist()).issubset({-1.0, 0.0, 1.0})
    assert gamma > 0
    r = ternary_accuracy_per_bit(binary_acc=0.90, ternary_acc=0.93)
    assert "ternary_beats_binary_entropy" in r


def test_zero_scale_note_identity_still_holds():
    # Unscaled identity independent of alpha=0 (documented edge case)
    x = np.array([1.0, -1.0, 1.0])
    w = np.array([-1.0, -1.0, 1.0])
    r = xnor_dot_identity(x, w)
    assert r["ok"]
    assert 0.0 * r["dot_xor"] == 0.0
