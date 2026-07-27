"""Kernel input validation, thread control, and the theoretical-ops model.

These are the fail-fast contracts: a packed GEMM given mismatched shapes must
raise, never silently compute garbage that looks like a result.
"""

from __future__ import annotations

import numpy as np
import pytest

from bnn.kernels.packed import (
    _env_num_threads,
    binary_gemm_numpy_prepacked,
    binary_gemm_packed,
    get_num_threads,
    native_kernel_available,
    openmp_enabled,
    pack_binary_pm1,
    set_num_threads,
    theoretical_ops,
)
from bnn.kernels.ternary_gemm import ternary_bitplane_gemm_numpy
from bnn.math.effectiveness import (
    amdahl_speedup,
    bytes_per_mac,
    effective_ops_per_mac,
)


@pytest.fixture
def restore_threads():
    yield
    set_num_threads(None)


# --------------------------------------------------------------------------
# packed GEMM validation
# --------------------------------------------------------------------------

def _packed(B: int, N: int, M: int):
    rng = np.random.default_rng(0)
    x = rng.choice([-1.0, 1.0], size=(B, N)).astype(np.float32)
    w = rng.choice([-1.0, 1.0], size=(M, N)).astype(np.float32)
    xp, n = pack_binary_pm1(x, 1)
    wp, _ = pack_binary_pm1(w, 1)
    return xp, wp, n


def test_prepacked_rejects_non_uint64():
    xp, wp, n = _packed(4, 128, 8)
    with pytest.raises(TypeError, match="uint64"):
        binary_gemm_numpy_prepacked(xp.astype(np.int64), wp, n)


def test_prepacked_rejects_wrong_ndim():
    xp, wp, n = _packed(4, 128, 8)
    with pytest.raises(ValueError, match="2D"):
        binary_gemm_numpy_prepacked(xp.ravel(), wp, n)


def test_prepacked_rejects_word_count_mismatch():
    xp, _, n = _packed(4, 128, 8)
    _, wp_other, _ = _packed(4, 256, 8)
    with pytest.raises(ValueError, match="word mismatch"):
        binary_gemm_numpy_prepacked(xp, wp_other, n)


def test_prepacked_rejects_n_inconsistent_with_words():
    xp, wp, n = _packed(4, 128, 8)
    with pytest.raises(ValueError, match="implies"):
        binary_gemm_numpy_prepacked(xp, wp, n + 64)


def test_binary_gemm_packed_rejects_bad_w_ndim():
    x = np.ones((4, 64), dtype=np.float32)
    with pytest.raises(ValueError, match="w_pm1 must be 2D"):
        binary_gemm_packed(x, np.ones(64, dtype=np.float32))


def test_binary_gemm_packed_rejects_feature_mismatch():
    x = np.ones((4, 64), dtype=np.float32)
    w = np.ones((8, 128), dtype=np.float32)
    with pytest.raises(ValueError, match="in_features mismatch"):
        binary_gemm_packed(x, w)


def test_binary_gemm_packed_rejects_prepacked_n_mismatch():
    x = np.ones((4, 64), dtype=np.float32)
    _, wp, _ = _packed(4, 128, 8)
    with pytest.raises(ValueError, match="n mismatch"):
        binary_gemm_packed(x, None, prepacked_w=(wp, 128))


def test_pack_rejects_non_numeric_dtype():
    with pytest.raises(TypeError, match="numeric"):
        pack_binary_pm1(np.array([["a", "b"]], dtype=object))


def test_pack_pads_to_word_boundary():
    xp, n = pack_binary_pm1(np.ones((3, 100), dtype=np.float32), 1)
    assert n == 100
    assert xp.shape == (3, 2)  # ceil(100/64)


# --------------------------------------------------------------------------
# ternary bitplane validation
# --------------------------------------------------------------------------

def test_ternary_bitplane_rejects_non_uint64():
    a = np.zeros((2, 2), dtype=np.uint64)
    with pytest.raises(TypeError, match="uint64"):
        ternary_bitplane_gemm_numpy(a.astype(np.int64), a, a, 1.0)


def test_ternary_bitplane_rejects_wrong_ndim():
    a = np.zeros((2, 2), dtype=np.uint64)
    with pytest.raises(ValueError, match="2D"):
        ternary_bitplane_gemm_numpy(a.ravel(), a, a, 1.0)


def test_ternary_bitplane_rejects_plane_shape_mismatch():
    a = np.zeros((2, 2), dtype=np.uint64)
    b = np.zeros((2, 3), dtype=np.uint64)
    with pytest.raises(ValueError, match="shape mismatch"):
        ternary_bitplane_gemm_numpy(a, a, b, 1.0)


# --------------------------------------------------------------------------
# thread control
# --------------------------------------------------------------------------

def test_get_num_threads_is_positive():
    assert get_num_threads() >= 1


def test_set_num_threads_round_trips_when_openmp(restore_threads):
    if not (native_kernel_available() and openmp_enabled()):
        pytest.skip("native OpenMP not available")
    set_num_threads(2)
    assert get_num_threads() == 2
    set_num_threads(None)  # library default
    assert get_num_threads() >= 1


def test_set_num_threads_accepts_zero_and_negative_as_default(restore_threads):
    set_num_threads(0)
    assert get_num_threads() >= 1
    set_num_threads(-4)
    assert get_num_threads() >= 1


@pytest.mark.parametrize(
    "value,expected",
    [("4", 4), ("1", 1), ("", None), ("0", None), ("-2", None), ("abc", None)],
)
def test_env_thread_parsing(monkeypatch, value, expected):
    monkeypatch.setenv("BNN_NUM_THREADS", value)
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    assert _env_num_threads() == expected


def test_env_thread_unset_is_none(monkeypatch):
    monkeypatch.delenv("BNN_NUM_THREADS", raising=False)
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    assert _env_num_threads() is None


def test_thread_count_does_not_change_results(restore_threads):
    """Parallel decomposition must be numerically identical to serial."""
    xp, wp, n = _packed(16, 512, 64)
    set_num_threads(1)
    a = binary_gemm_numpy_prepacked(xp, wp, n)
    set_num_threads(4)
    b = binary_gemm_numpy_prepacked(xp, wp, n)
    assert np.array_equal(a, b)


# --------------------------------------------------------------------------
# theoretical model — must stay labelled theoretical
# --------------------------------------------------------------------------

def test_theoretical_ops_reports_word_reduction_and_compression():
    ops = theoretical_ops(64, 4096, 4096)
    assert ops["theoretical_word_reduction"] == pytest.approx(64.0)
    assert ops["weight_compression"] == pytest.approx(32.0)
    assert ops["fp32_macs"] > ops["binary_word_xnor_popcount"]


def test_effective_ops_per_mac_matches_ceil_division():
    r = effective_ops_per_mac(k=100, word_bits=64)
    assert r["binary_word_ops"] == 2.0  # ceil(100/64)
    assert r["theoretical_word_reduction"] == pytest.approx(50.0)
    # The uop proxy must be strictly less optimistic than raw word reduction.
    assert r["uop_proxy_reduction"] < r["theoretical_word_reduction"]


def test_effective_ops_per_mac_zero_k_is_infinite_not_a_crash():
    assert effective_ops_per_mac(k=0)["theoretical_word_reduction"] == float("inf")


@pytest.mark.parametrize("kwargs", [{"k": -1}, {"k": 8, "word_bits": 0}])
def test_effective_ops_per_mac_rejects_bad_input(kwargs):
    with pytest.raises(ValueError):
        effective_ops_per_mac(**kwargs)


def test_bytes_per_mac_rejects_bad_shape():
    with pytest.raises(ValueError):
        bytes_per_mac(k=-1)
    with pytest.raises(ValueError):
        bytes_per_mac(k=8, batch=0)


def test_bytes_per_mac_binary_weights_are_lighter():
    r = bytes_per_mac(k=4096, out_features=4096)
    assert r["weight_bytes_binary"] < r["weight_bytes_fp32"]
    assert r["weight_compression"] == pytest.approx(32.0)
    # Memory-bound is the real story for binary inference: bytes/MAC drops far
    # more than the op count does.
    assert r["bytes_per_mac_binary"] < r["bytes_per_mac_fp32"]


def test_amdahl_caps_speedup_by_unaccelerated_fraction():
    """Even an infinitely fast kernel cannot beat 1/(1-f)."""
    assert amdahl_speedup(0.5, 1e9) == pytest.approx(2.0, rel=1e-3)
    assert amdahl_speedup(0.0, 100.0) == pytest.approx(1.0)
    assert amdahl_speedup(1.0, 4.0) == pytest.approx(4.0)
