"""docs/45 P1: no-native GEMM must not lose to FP32 BLAS at B=64.

``binary_gemm_numpy_prepacked`` stays the ISA-parity reference (Python popcount
loop). ``binary_gemm_packed`` / wrap dispatch dequant+BLAS at/above the batch
crossover when native is absent.
"""

from __future__ import annotations

import inspect
import time

import numpy as np
import pytest
import torch

from bnn.kernels.packed import (
    NUMPY_PACKED_BLAS_CROSSOVER_BATCH,
    binary_gemm_numpy_or_blas,
    binary_gemm_numpy_prepacked,
    binary_gemm_packed,
    fp32_gemm,
    kernel_name,
    native_kernel_available,
    numpy_packed_blas_crossover_batch,
    pack_binary_pm1,
    prefer_numpy_blas_fallback,
    unpack_binary_pm1,
)
from bnn.wrap.packed_linear import PackedBinaryXNORLinear


def _pm1(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    return rng.choice([-1.0, 1.0], size=shape).astype(np.float32)


def _best_s(fn, *, warmup: int = 2, reps: int = 3, rounds: int = 3) -> float:
    """Min-of-rounds mean; same estimator as docs/45 §6."""
    for _ in range(warmup):
        fn()
    best = float("inf")
    for _ in range(rounds):
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        best = min(best, (time.perf_counter() - t0) / reps)
    return best


@pytest.fixture
def force_numpy(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BNN_FORCE_NUMPY", "1")


def test_numpy_prepacked_reference_stays_the_python_loop():
    """Guard rail: do not 'fix' P1 by rewriting the ISA-parity reference."""
    src = inspect.getsource(binary_gemm_numpy_prepacked)
    assert "for b in range(B)" in src
    assert "fp32_gemm" not in src
    assert "prefer_numpy_blas_fallback" not in src


def test_crossover_default_biases_toward_blas():
    assert NUMPY_PACKED_BLAS_CROSSOVER_BATCH == 8
    assert prefer_numpy_blas_fallback(7) is False
    assert prefer_numpy_blas_fallback(8) is True
    assert prefer_numpy_blas_fallback(64) is True


@pytest.mark.parametrize(
    "value,expected",
    [("0", 0), ("16", 16), ("", 8), ("abc", 8), ("-2", 8)],
)
def test_crossover_env_override(monkeypatch: pytest.MonkeyPatch, value: str, expected: int):
    if value == "":
        monkeypatch.delenv("BNN_NUMPY_BLAS_BATCH", raising=False)
    else:
        monkeypatch.setenv("BNN_NUMPY_BLAS_BATCH", value)
    assert numpy_packed_blas_crossover_batch() == expected
    if expected == 0:
        assert prefer_numpy_blas_fallback(1) is True
    elif expected == 16:
        assert prefer_numpy_blas_fallback(8) is False
        assert prefer_numpy_blas_fallback(16) is True


def test_unpack_roundtrip_matches_pack():
    rng = np.random.default_rng(0)
    w = _pm1(rng, (9, 100))
    wp, n = pack_binary_pm1(w, 1)
    recovered = unpack_binary_pm1(wp, n)
    assert recovered.dtype == np.float32
    assert np.array_equal(recovered, w)


def test_unpack_rejects_bad_layout():
    rng = np.random.default_rng(1)
    wp, n = pack_binary_pm1(_pm1(rng, (4, 128)), 1)
    with pytest.raises(ValueError, match="2D"):
        unpack_binary_pm1(wp.ravel(), n)
    with pytest.raises(ValueError, match="implies"):
        unpack_binary_pm1(wp, n + 64)


def test_force_numpy_hides_native(force_numpy: None):
    assert native_kernel_available() is False
    assert kernel_name() == "numpy"


@pytest.mark.parametrize("B,N,M", [(1, 64, 16), (7, 128, 32), (8, 128, 32), (64, 128, 64)])
def test_fallback_paths_err_zero_vs_fp32_and_numpy_ref(
    force_numpy: None, monkeypatch: pytest.MonkeyPatch, B: int, N: int, M: int
):
    rng = np.random.default_rng(2)
    x = _pm1(rng, (B, N))
    w = _pm1(rng, (M, N))
    xp, n = pack_binary_pm1(x, 1)
    wp, _ = pack_binary_pm1(w, 1)
    y_fp = fp32_gemm(x, w)
    y_np = binary_gemm_numpy_prepacked(xp, wp, n)

    for threshold in ("0", "8", "1000000"):
        monkeypatch.setenv("BNN_NUMPY_BLAS_BATCH", threshold)
        y = binary_gemm_packed(x, w)
        y_pre = binary_gemm_packed(x, None, prepacked_w=(wp, n))
        y_disp = binary_gemm_numpy_or_blas(xp, wp, n, x_pm1=x, w_pm1=w)
        assert float(np.max(np.abs(y_fp - y))) == 0.0, threshold
        assert float(np.max(np.abs(y_np - y))) == 0.0, threshold
        assert float(np.max(np.abs(y_fp - y_pre))) == 0.0, threshold
        assert float(np.max(np.abs(y_fp - y_disp))) == 0.0, threshold


def test_b64_fallback_not_worse_than_fp32_small_margin(force_numpy: None):
    """Dispatched no-native path at B=64 must stay within 2× of FP32 BLAS.

    The bug this locks is the Python popcount loop at 5–11× slower than BLAS.
    """
    assert prefer_numpy_blas_fallback(64) is True
    rng = np.random.default_rng(3)
    B, N, M = 64, 1024, 1024
    x = _pm1(rng, (B, N))
    w = _pm1(rng, (M, N))
    y = binary_gemm_packed(x, w)
    y_fp = fp32_gemm(x, w)
    assert float(np.max(np.abs(y_fp - y))) == 0.0

    t_fp = _best_s(lambda: fp32_gemm(x, w), warmup=4, reps=5, rounds=4)
    t_dispatch = _best_s(lambda: binary_gemm_packed(x, w), warmup=4, reps=5, rounds=4)
    ratio = t_dispatch / t_fp
    assert ratio <= 2.0, (
        f"no-native dispatch {t_dispatch * 1e3:.2f} ms vs FP32 {t_fp * 1e3:.2f} ms "
        f"({ratio:.2f}×); packed NumPy would be 5–11× (docs/45 P1)"
    )


def test_wrap_xnor_fallback_err_zero_and_weights_stay_packed(force_numpy: None):
    torch.manual_seed(0)
    lin = torch.nn.Linear(128, 64)
    mod = PackedBinaryXNORLinear(lin.weight.data, lin.bias.data)
    assert mod.uses_native is False
    packed_bytes = mod.packed_weight_bytes()
    x = torch.randn(64, 128)
    y = mod(x)

    xp, n = pack_binary_pm1(x.numpy(), 1)
    y_np = binary_gemm_numpy_prepacked(xp, mod._wp_np, n)
    y_np = y_np * mod._alpha_np
    if mod._bias_np is not None:
        y_np = y_np + mod._bias_np
    assert float(np.max(np.abs(y.detach().numpy() - y_np))) == 0.0
    assert mod.packed_weight_bytes() == packed_bytes
    assert packed_bytes == 64 * ((128 + 63) // 64) * 8
