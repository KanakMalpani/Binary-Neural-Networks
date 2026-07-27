"""Native / NumPy binary GEMM correctness."""

from __future__ import annotations

import numpy as np
import pytest

from bnn.kernels.packed import (
    available_kernels,
    binary_gemm_native_prepacked,
    binary_gemm_native_scaled,
    binary_gemm_numpy_prepacked,
    binary_gemm_packed,
    cpu_features,
    fp32_gemm,
    kernel_name,
    native_kernel_available,
    pack_binary_pm1,
    set_kernel,
)


@pytest.fixture
def restore_kernel():
    """Kernel selection is process-global C state — always hand it back."""
    yield
    set_kernel(None)


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


# Shapes chosen to exercise every blocking boundary: batch below / at / above
# the 4-row block, and word counts that hit the AVX-512 (8), AVX2 (4) and NEON
# (2) vector remainders as well as sub-word padding.
_DISPATCH_SHAPES = [
    (1, 4096, 64),    # B < block, full 64-word rows
    (3, 128, 33),     # B < block, 2-word rows, odd M
    (4, 64, 16),      # B == block exactly, single word
    (5, 65, 9),       # block + 1 remainder row, padded word
    (7, 1000, 129),   # odd everything, 16-word rows (AVX-512 remainder)
    (8, 320, 40),     # 5-word rows: remainder for all three vector widths
    (9, 512, 130),    # block*2 + 1, 8-word rows
    (16, 2048, 256),  # larger, exercises OpenMP threshold
]


@pytest.mark.parametrize("B,N,M", _DISPATCH_SHAPES)
def test_all_kernel_paths_identical(B, N, M, restore_kernel):
    """Every ISA path this CPU supports must agree exactly with FP32.

    This is what makes the library safe to ship worldwide: an AVX-512 machine
    and a scalar-only machine must produce the same numbers, not merely close
    ones. Binary GEMM is exact integer arithmetic, so the bar is err == 0.
    """
    if not native_kernel_available():
        pytest.skip("native library not available")
    paths = available_kernels()
    if kernel_name() == "unknown":
        pytest.skip("native library predates runtime dispatch")

    rng = np.random.default_rng(11)
    x = rng.choice([-1.0, 1.0], size=(B, N)).astype(np.float32)
    w = rng.choice([-1.0, 1.0], size=(M, N)).astype(np.float32)
    xp, n = pack_binary_pm1(x, 1)
    wp, _ = pack_binary_pm1(w, 1)
    y_fp = fp32_gemm(x, w)
    y_np = binary_gemm_numpy_prepacked(xp, wp, n)

    for path in paths:
        assert set_kernel(path) == path, f"could not select {path}"
        y = binary_gemm_native_prepacked(xp, wp, n)
        assert y is not None
        assert float(np.max(np.abs(y_fp - y))) == 0.0, f"{path} disagrees with FP32"
        assert float(np.max(np.abs(y_np - y))) == 0.0, f"{path} disagrees with NumPy"


@pytest.mark.parametrize("B,N,M", _DISPATCH_SHAPES)
def test_fused_epilogue_matches_unfused(B, N, M, restore_kernel):
    """alpha/bias folded into the kernel must equal the NumPy two-pass form."""
    if not native_kernel_available():
        pytest.skip("native library not available")
    rng = np.random.default_rng(23)
    x = rng.choice([-1.0, 1.0], size=(B, N)).astype(np.float32)
    w = rng.choice([-1.0, 1.0], size=(M, N)).astype(np.float32)
    xp, n = pack_binary_pm1(x, 1)
    wp, _ = pack_binary_pm1(w, 1)
    alpha = rng.standard_normal(M).astype(np.float32)
    bias = rng.standard_normal(M).astype(np.float32)

    fused = binary_gemm_native_scaled(xp, wp, n, alpha, bias)
    if fused is None:
        pytest.skip("native library predates the fused epilogue")
    expected = binary_gemm_numpy_prepacked(xp, wp, n) * alpha + bias
    # Same arithmetic, different association order -> float32 rounding only.
    np.testing.assert_allclose(fused, expected, rtol=1e-5, atol=1e-4)

    for path in available_kernels():
        assert set_kernel(path) == path
        y = binary_gemm_native_scaled(xp, wp, n, alpha, bias)
        np.testing.assert_allclose(y, expected, rtol=1e-5, atol=1e-4)


def test_fused_epilogue_none_args_are_identity():
    """No alpha/bias must reproduce the plain GEMM exactly (no rounding)."""
    if not native_kernel_available():
        pytest.skip("native library not available")
    rng = np.random.default_rng(24)
    x = rng.choice([-1.0, 1.0], size=(9, 512)).astype(np.float32)
    w = rng.choice([-1.0, 1.0], size=(130, 512)).astype(np.float32)
    xp, n = pack_binary_pm1(x, 1)
    wp, _ = pack_binary_pm1(w, 1)
    fused = binary_gemm_native_scaled(xp, wp, n, None, None)
    if fused is None:
        pytest.skip("native library predates the fused epilogue")
    assert np.array_equal(fused, binary_gemm_numpy_prepacked(xp, wp, n))


def test_fused_epilogue_rejects_wrong_length_vectors():
    if not native_kernel_available():
        pytest.skip("native library not available")
    rng = np.random.default_rng(25)
    x = rng.choice([-1.0, 1.0], size=(4, 128)).astype(np.float32)
    w = rng.choice([-1.0, 1.0], size=(16, 128)).astype(np.float32)
    xp, n = pack_binary_pm1(x, 1)
    wp, _ = pack_binary_pm1(w, 1)
    if binary_gemm_native_scaled(xp, wp, n, None, None) is None:
        pytest.skip("native library predates the fused epilogue")
    with pytest.raises(ValueError):
        binary_gemm_native_scaled(xp, wp, n, np.ones(15, dtype=np.float32), None)
    with pytest.raises(ValueError):
        binary_gemm_native_scaled(xp, wp, n, None, np.ones(17, dtype=np.float32))


def test_kernel_dispatch_reports_a_real_path():
    if not native_kernel_available():
        assert kernel_name() == "numpy"
        return
    name = kernel_name()
    assert name in {"scalar", "avx2", "avx512", "neon", "unknown"}
    if name == "unknown":
        pytest.skip("native library predates runtime dispatch")
    # Scalar is always legal; anything faster must be backed by a real CPU flag.
    feats = cpu_features()
    assert "scalar" in available_kernels()
    if name == "avx512":
        assert feats["avx512_vpopcntdq"]
    if name == "avx2":
        assert feats["avx2"]
    if name == "neon":
        assert feats["neon"]


def test_set_kernel_rejects_unknown_name():
    if not native_kernel_available() or kernel_name() == "unknown":
        pytest.skip("native dispatch not available")
    with pytest.raises(ValueError):
        set_kernel("avx1024")


def test_unsupported_kernel_falls_back_to_scalar(restore_kernel):
    """Asking for a path this CPU lacks must degrade, never crash."""
    if not native_kernel_available() or kernel_name() == "unknown":
        pytest.skip("native dispatch not available")
    feats = cpu_features()
    missing = [
        name
        for name, ok in (
            ("avx2", feats["avx2"]),
            ("avx512", feats["avx512_vpopcntdq"]),
            ("neon", feats["neon"]),
        )
        if not ok
    ]
    if not missing:
        pytest.skip("this CPU supports every path")
    for name in missing:
        assert set_kernel(name) == "scalar"
