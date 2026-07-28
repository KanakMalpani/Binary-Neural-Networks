"""Ternary bitplane GEMM: cross-ISA parity (W2.T09).

The binary kernel is exhaustively checked across scalar / AVX2 / AVX-512 / NEON;
the ternary kernel has its own `popcount_and` implementation per ISA and needs
the same treatment, or a SIMD-only bug would ship silently.

Bitplane arithmetic is exact integer work scaled by one float32 multiply, so the
ISA paths must be **bit-identical to each other**; only the comparison against
the dequantised FP reference carries float32 rounding.
"""

from __future__ import annotations

import numpy as np
import pytest

from bnn.kernels.packed import (
    available_kernels,
    kernel_name,
    native_kernel_available,
    pack_binary_pm1,
    set_kernel,
)
from bnn.kernels.ternary_gemm import (
    ternary_bitplane_gemm_native,
    ternary_bitplane_gemm_numpy,
    ternary_dequant_gemm,
)
from bnn.kernels.ternary_pack import pack_ternary_bitplanes, precompute_bitplane_pops

# Word-count boundaries for every vector width: AVX-512 (8), AVX2 (4), NEON (2),
# plus sub-word padding and batch sizes above/below the blocking factor.
SHAPES = [
    (1, 64, 8),       # single row, single word
    (5, 65, 9),       # padded word, odd M
    (3, 128, 33),     # 2 words
    (9, 320, 40),     # 5 words: remainder for all three vector widths
    (8, 512, 64),     # 8 words: exact AVX-512 stride
    (16, 1024, 128),  # larger, crosses the OpenMP threshold
]


@pytest.fixture
def restore_kernel():
    yield
    set_kernel(None)


def _fixture(B: int, N: int, M: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.choice([-1.0, 1.0], size=(B, N)).astype(np.float32)
    q = rng.choice([-1, 0, 1], size=(M, N)).astype(np.int8)
    xp, n = pack_binary_pm1(x, 1)
    wp, wn, _ = pack_ternary_bitplanes(q)
    pop_p, pop_n = precompute_bitplane_pops(wp, wn)
    return x, q, xp, wp, wn, pop_p, pop_n


@pytest.mark.parametrize("B,N,M", SHAPES)
def test_every_isa_path_is_bit_identical(B, N, M, restore_kernel):
    if not native_kernel_available():
        pytest.skip("native library not available")
    if kernel_name() == "unknown":
        pytest.skip("native library predates runtime dispatch")

    _x, _q, xp, wp, wn, pop_p, pop_n = _fixture(B, N, M)
    scale = 0.37

    results: dict[str, np.ndarray] = {}
    for path in available_kernels():
        assert set_kernel(path) == path, f"could not select {path}"
        out = ternary_bitplane_gemm_native(xp, wp, wn, scale, pop_p, pop_n)
        if out is None:
            pytest.skip("ternary native entry point unavailable")
        results[path] = out

    reference_path, reference = next(iter(results.items()))
    for path, got in results.items():
        assert np.array_equal(got, reference), (
            f"{path} differs from {reference_path} — a SIMD path is wrong"
        )


@pytest.mark.parametrize("B,N,M", SHAPES)
def test_every_isa_path_matches_the_fp_reference(B, N, M, restore_kernel):
    """Bitplane result must equal dequantised FP32 up to the scale rounding."""
    if not native_kernel_available() or kernel_name() == "unknown":
        pytest.skip("native dispatch not available")

    x, q, xp, wp, wn, pop_p, pop_n = _fixture(B, N, M, seed=3)
    scale = 0.37
    ref = ternary_dequant_gemm(x, q, scale)
    tol = 1e-4 * max(1.0, abs(scale) * N / 64.0)

    for path in available_kernels():
        set_kernel(path)
        got = ternary_bitplane_gemm_native(xp, wp, wn, scale, pop_p, pop_n)
        if got is None:
            pytest.skip("ternary native entry point unavailable")
        assert np.max(np.abs(ref - got)) < tol, f"{path} disagrees with FP reference"


@pytest.mark.parametrize("B,N,M", SHAPES)
def test_native_matches_numpy_path(B, N, M, restore_kernel):
    """The NumPy fallback is what runs where no compiler exists — keep it equal."""
    if not native_kernel_available() or kernel_name() == "unknown":
        pytest.skip("native dispatch not available")

    _x, _q, xp, wp, wn, pop_p, pop_n = _fixture(B, N, M, seed=7)
    scale = -0.85  # negative scale must not flip a sign anywhere
    expected = ternary_bitplane_gemm_numpy(xp, wp, wn, scale, pop_p, pop_n)
    for path in available_kernels():
        set_kernel(path)
        got = ternary_bitplane_gemm_native(xp, wp, wn, scale, pop_p, pop_n)
        if got is None:
            pytest.skip("ternary native entry point unavailable")
        assert np.array_equal(got, expected), f"{path} differs from the NumPy path"


def test_precomputed_pops_match_on_the_fly(restore_kernel):
    """Passing pop_p/pop_n must equal letting the kernel compute them."""
    if not native_kernel_available() or kernel_name() == "unknown":
        pytest.skip("native dispatch not available")

    _x, _q, xp, wp, wn, pop_p, pop_n = _fixture(8, 512, 64, seed=11)
    scale = 0.5
    for path in available_kernels():
        set_kernel(path)
        with_pre = ternary_bitplane_gemm_native(xp, wp, wn, scale, pop_p, pop_n)
        without = ternary_bitplane_gemm_native(xp, wp, wn, scale, None, None)
        if with_pre is None or without is None:
            pytest.skip("ternary native entry point unavailable")
        assert np.array_equal(with_pre, without), f"{path}: pop precompute changes result"


def test_all_zero_ternary_weights_give_zero(restore_kernel):
    """An all-zero ternary row is |Wp| = |Wn| = 0 — the output must be exactly 0."""
    if not native_kernel_available() or kernel_name() == "unknown":
        pytest.skip("native dispatch not available")

    rng = np.random.default_rng(5)
    x = rng.choice([-1.0, 1.0], size=(4, 128)).astype(np.float32)
    q = np.zeros((16, 128), dtype=np.int8)
    xp, _n = pack_binary_pm1(x, 1)
    wp, wn, _ = pack_ternary_bitplanes(q)
    pop_p, pop_n = precompute_bitplane_pops(wp, wn)

    for path in available_kernels():
        set_kernel(path)
        out = ternary_bitplane_gemm_native(xp, wp, wn, 1.0, pop_p, pop_n)
        if out is None:
            pytest.skip("ternary native entry point unavailable")
        assert np.array_equal(out, np.zeros_like(out)), f"{path}: zero weights leaked"


def test_forced_scalar_env_path_agrees(restore_kernel):
    """Explicitly exercising the fallback every unknown CPU takes."""
    if not native_kernel_available() or kernel_name() == "unknown":
        pytest.skip("native dispatch not available")

    _x, _q, xp, wp, wn, pop_p, pop_n = _fixture(9, 320, 40, seed=13)
    scale = 0.25
    assert set_kernel("scalar") == "scalar"
    scalar_out = ternary_bitplane_gemm_native(xp, wp, wn, scale, pop_p, pop_n)
    if scalar_out is None:
        pytest.skip("ternary native entry point unavailable")
    expected = ternary_bitplane_gemm_numpy(xp, wp, wn, scale, pop_p, pop_n)
    assert np.array_equal(scalar_out, expected)
