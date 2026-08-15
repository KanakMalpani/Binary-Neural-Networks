"""Bit-layout contract for the packed encodings.

``pack_bits_u64`` is the single definition of how bits map to uint64 words. The
binary packer and both ternary bitplanes go through it, and the C kernel decodes
the same layout — so a change here silently corrupts every GEMM rather than
failing loudly. These tests pin the layout itself, not just round-trips.
"""

from __future__ import annotations

import numpy as np
import pytest

from bnn.kernels.packed import (
    binary_gemm_numpy_prepacked,
    pack_binary_pm1,
    pack_bits_u64,
)
from bnn.kernels.popcount import bitwise_count
from bnn.kernels.ternary_pack import (
    pack_ternary_2bit,
    pack_ternary_bitplanes,
    unpack_ternary_2bit,
)

# --------------------------------------------------------------------------
# the layout itself
# --------------------------------------------------------------------------

def test_bit_j_of_word_w_is_element_64w_plus_j():
    """The whole contract in one assertion."""
    bits = np.zeros((1, 128), dtype=bool)
    bits[0, 0] = True     # word 0, bit 0
    bits[0, 63] = True    # word 0, bit 63
    bits[0, 64] = True    # word 1, bit 0
    bits[0, 127] = True   # word 1, bit 63
    packed = pack_bits_u64(bits)
    assert packed.shape == (1, 2)
    assert packed[0, 0] == (1 << 0) | (1 << 63)
    assert packed[0, 1] == (1 << 0) | (1 << 63)


@pytest.mark.parametrize("j", [0, 1, 7, 8, 31, 32, 62, 63])
def test_single_bit_positions_round_trip(j: int):
    bits = np.zeros((1, 64), dtype=bool)
    bits[0, j] = True
    assert int(pack_bits_u64(bits)[0, 0]) == 1 << j


def test_packing_is_little_endian_regardless_of_host():
    """`.view('<u8')` must pin byte order — a native-endian view would differ."""
    bits = np.zeros((1, 64), dtype=bool)
    bits[0, 0] = True
    assert int(pack_bits_u64(bits)[0, 0]) == 1


def test_padding_is_false_not_garbage():
    """Sub-word tails must pad with 0 bits, or popcounts drift."""
    bits = np.ones((1, 65), dtype=bool)
    packed = pack_bits_u64(bits)
    assert packed.shape == (1, 2)
    assert int(packed[0, 0]) == (1 << 64) - 1
    assert int(packed[0, 1]) == 1  # only the real 65th bit, rest padded False


def test_output_is_contiguous_uint64():
    """The C kernel reads a raw uint64 pointer — non-contiguity would corrupt it."""
    packed = pack_bits_u64(np.random.default_rng(0).random((7, 130)) > 0.5)
    assert packed.dtype == np.uint64
    assert packed.flags["C_CONTIGUOUS"]


def test_leading_dimensions_are_preserved():
    packed = pack_bits_u64(np.zeros((3, 5, 128), dtype=bool))
    assert packed.shape == (3, 5, 2)


# --------------------------------------------------------------------------
# binary and ternary must share the layout
# --------------------------------------------------------------------------

def test_binary_packer_sets_bit_for_non_positive():
    """bit 1 == value <= 0 is the documented binary convention."""
    x = np.array([[1.0, -1.0, 0.0, 2.0]], dtype=np.float32)
    packed, n = pack_binary_pm1(x, 1)
    assert n == 4
    # -1 at index 1 and 0.0 at index 2 -> bits 1 and 2 set.
    assert int(packed[0, 0]) == (1 << 1) | (1 << 2)


def test_ternary_bitplanes_use_the_same_layout_as_binary():
    """Wp/Wn must be decodable by the same kernel that reads binary packs."""
    q = np.array([[1, -1, 0, 1]], dtype=np.int8)
    wp, wn, n = pack_ternary_bitplanes(q)
    assert n == 4
    assert int(wp[0, 0]) == (1 << 0) | (1 << 3)  # +1 at 0 and 3
    assert int(wn[0, 0]) == (1 << 1)             # -1 at 1
    # A zero weight appears in neither plane.
    assert not (int(wp[0, 0]) >> 2) & 1
    assert not (int(wn[0, 0]) >> 2) & 1


def test_bitplanes_are_disjoint():
    """No weight can be both +1 and -1."""
    rng = np.random.default_rng(3)
    q = rng.choice([-1, 0, 1], size=(32, 256)).astype(np.int8)
    wp, wn, _ = pack_ternary_bitplanes(q)
    assert not np.any(wp & wn), "a weight landed in both bitplanes"


def test_bitplane_popcounts_match_the_source_matrix():
    rng = np.random.default_rng(4)
    q = rng.choice([-1, 0, 1], size=(16, 320)).astype(np.int8)
    wp, wn, _ = pack_ternary_bitplanes(q)
    assert int(bitwise_count(wp).sum()) == int((q == 1).sum())
    assert int(bitwise_count(wn).sum()) == int((q == -1).sum())


# --------------------------------------------------------------------------
# 2-bit codec
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "shape", [(1, 1), (7, 13), (3, 64), (5, 65), (64, 100), (2, 255), (128, 128)]
)
def test_two_bit_round_trip_is_lossless(shape):
    rng = np.random.default_rng(5)
    q = rng.choice([-1, 0, 1], size=shape).astype(np.int8)
    assert np.array_equal(unpack_ternary_2bit(pack_ternary_2bit(q), *shape), q)


def test_two_bit_encoding_values_are_pinned():
    """-1 -> 0b10, 0 -> 0b00, +1 -> 0b01, four weights per byte, lane i at bit 2i."""
    q = np.array([[1, -1, 0, 1]], dtype=np.int8)
    packed = pack_ternary_2bit(q)
    assert packed.size == 1
    assert int(packed[0]) == 0b01 | (0b10 << 2) | (0b00 << 4) | (0b01 << 6)


@pytest.mark.parametrize("fill", [-1, 0, 1])
def test_two_bit_uniform_matrices(fill: int):
    q = np.full((4, 64), fill, dtype=np.int8)
    assert np.array_equal(unpack_ternary_2bit(pack_ternary_2bit(q), 4, 64), q)


def test_unpack_rejects_short_buffer():
    with pytest.raises(ValueError, match="too short"):
        unpack_ternary_2bit(np.zeros(1, dtype=np.uint8), 64, 64)


def test_two_bit_pack_size_is_four_weights_per_byte():
    q = np.zeros((8, 32), dtype=np.int8)  # 256 weights
    assert pack_ternary_2bit(q).size == 256 // 4


# --------------------------------------------------------------------------
# the unsigned-subtraction trap
# --------------------------------------------------------------------------

def test_numpy_gemm_handles_negative_dot_products():
    """Regression: popcount sums promote to *unsigned*, so `n - 2*dist` wraps.

    `bitwise_count(...).sum()` yields uint64. Without the int32 cast in
    binary_gemm_numpy_prepacked, any negative dot product becomes ~1.8e19
    instead. Anti-correlated inputs make every output negative, so this fails
    loudly if that cast is ever "tidied away".
    """
    n = 256
    x = np.ones((4, n), dtype=np.float32)
    w = -np.ones((8, n), dtype=np.float32)  # perfectly anti-correlated
    xp, packed_n = pack_binary_pm1(x, 1)
    wp, _ = pack_binary_pm1(w, 1)
    out = binary_gemm_numpy_prepacked(xp, wp, packed_n)
    assert np.all(out == -n), f"expected all {-n}, got {np.unique(out)[:4]}"
    assert out.dtype == np.float32


def test_popcount_sum_signedness_is_not_guaranteed():
    """Documents *why* the int32 cast in the GEMM is load-bearing.

    The signedness of the popcount sum depends on which implementation runs:

    * NumPy >= 2.0 -> ``np.bitwise_count`` returns uint8, so ``.sum()``
      accumulates into an **unsigned** type and ``n - 2*dist`` wraps.
    * NumPy < 2.0  -> the LUT fallback in ``bnn/kernels/popcount.py`` ends with
      ``.astype(np.intp)``, which is **signed**, and the subtraction behaves.

    constraints.txt pins NumPy < 2 while local dev often has 2.x, so the cast
    must stay regardless of which path is active. Asserting one environment's
    signedness as universal is exactly the mistake this test used to make.
    """
    counts = bitwise_count(np.array([[0xFFFFFFFFFFFFFFFF]], dtype=np.uint64))
    total = counts.sum(axis=1)
    assert int(total[0]) == 64

    if np.issubdtype(total.dtype, np.signedinteger):
        # Signed accumulator: no wrap, but the cast is still required for the
        # other path, so this branch only records that we are on it.
        assert int(0 - 2 * total[0]) == -128
    else:
        # Unsigned accumulator: the trap. The overflow warning is the
        # demonstration, not a defect.
        with np.errstate(over="ignore"):
            wrapped = int(0 - 2 * total[0])
        assert wrapped > 0, "unsigned subtraction should wrap to a huge positive"

    # The invariant that holds on every NumPy: cast to signed first and the
    # arithmetic is correct. This is what binary_gemm_numpy_prepacked does.
    assert int(0 - 2 * total.astype(np.int32)[0]) == -128


@pytest.fixture
def force_popcount_lut(monkeypatch):
    """Run the NumPy < 2.0 LUT fallback even on NumPy >= 2.0.

    constraints.txt pins NumPy < 2 for CI, so the LUT path is what actually
    ships, while local dev usually has 2.x and takes `np.bitwise_count`. Without
    this the two paths are never exercised on the same machine.
    """
    monkeypatch.delattr(np, "bitwise_count", raising=False)
    return True


def test_popcount_paths_agree(force_popcount_lut):
    """The LUT fallback must return the same counts as np.bitwise_count."""
    rng = np.random.default_rng(9)
    a = rng.integers(0, 2**63, size=(64, 8), dtype=np.uint64)
    lut = bitwise_count(a)
    assert lut.sum() == sum(bin(int(v)).count("1") for v in a.ravel())


def test_numpy_gemm_correct_under_lut_popcount(force_popcount_lut):
    """The shipped (NumPy < 2) path must also survive negative dot products."""
    n = 192
    x = np.ones((4, n), dtype=np.float32)
    w = -np.ones((8, n), dtype=np.float32)
    xp, packed_n = pack_binary_pm1(x, 1)
    wp, _ = pack_binary_pm1(w, 1)
    assert np.all(binary_gemm_numpy_prepacked(xp, wp, packed_n) == -n)


def test_gemm_matches_across_both_popcount_paths(monkeypatch):
    """Same inputs, both popcount implementations, identical output."""
    rng = np.random.default_rng(10)
    x = rng.choice([-1.0, 1.0], size=(9, 320)).astype(np.float32)
    w = rng.choice([-1.0, 1.0], size=(40, 320)).astype(np.float32)
    xp, n = pack_binary_pm1(x, 1)
    wp, _ = pack_binary_pm1(w, 1)

    fast = binary_gemm_numpy_prepacked(xp, wp, n)
    monkeypatch.delattr(np, "bitwise_count", raising=False)
    lut = binary_gemm_numpy_prepacked(xp, wp, n)
    assert np.array_equal(fast, lut)
