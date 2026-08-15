"""2-bit packing for ternary weights {-1,0,+1} (BitNet pedagogy).

Also provides bitplane packing for the fast ternary GEMM path:
  Wp[i]=1 where W==+1, Wn[i]=1 where W==-1 (same uint64 layout as binary).
"""

from __future__ import annotations

import numpy as np

# Encode: -1 -> 0b10, 0 -> 0b00, +1 -> 0b01  (2 bits)
_ENC = {np.int8(-1): 0b10, np.int8(0): 0b00, np.int8(1): 0b01}
_DEC = {0b10: np.int8(-1), 0b00: np.int8(0), 0b01: np.int8(1), 0b11: np.int8(0)}
# Same mapping as _DEC, indexable by the 2-bit code: 00->0, 01->+1, 10->-1,
# 11->0 (unused encoding, decoded as zero rather than raising).
_LUT_2BIT = np.array([0, 1, -1, 0], dtype=np.int8)


def pack_ternary_2bit(q: np.ndarray) -> np.ndarray:
    """Pack int8 ternary matrix (R,C) into uint8 with 4 weights/byte (vectorized).

    Layout matches legacy: codes for flat[i::4] occupy bits [2i, 2i+1].
    """
    flat = np.asarray(q, dtype=np.int8).ravel()
    n = flat.size
    pad = (-n) % 4
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=np.int8)])
    # Branch-free code assignment; the masked-scatter form allocated two extra
    # boolean temporaries the size of the whole weight matrix.
    codes = (flat > 0).view(np.uint8) | ((flat < 0).view(np.uint8) << 1)
    # Reshape to (-1, 4) so the four lanes are contiguous. The old `codes[i::4]`
    # slicing walked memory with stride 4, which is cache-hostile on large
    # matrices — this is most of the ~4x.
    g = codes.reshape(-1, 4)
    return (g[:, 0] | (g[:, 1] << 2) | (g[:, 2] << 4) | (g[:, 3] << 6)).astype(np.uint8)


def unpack_ternary_2bit(packed: np.ndarray, rows: int, cols: int) -> np.ndarray:
    n = rows * cols
    p = np.asarray(packed, dtype=np.uint8)
    n_bytes = (n + 3) // 4
    if p.size < n_bytes:
        raise ValueError(f"packed too short: need {n_bytes} bytes, got {p.size}")
    p = p[:n_bytes]
    # Extract the four 2-bit lanes into contiguous columns, then decode with a
    # single LUT gather. The old form did eight boolean-mask scatters over
    # stride-4 views; this is one gather over contiguous memory.
    codes = np.empty((n_bytes, 4), dtype=np.uint8)
    for i in range(4):
        np.right_shift(p, 2 * i, out=codes[:, i], casting="unsafe")
    np.bitwise_and(codes, 0b11, out=codes)
    return _LUT_2BIT[codes].reshape(-1)[:n].reshape(rows, cols)


def ternary_bytes(rows: int, cols: int) -> int:
    return (rows * cols * 2 + 7) // 8


def pack_ternary_bitplanes(q: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Pack ternary Q (M, N) into (+1) and (-1) uint64 bitplanes.

    Returns (Wp, Wn, n) with shapes (M, ceil(N/64)).
    """
    from .packed import pack_bits_u64

    q = np.asarray(q)
    if q.ndim != 2:
        raise ValueError(f"expected 2D ternary weights, got shape {q.shape}")
    # Pack the masks straight through the shared bit-layout helper. The previous
    # route built two float32 (M, N) stand-in matrices — 134 MB of temporaries at
    # 4096x4096 — purely so pack_binary_pm1 could re-derive a sign we already
    # knew. Bit-identical output, ~10x faster.
    n = int(q.shape[1])
    return pack_bits_u64(q == 1), pack_bits_u64(q == -1), n


def precompute_bitplane_pops(wp: np.ndarray, wn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Row-wise popcounts for ternary bitplanes (int32 length M)."""
    from .popcount import bitwise_count  # local: avoid import cycle with packed

    pop_p = bitwise_count(wp).sum(axis=1).astype(np.int32)
    pop_n = bitwise_count(wn).sum(axis=1).astype(np.int32)
    return pop_p, pop_n
