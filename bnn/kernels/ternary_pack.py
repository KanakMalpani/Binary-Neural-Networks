"""2-bit packing for ternary weights {-1,0,+1} (BitNet pedagogy).

Also provides bitplane packing for the fast ternary GEMM path:
  Wp[i]=1 where W==+1, Wn[i]=1 where W==-1 (same uint64 layout as binary).
"""

from __future__ import annotations

import numpy as np

# Encode: -1 -> 0b10, 0 -> 0b00, +1 -> 0b01  (2 bits)
_ENC = {np.int8(-1): 0b10, np.int8(0): 0b00, np.int8(1): 0b01}
_DEC = {0b10: np.int8(-1), 0b00: np.int8(0), 0b01: np.int8(1), 0b11: np.int8(0)}


def pack_ternary_2bit(q: np.ndarray) -> np.ndarray:
    """Pack int8 ternary matrix (R,C) into uint8 with 4 weights/byte (vectorized).

    Layout matches legacy: codes for flat[i::4] occupy bits [2i, 2i+1].
    """
    flat = np.asarray(q, dtype=np.int8).ravel()
    n = flat.size
    pad = (-n) % 4
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=np.int8)])
    codes = np.zeros(flat.shape, dtype=np.uint8)
    codes[flat == 1] = 0b01
    codes[flat == -1] = 0b10
    out = np.zeros(flat.size // 4, dtype=np.uint8)
    for i in range(4):
        out |= (codes[i::4] << (2 * i)).astype(np.uint8)
    return out


def unpack_ternary_2bit(packed: np.ndarray, rows: int, cols: int) -> np.ndarray:
    n = rows * cols
    p = np.asarray(packed, dtype=np.uint8)
    n_bytes = (n + 3) // 4
    if p.size < n_bytes:
        raise ValueError(f"packed too short: need {n_bytes} bytes, got {p.size}")
    p = p[:n_bytes]
    flat = np.zeros(n_bytes * 4, dtype=np.int8)
    for i in range(4):
        c = (p >> (2 * i)) & 0b11
        slot = flat[i::4]
        slot[c == 0b01] = 1
        slot[c == 0b10] = -1
    return flat[:n].reshape(rows, cols)


def ternary_bytes(rows: int, cols: int) -> int:
    return (rows * cols * 2 + 7) // 8


def pack_ternary_bitplanes(q: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Pack ternary Q (M, N) into (+1) and (-1) uint64 bitplanes.

    Returns (Wp, Wn, n) with shapes (M, ceil(N/64)).
    """
    from .packed import pack_binary_pm1

    q = np.asarray(q)
    if q.ndim != 2:
        raise ValueError(f"expected 2D ternary weights, got shape {q.shape}")
    # pack_binary_pm1 sets bit when value <= 0. For masks we need bit=1 where True.
    # Use ±1 stand-ins: True → -1 (bit1), False → +1 (bit0).
    pos = np.where(q == 1, -1.0, 1.0).astype(np.float32)
    neg = np.where(q == -1, -1.0, 1.0).astype(np.float32)
    wp, n = pack_binary_pm1(pos, axis=1)
    wn, n2 = pack_binary_pm1(neg, axis=1)
    if n != n2:
        raise ValueError("bitplane n mismatch")
    return wp, wn, n


def precompute_bitplane_pops(wp: np.ndarray, wn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Row-wise popcounts for ternary bitplanes (int32 length M)."""
    from .popcount import bitwise_count  # local: avoid import cycle with packed

    pop_p = bitwise_count(wp).sum(axis=1).astype(np.int32)
    pop_n = bitwise_count(wn).sum(axis=1).astype(np.int32)
    return pop_p, pop_n
