"""2-bit packing for ternary weights {-1,0,+1} (BitNet pedagogy)."""

from __future__ import annotations

import numpy as np

# Encode: -1 -> 0b10, 0 -> 0b00, +1 -> 0b01  (2 bits)
_ENC = {np.int8(-1): 0b10, np.int8(0): 0b00, np.int8(1): 0b01}
_DEC = {0b10: np.int8(-1), 0b00: np.int8(0), 0b01: np.int8(1), 0b11: np.int8(0)}


def pack_ternary_2bit(q: np.ndarray) -> np.ndarray:
    """Pack int8 ternary matrix (R,C) into uint8 with 4 weights/byte."""
    flat = q.astype(np.int8).ravel()
    n = flat.size
    pad = (-n) % 4
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=np.int8)])
    out = np.zeros(flat.size // 4, dtype=np.uint8)
    for i in range(4):
        codes = np.array([_ENC[np.int8(v)] for v in flat[i::4]], dtype=np.uint8)
        out |= (codes << (2 * i)).astype(np.uint8)
    return out


def unpack_ternary_2bit(packed: np.ndarray, rows: int, cols: int) -> np.ndarray:
    n = rows * cols
    flat = np.zeros(((n + 3) // 4) * 4, dtype=np.int8)
    for i in range(4):
        codes = (packed >> (2 * i)) & 0b11
        flat[i::4] = np.array([_DEC[int(c)] for c in codes], dtype=np.int8)
    return flat[:n].reshape(rows, cols)


def ternary_bytes(rows: int, cols: int) -> int:
    return (rows * cols * 2 + 7) // 8
