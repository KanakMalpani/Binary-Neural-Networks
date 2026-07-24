"""Packing / unpacking bijections for ±1 ↔ uint64 (lab encoding).

bit 0 → +1, bit 1 → −1.  Little bit-order within each uint64 word.
Trailing pad bits are 0 (+1) and must round-trip as +1 when unpacking
only the logical length ``n``.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def pack_pm1_uint64(x: np.ndarray, axis: int = -1) -> tuple[np.ndarray, int]:
    """Pack ±1 (or any numeric) along ``axis`` into uint64 words.

    Same semantics as ``bnn.kernels.packed.pack_binary_pm1`` but kept here as
    a pure math-path dependency (no native DLL).
    """
    x = np.ascontiguousarray(np.asarray(x))
    if x.size == 0:
        raise ValueError("pack_pm1_uint64: empty array")
    x = np.moveaxis(x, axis, -1)
    shape = x.shape
    n = int(shape[-1])
    bits = np.less_equal(x, 0)
    pad = (-n) % 64
    if pad:
        bits = np.pad(bits, [(0, 0)] * (bits.ndim - 1) + [(0, pad)], constant_values=False)
    packed_shape = bits.shape[:-1] + (bits.shape[-1] // 64, 64)
    bits64 = bits.reshape(packed_shape)
    u8 = np.packbits(bits64.astype(np.uint8, copy=False), axis=-1, bitorder="little")
    u8 = np.ascontiguousarray(u8)
    packed = u8.view("<u8").reshape(*shape[:-1], -1).astype(np.uint64, copy=False)
    return np.ascontiguousarray(packed, dtype=np.uint64), n


def unpack_pm1_uint64(packed: np.ndarray, n: int) -> np.ndarray:
    """Inverse of ``pack_pm1_uint64`` for 1-D or row-packed 2-D arrays.

    Returns float64 array of shape ``(*packed.shape[:-1], n)`` with ±1.
    """
    packed = np.asarray(packed, dtype=np.uint64)
    if packed.ndim == 0:
        raise ValueError("packed must be at least 1-D")
    words = packed.shape[-1]
    expected = (n + 63) // 64 if n > 0 else 0
    if words != expected:
        raise ValueError(f"n={n} expects {expected} words, got {words}")
    if n == 0:
        return np.zeros(packed.shape[:-1] + (0,), dtype=np.float64)

    # Expand each word to 64 bits (little-endian bit order)
    flat = packed.reshape(-1, words)
    out_rows = []
    for row in flat:
        bits = np.empty(words * 64, dtype=np.uint8)
        for wi, word in enumerate(row):
            w = int(word)
            for b in range(64):
                bits[wi * 64 + b] = (w >> b) & 1
        bits = bits[:n]
        pm1 = np.where(bits == 0, 1.0, -1.0)
        out_rows.append(pm1)
    arr = np.stack(out_rows, axis=0)
    return arr.reshape(packed.shape[:-1] + (n,))


def pack_unpack_roundtrip(
    x_pm1: np.ndarray,
    *,
    atol: float = 0.0,
) -> dict[str, Any]:
    """Prove pack→unpack recovers the lab ±1 projection of ``x_pm1``."""
    x = np.asarray(x_pm1, dtype=np.float64)
    if x.ndim == 1:
        x_work = x.reshape(1, -1)
        squeeze = True
    elif x.ndim == 2:
        x_work = x
        squeeze = False
    else:
        raise ValueError("only 1-D or 2-D supported")

    canonical = np.where(x_work > 0, 1.0, -1.0)
    packed, n = pack_pm1_uint64(canonical, axis=-1)
    recovered = unpack_pm1_uint64(packed, n)
    ok = bool(np.allclose(canonical, recovered, atol=atol))
    # Pad bits in last word must be zero (+1 encoding)
    rem = n % 64
    pad_ok = True
    if rem:
        last = packed[..., -1]
        high_mask = ~np.uint64((1 << rem) - 1) if rem else np.uint64(0)
        # When rem==0 there is no partial word; when rem>0 high bits must be 0
        pad_ok = bool(np.all((last & high_mask) == 0))
    return {
        "n": n,
        "words": int(packed.shape[-1]),
        "ok": ok,
        "pad_bits_zero": pad_ok,
        "max_abs_err": float(np.max(np.abs(canonical - recovered))) if n else 0.0,
        "shape": tuple(recovered.shape[1:] if squeeze else recovered.shape),
    }
