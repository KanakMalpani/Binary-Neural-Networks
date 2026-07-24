"""Popcount helpers compatible with NumPy 1.24+ and 2.x.

``numpy.bitwise_count`` exists only on NumPy ≥ 2.0. Constraints allow 1.24–2.x,
so the NumPy reference GEMM path must not hard-require 2.0.
"""

from __future__ import annotations

import numpy as np

# 8-bit LUT for NumPy < 2.0 fallback
_POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def bitwise_count(x: np.ndarray) -> np.ndarray:
    """Element-wise popcount; same contract as ``numpy.bitwise_count`` (NumPy 2+).

    Returns an integer array of the same shape as ``x`` (uint8 counts per element
    for integer inputs up to the bit-width of the dtype).
    """
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x)
    arr = np.asarray(x)
    # View as uint8 bytes and sum popcounts per original element.
    # Works for uint64 / int64 / uint32 / etc. (endian-agnostic byte sum).
    flat = arr.ravel()
    nbytes = flat.dtype.itemsize
    as_u8 = flat.view(np.uint8).reshape(-1, nbytes)
    counts = _POP8[as_u8].sum(axis=1)
    return counts.reshape(arr.shape).astype(np.intp, copy=False)
