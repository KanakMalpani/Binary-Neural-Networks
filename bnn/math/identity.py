r"""XNOR / XOR–popcount ↔ ±1 dot-product identities.

Encoding used throughout this lab (matches ``bnn.kernels.packed``):

- bit ``0``  ↦  ``+1``
- bit ``1``  ↦  ``-1``

Derivation
----------
Let \(x_i, w_i \in \{+1,-1\}\) for \(i=1\ldots n\).  Map to bits
\(b_i = \mathbf{1}[x_i = -1]\) (so XOR bit = 1 iff signs disagree).

Agreeing coordinates contribute \(+1\) to the product; disagreeing contribute
\(-1\).  If \(d = \mathrm{popcount}(x_{\mathrm{bit}} \oplus w_{\mathrm{bit}})\)
is the number of disagreements (Hamming distance), then

\[
\langle x, w \rangle
  = (n - d)\cdot(+1) + d\cdot(-1)
  = n - 2d.
\]

Equivalently with XNOR (agreements):
\( \langle x,w\rangle = 2\cdot\mathrm{popcount}(\mathrm{XNOR}) - n \).

Padding
-------
When packing into 64-bit words, trailing pad bits must encode ``+1`` (bit 0)
so they do **not** inflate Hamming distance.  The identity then uses the
*logical* length ``n``, not ``64 * words``.

Zero / missing scales
---------------------
If a channel scale \(\alpha = 0\), the scaled binary matmul is identically
zero regardless of the ±1 dot; the identity still holds for the unscaled
inner product.  Empty ``n = 0`` yields ``0``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from bnn.kernels.popcount import bitwise_count

from .packing import pack_pm1_uint64


def pm1_dot(x: np.ndarray, w: np.ndarray) -> float:
    """Exact ±1 (or real) inner product as float64."""
    x = np.asarray(x, dtype=np.float64).ravel()
    w = np.asarray(w, dtype=np.float64).ravel()
    if x.size != w.size:
        raise ValueError(f"length mismatch: {x.size} vs {w.size}")
    return float(np.dot(x, w))


def hamming_to_dot(n: int, hamming: int) -> float:
    """Map Hamming distance on ±1 bit-codes to the real inner product."""
    if n < 0:
        raise ValueError("n must be >= 0")
    if hamming < 0 or hamming > n:
        raise ValueError(f"hamming {hamming} outside [0, {n}]")
    return float(n - 2 * hamming)


def xor_popcount_dot(x_bits: np.ndarray, w_bits: np.ndarray, n: int) -> float:
    """``n - 2 * popcount(x XOR w)`` over the first ``n`` logical bits.

    ``x_bits`` / ``w_bits`` are uint64 words (little bit-order within each word).
    Pad bits beyond ``n`` must be 0 (encode +1); they are masked out.
    """
    x_bits = np.asarray(x_bits, dtype=np.uint64).ravel()
    w_bits = np.asarray(w_bits, dtype=np.uint64).ravel()
    if x_bits.shape != w_bits.shape:
        raise ValueError("packed shapes must match")
    words = int(x_bits.size)
    expected = (n + 63) // 64 if n > 0 else 0
    if words != expected:
        raise ValueError(f"n={n} expects {expected} words, got {words}")
    if n == 0:
        return 0.0

    xor = np.bitwise_xor(x_bits, w_bits)
    # Mask unused high bits in the last word so padding never counts.
    full = n // 64
    rem = n % 64
    dist = 0
    if full:
        dist += int(bitwise_count(xor[:full]).sum())
    if rem:
        mask = np.uint64((1 << rem) - 1)
        dist += int(bitwise_count(xor[full] & mask))
    return hamming_to_dot(n, dist)


def xnor_dot_identity(
    x_pm1: np.ndarray,
    w_pm1: np.ndarray,
    *,
    rtol: float = 0.0,
    atol: float = 0.0,
) -> dict[str, Any]:
    """Prove ``<x,w> = n - 2 popcount(x_bit XOR w_bit)`` for ±1 vectors.

    Parameters
    ----------
    x_pm1, w_pm1 :
        Same-length vectors with values in ``{+1,-1}`` (other values are
        projected with the lab packing rule: ``<=0 → -1``, ``>0 → +1``).
    rtol, atol :
        Tolerances for the equality check (exact for true ±1).

    Returns
    -------
    dict with ``n``, ``dot_pm1``, ``dot_xor``, ``hamming``, ``ok``.
    """
    x = np.asarray(x_pm1, dtype=np.float64).ravel()
    w = np.asarray(w_pm1, dtype=np.float64).ravel()
    if x.size != w.size:
        raise ValueError(f"length mismatch: {x.size} vs {w.size}")
    n = int(x.size)
    # Canonical ±1 under lab encoding
    x_b = np.where(x > 0, 1.0, -1.0)
    w_b = np.where(w > 0, 1.0, -1.0)
    dot = pm1_dot(x_b, w_b)
    xp, n_pack = pack_pm1_uint64(x_b)
    wp, n_pack2 = pack_pm1_uint64(w_b)
    assert n_pack == n_pack2 == n
    xor_dot = xor_popcount_dot(xp, wp, n)
    hamming = int(round((n - xor_dot) / 2.0))
    ok = bool(np.isclose(dot, xor_dot, rtol=rtol, atol=atol))
    return {
        "n": n,
        "dot_pm1": dot,
        "dot_xor": xor_dot,
        "hamming": hamming,
        "ok": ok,
        "scale_note": "identity is unscaled; multiply by alpha*beta outside",
    }


def prove_identity_sample(
    n: int,
    *,
    rng: np.random.Generator | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Random ±1 sample of length ``n``; returns ``xnor_dot_identity`` result."""
    if rng is None:
        rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=n)
    x = np.where(bits == 0, 1.0, -1.0)
    bits_w = rng.integers(0, 2, size=n)
    w = np.where(bits_w == 0, 1.0, -1.0)
    return xnor_dot_identity(x, w)
