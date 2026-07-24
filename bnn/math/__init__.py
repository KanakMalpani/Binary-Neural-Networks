"""Binary / ternary linear-algebra identities and effectiveness calculators.

This package is the *math path* for the lab: pure-Python / NumPy proofs that
packed XNOR–popcount equals the ±1 inner product, packing bijections, and
back-of-envelope ops/byte effectiveness — independent of the C GEMM owner.
"""

from __future__ import annotations

from .effectiveness import (
    amdahl_speedup,
    bytes_per_mac,
    effective_ops_per_mac,
    effectiveness_report,
    when_binary_less_effective,
)
from .identity import (
    hamming_to_dot,
    pm1_dot,
    prove_identity_sample,
    xnor_dot_identity,
    xor_popcount_dot,
)
from .packing import (
    pack_pm1_uint64,
    pack_unpack_roundtrip,
    unpack_pm1_uint64,
)
from .reference import binary_dot_ref, binary_gemm_ref
from .ternary_math import (
    absmean_scale,
    ternary_accuracy_per_bit,
    ternary_quantize_pm1_0,
)

__all__ = [
    "amdahl_speedup",
    "bytes_per_mac",
    "effective_ops_per_mac",
    "effectiveness_report",
    "when_binary_less_effective",
    "hamming_to_dot",
    "pm1_dot",
    "prove_identity_sample",
    "xnor_dot_identity",
    "xor_popcount_dot",
    "pack_pm1_uint64",
    "pack_unpack_roundtrip",
    "unpack_pm1_uint64",
    "binary_dot_ref",
    "binary_gemm_ref",
    "absmean_scale",
    "ternary_accuracy_per_bit",
    "ternary_quantize_pm1_0",
]
