"""Pedagogy WASM / portable binary GEMM (W2.T06).

Python reference matching ``wasm/binary_gemm_wasm.c`` and the JS demo.
This is **not** wired into ``binary_gemm_packed`` — native CPU kernels remain
the production path (see ``docs/41_PORTABLE_SIMD_KERNEL.md``).
"""

from __future__ import annotations

from .gemm import (
    KERNEL_SCALAR,
    KERNEL_SIMD128,
    binary_gemm_wasm_numpy,
    binary_gemm_wasm_prepacked,
    kernel_name,
    set_kernel,
)

__all__ = [
    "KERNEL_SCALAR",
    "KERNEL_SIMD128",
    "binary_gemm_wasm_numpy",
    "binary_gemm_wasm_prepacked",
    "kernel_name",
    "set_kernel",
]
