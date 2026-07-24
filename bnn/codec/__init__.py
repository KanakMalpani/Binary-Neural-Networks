"""Portable weight encode/decode — FP/STE Linear → packed ``.bnnpack`` artifacts.

This is the shipable wrap-layer artifact story: encode once, load packed modules
for CPU XNOR/popcount inference (thesis: not a GPU 32× claim).
"""

from __future__ import annotations

from .packfile import (
    BNNPACK_MAGIC,
    BNNPACK_VERSION,
    decode_file,
    decode_to_packed_linear,
    encode_file,
    encode_from_packed_module,
    encode_linear_state,
    encode_model_linears,
    load_bnnpack,
    packed_module_fp_err,
    roundtrip_gemm_err,
    save_bnnpack,
    unpack_binary_pm1,
)

__all__ = [
    "BNNPACK_MAGIC",
    "BNNPACK_VERSION",
    "encode_linear_state",
    "decode_to_packed_linear",
    "encode_from_packed_module",
    "encode_model_linears",
    "encode_file",
    "decode_file",
    "save_bnnpack",
    "load_bnnpack",
    "roundtrip_gemm_err",
    "packed_module_fp_err",
    "unpack_binary_pm1",
]
