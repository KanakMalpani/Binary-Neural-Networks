"""Portable weight encode/decode — FP/STE Linear → packed ``.bnnpack`` artifacts.

This is the shipable wrap-layer artifact story: encode once, load packed modules
for CPU XNOR/popcount inference (thesis: not a GPU 32× claim).

v2 adds ternary + Conv2d blobs, content hashes, and optional safetensors export.
"""

from __future__ import annotations

from .packfile import (
    BNNPACK_MAGIC,
    BNNPACK_VERSION,
    BNNPACK_VERSION_V1,
    BNNPACK_VERSION_V2,
    KIND_BINARY_CONV,
    KIND_BINARY_XNOR,
    KIND_CODE,
    KIND_TERNARY,
    SUPPORTED_VERSIONS,
    content_sha256_tensor,
    decode_file,
    decode_layer,
    decode_to_packed_conv,
    decode_to_packed_linear,
    decode_to_ternary_linear,
    encode_conv_state,
    encode_file,
    encode_from_packed_conv,
    encode_from_packed_module,
    encode_from_ternary_module,
    encode_linear_state,
    encode_model_linears,
    encode_ternary_state,
    load_bnnpack,
    packed_module_fp_err,
    roundtrip_gemm_err,
    save_bnnpack,
    unpack_binary_pm1,
    verify_layer_hashes,
)
from .safetensors_export import (
    bnnpack_tensors_for_safetensors,
    export_bnnpack_safetensors,
)

__all__ = [
    "BNNPACK_MAGIC",
    "BNNPACK_VERSION",
    "BNNPACK_VERSION_V1",
    "BNNPACK_VERSION_V2",
    "KIND_BINARY_CONV",
    "KIND_BINARY_XNOR",
    "KIND_CODE",
    "KIND_TERNARY",
    "SUPPORTED_VERSIONS",
    "content_sha256_tensor",
    "encode_linear_state",
    "encode_ternary_state",
    "encode_conv_state",
    "decode_to_packed_linear",
    "decode_to_ternary_linear",
    "decode_to_packed_conv",
    "decode_layer",
    "encode_from_packed_module",
    "encode_from_ternary_module",
    "encode_from_packed_conv",
    "encode_model_linears",
    "encode_file",
    "decode_file",
    "save_bnnpack",
    "load_bnnpack",
    "verify_layer_hashes",
    "roundtrip_gemm_err",
    "packed_module_fp_err",
    "unpack_binary_pm1",
    "bnnpack_tensors_for_safetensors",
    "export_bnnpack_safetensors",
]
