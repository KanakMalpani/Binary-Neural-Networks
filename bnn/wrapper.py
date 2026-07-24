"""Drop-in wrappers that replace nn.Linear for CPU packed inference.

Back-compat shim — implementation lives in ``bnn.wrap``.

Honest modes
------------
1. ``binary_xnor`` — packed ±1 weights + signed activations → native XNOR GEMM.
2. ``ternary_weight_only`` — absmean ternary weights, FP activations (size; FP GEMM).
3. ``binary_weight_only_dequant`` — packed store, dequant GEMM (anti-pattern for speed).
"""

from __future__ import annotations

from .wrap import (
    DEFAULT_SKIP,
    HYBRID_FFN_ALLOW,
    HYBRID_FFN_SKIP,
    BinaryWeightOnlyDequantLinear,
    CalibConfig,
    EffectivenessReport,
    HardwareInfo,
    PackedBinaryConv2d,
    PackedBinaryXNORLinear,
    PolicyDecision,
    TernaryWeightOnlyLinear,
    WrapMode,
    WrapPolicy,
    WrapReport,
    absmean_ternary,
    attach_effectiveness,
    calibrate_linear_scales,
    drop_in_ok,
    light_qat_recover,
    measure_agreement,
    model_param_bytes,
    recommend_wrap_policy,
    resolve_skip_list,
    sign_pm1,
    wrap_conv_modules,
    wrap_linear_modules,
    wrap_model,
)

__all__ = [
    "WrapMode",
    "WrapPolicy",
    "WrapReport",
    "wrap_model",
    "wrap_linear_modules",
    "wrap_conv_modules",
    "model_param_bytes",
    "PackedBinaryXNORLinear",
    "TernaryWeightOnlyLinear",
    "BinaryWeightOnlyDequantLinear",
    "PackedBinaryConv2d",
    "absmean_ternary",
    "sign_pm1",
    "DEFAULT_SKIP",
    "HYBRID_FFN_ALLOW",
    "HYBRID_FFN_SKIP",
    "resolve_skip_list",
    "CalibConfig",
    "calibrate_linear_scales",
    "EffectivenessReport",
    "measure_agreement",
    "drop_in_ok",
    "attach_effectiveness",
    "HardwareInfo",
    "PolicyDecision",
    "recommend_wrap_policy",
    "light_qat_recover",
]
