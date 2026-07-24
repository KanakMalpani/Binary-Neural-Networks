"""Ultra wrap layer: hybrid low-bit policies over existing FP models.

Public surface stays compatible with ``bnn.wrapper`` / ``from bnn import wrap_model``.
"""

from __future__ import annotations

from .api import (
    WrapMode,
    WrapPolicy,
    WrapReport,
    attach_effectiveness,
    model_param_bytes,
    wrap_conv_modules,
    wrap_linear_modules,
    wrap_model,
)
from .calibrate import CalibConfig, calibrate_linear_scales, percentile_scale, absmean_scale
from .metrics import EffectivenessReport, measure_agreement, drop_in_ok
from .packed_linear import (
    BinaryWeightOnlyDequantLinear,
    PackedBinaryConv2d,
    PackedBinaryXNORLinear,
    TernaryWeightOnlyLinear,
    absmean_ternary,
    sign_pm1,
)
from .policy import (
    DEFAULT_SKIP,
    HYBRID_FFN_ALLOW,
    HYBRID_FFN_SKIP,
    HardwareInfo,
    PolicyDecision,
    recommend_wrap_policy,
    resolve_skip_list,
    select_linears,
)
from .qat import light_qat_recover
from .schema import (
    SCHEMA_ID,
    SCHEMA_VERSION,
    envelope,
    is_valid_optimise_report,
    validate_optimise_report,
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
    "CalibConfig",
    "calibrate_linear_scales",
    "percentile_scale",
    "absmean_scale",
    "EffectivenessReport",
    "measure_agreement",
    "drop_in_ok",
    "DEFAULT_SKIP",
    "HYBRID_FFN_ALLOW",
    "HYBRID_FFN_SKIP",
    "HardwareInfo",
    "PolicyDecision",
    "recommend_wrap_policy",
    "resolve_skip_list",
    "select_linears",
    "light_qat_recover",
    "attach_effectiveness",
    "SCHEMA_ID",
    "SCHEMA_VERSION",
    "envelope",
    "validate_optimise_report",
    "is_valid_optimise_report",
]
