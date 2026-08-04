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
from .calibrate import (
    CalibConfig,
    CalibReport,
    LayerScale,
    absmean_scale,
    calibrate,
    calibrate_linear_scales,
    calibrate_model,
    percentile_scale,
)
from .distill import DistillConfig, DistillReport, distill_binary_student, distill_from_teacher_copy
from .fuse import (
    FuseReport,
    fuse_bn_for_wrap_,
    fuse_linear_bn1d_,
    fuse_sequential_linear_bn_,
)
from .guardrails import GuardrailVerdict, check_linear_wrap_guardrails
from .metrics import (
    EffectivenessReport,
    drop_in_ok,
    measure_agreement,
    unmeasured_effectiveness,
)
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
from .sensitivity import (
    LayerSensitivity,
    ModeSearchReport,
    SearchAssignment,
    SensitivityReport,
    apply_sensitivity_skips,
    score_layer_sensitivity,
    search_layer_modes,
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
    "CalibReport",
    "LayerScale",
    "calibrate",
    "calibrate_model",
    "calibrate_linear_scales",
    "percentile_scale",
    "absmean_scale",
    "EffectivenessReport",
    "measure_agreement",
    "unmeasured_effectiveness",
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
    "DistillConfig",
    "DistillReport",
    "distill_binary_student",
    "distill_from_teacher_copy",
    "FuseReport",
    "fuse_bn_for_wrap_",
    "fuse_linear_bn1d_",
    "fuse_sequential_linear_bn_",
    "attach_effectiveness",
    "SCHEMA_ID",
    "SCHEMA_VERSION",
    "envelope",
    "validate_optimise_report",
    "is_valid_optimise_report",
    "LayerSensitivity",
    "SensitivityReport",
    "score_layer_sensitivity",
    "apply_sensitivity_skips",
    "search_layer_modes",
    "ModeSearchReport",
    "SearchAssignment",
    "GuardrailVerdict",
    "check_linear_wrap_guardrails",
]
