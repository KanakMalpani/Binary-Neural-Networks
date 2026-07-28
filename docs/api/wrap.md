# Wrapping models
Replace `nn.Linear` / `nn.Conv2d` with packed equivalents.

::: bnn.wrap.api.wrap_model
::: bnn.wrap.api.wrap_linear_modules
::: bnn.wrap.api.model_param_bytes

## Packed modules

::: bnn.wrap.packed_linear.PackedBinaryXNORLinear
::: bnn.wrap.packed_linear.TernaryWeightOnlyLinear
::: bnn.wrap.packed_linear.BinaryWeightOnlyDequantLinear
::: bnn.wrap.packed_linear.PackedBinaryConv2d

## Policy, calibration, guardrails

::: bnn.wrap.policy.recommend_wrap_policy
::: bnn.wrap.calibrate.CalibConfig
::: bnn.wrap.calibrate.calibrate_linear_scales
::: bnn.wrap.guardrails.check_linear_wrap_guardrails
::: bnn.wrap.qat.light_qat_recover
::: bnn.wrap.sensitivity.score_layer_sensitivity
::: bnn.wrap.sensitivity.search_layer_modes
::: bnn.wrap.sensitivity.ModeSearchReport
::: bnn.wrap.sensitivity.SearchAssignment
