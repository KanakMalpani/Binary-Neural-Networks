# API stub

Public package: **`bnn`** (`pip install -e .`).

## Core

| Symbol | Module | Role |
|--------|--------|------|
| `BinaryLinear`, `BinaryConv2d`, `BiRealBlock`, `TernaryLinear` | `bnn.layers` | STE training layers |
| `binary_sign`, `ternary_weight`, `clip_weights_` | `bnn.ste` | Estimators |
| `build_model`, `count_parameters` | `bnn.models` | Zoo |
| `wrap_model`, `wrap_linear_modules` | `bnn.wrapper` | Inference wrap |
| `save_checkpoint`, `load_checkpoint`, `save_packed_linears` | `bnn.export` | Checkpoints |
| `write_summary` | `bnn.eval_report` | SUMMARY.md |

## Kernels

| Symbol | Module |
|--------|--------|
| `pack_binary_pm1`, `binary_gemm_packed`, `native_kernel_available` | `bnn.kernels.packed` |
| `pack_ternary_2bit`, `unpack_ternary_2bit` | `bnn.kernels.ternary_pack` |
| `compile_native` | `python -m bnn.kernels.compile_native` |

## CLI

```bat
bnn compile-native | validate-native | bench | export-check
bnn train | train-cifar | wrap | energy-bound | eval-suite | recommend
```

Thesis: inference on CPU/edge with real packed kernels — not `sign()` for GPU 32×.
