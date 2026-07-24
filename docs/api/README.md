# Public API reference (`bnn`)

Install: `pip install -e ".[dev]" -c constraints.txt`  
Version: `import bnn; print(bnn.__version__)` · CLI: `bnn --version`

## Core (`import bnn`)

| Symbol | Role |
|--------|------|
| `BinaryLinear`, `BinaryConv2d`, `BiRealBlock`, `TernaryLinear` | STE training layers |
| `binary_sign`, `ternary_weight`, `clip_weights_` | Estimators / clip |
| `build_model`, `count_parameters` | MNIST zoo |
| `wrap_model`, `wrap_linear_modules`, `model_param_bytes` | Inference wrap |
| `save_checkpoint`, `load_checkpoint` | Latent STE weights (trusted paths only) |
| `save_packed_linears`, `load_packed_linears`, `pack_linear_weight` | Packed blobs |
| `set_repro_seed` | Seeds + deterministic/CPU policy for goldens |

```python
import bnn
from bnn import BinaryLinear, wrap_model, set_repro_seed

set_repro_seed(0, deterministic=True, force_cpu=True)
```

## Kernels (`bnn.kernels`)

| Symbol | Role |
|--------|------|
| `pack_binary_pm1` | ±1 → uint64 words |
| `binary_gemm_packed` | XNOR-popcount GEMM (native if available) |
| `binary_gemm_numpy_prepacked` / `binary_gemm_native_prepacked` | Explicit paths |
| `native_kernel_available` | DLL/SO probe |
| `theoretical_ops` | Theory (≠ wall-clock) |
| `pack_ternary_2bit` / `unpack_ternary_2bit` | Ternary pedagogy |

Compile (Windows MSVC x64): `python -m bnn.kernels.compile_native`

## Vision (`bnn.vision`)

`FP32CIFARCNN`, `BinaryCIFARCNN`, `TinyBinaryViT`, `build_vision_model`,
`check_imagenet_folder` (layout stub; full ImageNet train is a non-goal).

## Audio (`bnn.audio`)

`get_audio_loaders`, `synthesize_tone`, `waveform_to_features`, `build_audio_model`.
Synthetic tones only — not production ASR.

## Paths / logging

| Module | Role |
|--------|------|
| `bnn.paths.resolve_under` | Reject path traversal outside a root |
| `bnn.logutil.info/warn/error` | Flushing stdout/stderr conventions |

## Seq + codec

| Module | Role |
|--------|------|
| `bnn.seq` | `BinaryTransformerEncoder`, `BinaryTransformerDecoder`, `BinarySeq2Seq`, `BinaryAutoEncoder` |
| `bnn.codec` | `encode_linear_state`, `decode_to_packed_linear`, `encode_file` / `.bnnpack` |
| `bnn.profile` | `profile_packed_linear` pack/gemm/overhead breakdown |

## CLI

```bat
bnn --help
bnn --version
bnn repro                 # fast golden verify
bnn compile-native
bnn validate-native       # exit 2 if DLL missing
bnn export-check
bnn bench | train | train-image | train-audio | wrap
bnn train-seq2seq | encode | decode | wrap-transformer | profile
bnn eval-suite | recommend --goal edge-vision
```

Also: `python -m bnn <command>`.

## Security notes

- Prefer NPZ CIFAR (`data/cifar10_hf/`); pickle batches only from the official
  Toronto layout under your `data_dir`.
- `torch.load` prefers `weights_only=True`; legacy meta falls back with a warning —
  never load untrusted checkpoints.
- Dataset trees under `data/` are gitignored; do not commit them.
