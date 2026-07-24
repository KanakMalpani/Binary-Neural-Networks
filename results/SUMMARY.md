# Results summary (this workspace)

_Regenerated: 2026-07-24T13:27:10.113341+00:00_
_Machine: Windows-11-10.0.26200-SP0 | torch 2.12.0+cpu | CUDA=False_

## Kernel (CPU packed XNOR)

| Shape | S vs NumPy FP32 | S vs Torch FP32 | Err |
|-------|----------------:|----------------:|----:|
| 128×2048×2048 | 5.28 | 2.31 | 0 |
| 64×4096×4096 | 8.05 | 3.73 | 0 |
| 32×8192×8192 | 13.96 | 7.55 | 0 |

Compression: **32.0×**. Source: `benchmark.json`.

## MNIST

| Model | Acc % |
|-------|------:|
| fp32_mlp | 97.67 |
| binary_mlp | 96.36 |
| ternary_mlp | 97.16 |
| fp32_cnn | 96.61 |
| binary_cnn | 94.79 |

Source: `train_results.json`.

## Image (CIFAR-10 Bi-Real)

- FP32 CNN: **71.14%**
- Binary Bi-Real: **61.14%**
- Gap: **10.00 pp**
Source: `image_cifar.json`. Tutorial: `docs/tutorials/04_image_cifar.md`.

## Audio (synthetic tones)

- FP32 CNN: **94.50%**
- Binary CNN: **96.00%**
- Gap: **-1.50 pp**
Source: `audio_synth.json`. **Not production ASR** — INT8 Whisper/ORT for real speech. Tutorial: `docs/tutorials/05_audio.md`.

## Wrap / energy / robustness

- Wrap e2e latency: FP **21.55** ms → wrapped **18.65** ms (e2e **1.16×**)
- Weight compression (replaced layers): **32.0×** (exact bit-pack)
- Layer gemm_only vs torch Linear: **2.12×** (kernel ROI)
- Output cosine vs FP: **0.283** (low without QAT is expected — not a transparent wrap)
- Energy (latency-only, same power proxy): **1.16×** (`energy_bound.json`)
- FGSM fp32_mlp: clean 97.08% → 62.99% (drop 34.09 pp)
- FGSM binary_mlp: clean 95.96% → 60.38% (drop 35.58 pp)

## Honesty / dual reporting

| Quantity | Meaning | Do not claim as |
|----------|---------|-----------------|
| Weight compression **32×** | Bit-pack bytes | e2e latency |
| Theoretical word reduction ~64× | XNOR-popcount ops | wall-clock |
| Kernel speedup (bench) | Prepacked GEMM vs NumPy/Torch FP | full-model FPS |
| E2E wrap speedup | Whole forward | quality-preserving wrap |

Amdahl: \(S_{e2e}=\frac{1}{(1-f)+f/S_{kernel}}\). Fake `sign()`+torch Linear is often **slower** than FP32 on GPU.

Repro gates: `tests/golden_floors.json` · `bnn repro` · [`REPRODUCIBILITY.md`](../REPRODUCIBILITY.md).

More: `docs/19_GAP_CLOSURE_REPORT.md`, `docs/28_IMAGE_AUDIO_COMPLETION.md`, `docs/29_FINAL_COMPLETION.md`, `docs/31_QUALITY_UPGRADE.md`.
