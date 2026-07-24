# Results summary (this workspace)

_Regenerated: 2026-07-24T10:01:19.945536+00:00_
_Machine: Windows-11-10.0.26200-SP0 | torch 2.12.0+cpu | CUDA=False_

## Kernel (CPU packed XNOR)

| Shape | S vs NumPy FP32 | S vs Torch FP32 | Err |
|-------|----------------:|----------------:|----:|
| 128×2048×2048 | 3.04 | 1.14 | 0 |
| 64×4096×4096 | 3.75 | 2.29 | 0 |
| 32×8192×8192 | 4.19 | 1.76 | 0 |

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

- Wrap e2e: FP 21.553476666789116 ms → binary 18.65172999993471 ms (compression None)
- Energy latency-only reduction: **1.1555752022393935×** (`energy_bound.json`)
- FGSM fp32_mlp: clean 97.08% → 62.99% (drop 34.09 pp)
- FGSM binary_mlp: clean 95.96% → 60.38% (drop 35.58 pp)

## Formula reminder

\[
S_{e2e}=\frac{1}{(1-f)+f/S_{kernel}},\quad R_{arith}\approx 64,\quad compress=32\times
\]

Do not advertise \(R_{arith}\) as wall-clock.

Gap closure: `docs/19_GAP_CLOSURE_REPORT.md`. Image+audio: `docs/28_IMAGE_AUDIO_COMPLETION.md`. Final: `docs/29_FINAL_COMPLETION.md`.
