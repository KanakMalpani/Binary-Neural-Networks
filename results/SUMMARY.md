# Results summary (this workspace)

_Regenerated: 2026-07-23T16:41:33.443633+00:00_
_Machine: Windows-11-10.0.26200-SP0 | torch 2.12.0+cpu | CUDA=False_

## Kernel (CPU packed XNOR)

| Shape | S vs NumPy FP32 | S vs Torch FP32 | Err |
|-------|----------------:|----------------:|----:|
| 128×2048×2048 | 6.06 | 0.94 | 0 |
| 64×4096×4096 | 2.14 | 1.06 | 0 |
| 32×8192×8192 | 5.61 | 2.86 | 0 |

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

## CIFAR-10 proxy

- FP32: **61.90%**
- Binary Bi-Real: **52.85%**
- Gap: **9.05 pp**
Source: `cifar10_proxy.json`.

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

Gap closure: `docs/19_GAP_CLOSURE_REPORT.md`. Completion: `docs/22_COMPLETION_REPORT.md`.
