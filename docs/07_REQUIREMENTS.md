# Requirements Spec (derived from first principles + measurements)

## Problem

Reduce **inference** time and memory of neural networks on **bandwidth-bound**
hardware by extreme low-bit weights/activations **with real packed kernels**,
without catastrophic accuracy loss.

## Non-goals

- 32× wall-clock on unmodified PyTorch CUDA
- Faster *training* via binarization
- Replacing datacenter FP8/INT4 stacks for H100 serving
- Claiming ImageNet SOTA from the MNIST demo

---

## Must (P0)

| ID | Requirement | Rationale / evidence |
|----|-------------|----------------------|
| M1 | Separate **simulation training** from **packed inference** | Fake `sign()`+FP is slower (measured 1.8–2.3×) |
| M2 | Provide packed binary GEMM with numerical match to ±1 FP GEMM | `err=0` on validate_native |
| M3 | Keep first/last layers higher precision | Larq / BNN literature; capacity |
| M4 | BatchNorm (or equivalent) after binary layers | Training fails without |
| M5 | Report theoretical ops **and** wall-clock separately | Measurement gap closure |
| M6 | Document CPU vs GPU applicability | GPU Tensor Cores dominate binary on commodity NVIDIA |
| M7 | Clip latent weights + STE for trainability | BinaryNet practice |
| M8 | Reproducible scripts: train, bench, export_check | Definition of done |

## Should (P1)

| ID | Requirement | Rationale |
|----|-------------|-----------|
| S1 | Bi-Real-style FP residuals for CNN path | Accuracy recovery (ECCV 2018) |
| S2 | TernaryLinear (BitNet-style) module | Better LLM thesis than pure ±1 |
| S3 | Native popcount kernel on this machine | Measured 2.4–5.0× vs FP32 GEMM |
| S4 | Adam + BN momentum ~0.9 | Larq training guide |
| S5 | Compression assert ≈32× for binary packs | Catch export bugs |

## Could (P2)

| ID | Requirement |
|----|-------------|
| C1 | OpenMP / AVX2 SIMD popcount for higher \(S\) |
| C2 | bitnet.cpp integration for LLM demo |
| C3 | Distillation / ReActNet RSign |
| C4 | Larq Compute Engine / Android deploy |
| C5 | Energy counters (RAPL) |

## Success metrics

| Metric | MVP target | Credible target |
|--------|------------|-----------------|
| Binary GEMM correctness | max abs err = 0 vs ±1 FP | same |
| Kernel speedup vs FP32 (N≥4096) | ≥2× (**met: 3.6–9.3×**) | ≥5× SIMD/OpenMP vs torch |
| Weight compression | 32× binary layers | + end-to-end model size report |
| MNIST binary MLP acc | ≥95% within 3 epochs if FP≥97% | gap ≤2 pp |
| Docs | formulas + citations + failure register | ADR + roadmap |

## Constraints (this workspace)

- PyTorch **CPU-only** (no CUDA) → CPU kernels are the right proof surface
- MinGW gcc is 32-bit → use **MSVC x64** for native DLL
- Python 3.14 → numba unavailable → C ctypes path required
