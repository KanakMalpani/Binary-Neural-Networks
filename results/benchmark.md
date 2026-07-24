# Kernel benchmark (CPU, fair protocol)

Weights pre-packed once. `compute` = GEMM only; `e2e` = pack activations + GEMM.

| Shape | Native? | Compute ms | E2E ms | NumPy FP32 | Torch FP32 | Fake-bin | S_compute | S_e2e | Fake/FP | Theory↓ | Err |
|-------|---------|------------|--------|------------|------------|----------|----------|-------|---------|---------|-----|
| 128×2048×2048 | True | 5.42 | 8.19 | 16.47 | 6.20 | 6.90 | 3.04× | 2.01× | 1.11 | 64× | 0 |
| 64×4096×4096 | True | 11.44 | 15.36 | 42.85 | 26.17 | 42.01 | 3.75× | 2.79× | 1.61 | 64× | 0 |
| 32×8192×8192 | True | 31.76 | 31.12 | 133.11 | 55.85 | 79.39 | 4.19× | 4.28× | 1.42 | 64× | 0 |

## Interpretation

- **S_compute** is the honest kernel win with deployed packed weights.
- **S_e2e** includes activation packing (still usually >1× at large N).
- **Fake-binary > 1** means `sign`+FP GEMM is slower — simulation ≠ acceleration.
- Theory ~64× is word-op reduction, not wall-clock.
