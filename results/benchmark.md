# Kernel benchmark (CPU, fair protocol)

Weights pre-packed once. `compute` = GEMM only; `e2e` = pack activations + GEMM.

| Shape | Native? | Compute ms | E2E ms | NumPy FP32 | Torch FP32 | Fake-bin | S_compute | S_e2e | Fake/FP | Theory↓ | Err |
|-------|---------|------------|--------|------------|------------|----------|----------|-------|---------|---------|-----|
| 128×2048×2048 | True | 5.54 | 7.91 | 33.57 | 5.20 | 7.67 | 6.06× | 4.25× | 1.48 | 64× | 0 |
| 64×4096×4096 | True | 12.51 | 11.65 | 26.83 | 13.23 | 24.08 | 2.14× | 2.30× | 1.82 | 64× | 0 |
| 32×8192×8192 | True | 21.32 | 23.07 | 119.52 | 60.98 | 91.99 | 5.61× | 5.18× | 1.51 | 64× | 0 |

## Interpretation

- **S_compute** is the honest kernel win with deployed packed weights.
- **S_e2e** includes activation packing (still usually >1× at large N).
- **Fake-binary > 1** means `sign`+FP GEMM is slower — simulation ≠ acceleration.
- Theory ~64× is word-op reduction, not wall-clock.
