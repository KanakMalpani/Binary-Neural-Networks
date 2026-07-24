# Kernel benchmark (CPU, fair protocol)

Weights pre-packed once. `compute` = GEMM only; `e2e` = pack activations + GEMM.
Warmup=5, reps=10, OpenMP=True.

| Shape | Native? | Compute ms | E2E ms | NumPy FP32 | Torch FP32 | Fake-bin | S_compute | S_e2e | Fake/FP | Theory↓ | Err |
|-------|---------|------------|--------|------------|------------|----------|----------|-------|---------|---------|-----|
| 128×2048×2048 | True | 1.79 | 1.66 | 9.43 | 4.13 | 5.11 | 5.28× | 5.68× | 1.24 | 64× | 0 |
| 64×4096×4096 | True | 2.92 | 3.53 | 23.49 | 10.89 | 16.19 | 8.05× | 6.65× | 1.49 | 64× | 0 |
| 32×8192×8192 | True | 6.04 | 5.59 | 84.34 | 45.61 | 52.88 | 13.96× | 15.08× | 1.16 | 64× | 0 |

## Thread scaling (compute-only, native)

### 128×2048×2048

| Threads | Compute ms | vs 1-thread |
|--------:|-----------:|------------:|
| 1 | 5.22 | 1.00× |
| 2 | 2.90 | 1.80× |
| 4 | 1.98 | 2.63× |
| 8 | 1.80 | 2.91× |

### 64×4096×4096

| Threads | Compute ms | vs 1-thread |
|--------:|-----------:|------------:|
| 1 | 7.36 | 1.00× |
| 2 | 5.04 | 1.46× |
| 4 | 3.33 | 2.21× |
| 8 | 2.51 | 2.93× |

### 32×8192×8192

| Threads | Compute ms | vs 1-thread |
|--------:|-----------:|------------:|
| 1 | 16.53 | 1.00× |
| 2 | 10.27 | 1.61× |
| 4 | 6.53 | 2.53× |
| 8 | 5.69 | 2.91× |

## Pack vs compute

| Shape | Pack W ms | Pack X ms | Compute ms |
|-------|----------:|----------:|-----------:|
| 128×2048×2048 | 5.31 | 0.22 | 1.79 |
| 64×4096×4096 | 20.13 | 0.33 | 2.92 |
| 32×8192×8192 | 90.67 | 0.39 | 6.04 |

## Interpretation

- **S_compute** is the honest kernel win with deployed packed weights.
- **S_e2e** includes activation packing (still usually >1× at large N).
- **Fake-binary > 1** means `sign`+FP GEMM is slower — simulation ≠ acceleration.
- Theory ~64× is word-op reduction, not wall-clock.
- Thread scaling uses OpenMP over output rows; memory-bound shapes may plateau early.
- Ternary bitplane path is for ±1 activations; full-precision X still uses dequant FP.
