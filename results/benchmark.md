# Kernel benchmark (CPU, fair protocol)

Weights pre-packed once. `compute` = GEMM only; `e2e` = pack activations + GEMM.
Warmup=5, reps=10, OpenMP=True.

| Shape | Native? | Compute ms | E2E ms | NumPy FP32 | Torch FP32 | Fake-bin | S_compute | S_e2e | Fake/FP | Theory↓ | Err |
|-------|---------|------------|--------|------------|------------|----------|----------|-------|---------|---------|-----|
| 128×2048×2048 | True | 0.61 | 0.89 | 7.34 | 3.64 | 5.23 | 11.99× | 8.20× | 1.44 | 64× | 0 |
| 64×4096×4096 | True | 0.60 | 1.01 | 14.37 | 10.91 | 15.93 | 23.86× | 14.18× | 1.46 | 64× | 0 |
| 32×8192×8192 | True | 0.94 | 1.32 | 27.35 | 47.44 | 61.38 | 29.25× | 20.72× | 1.29 | 64× | 0 |

## Thread scaling (compute-only, native)

### 128×2048×2048

| Threads | Compute ms | vs 1-thread |
|--------:|-----------:|------------:|
| 1 | 1.93 | 1.00× |
| 2 | 1.19 | 1.62× |
| 4 | 0.85 | 2.28× |
| 8 | 1.31 | 1.48× |

### 64×4096×4096

| Threads | Compute ms | vs 1-thread |
|--------:|-----------:|------------:|
| 1 | 2.22 | 1.00× |
| 2 | 1.66 | 1.34× |
| 4 | 1.07 | 2.08× |
| 8 | 0.74 | 2.99× |

### 32×8192×8192

| Threads | Compute ms | vs 1-thread |
|--------:|-----------:|------------:|
| 1 | 3.69 | 1.00× |
| 2 | 2.52 | 1.46× |
| 4 | 1.65 | 2.23× |
| 8 | 1.00 | 3.68× |

## Pack vs compute

| Shape | Pack W ms | Pack X ms | Compute ms |
|-------|----------:|----------:|-----------:|
| 128×2048×2048 | 5.85 | 0.31 | 0.61 |
| 64×4096×4096 | 21.67 | 0.26 | 0.60 |
| 32×8192×8192 | 83.32 | 0.24 | 0.94 |

## Interpretation

- **S_compute** is the honest kernel win with deployed packed weights.
- **S_e2e** includes activation packing (still usually >1× at large N).
- **Fake-binary > 1** means `sign`+FP GEMM is slower — simulation ≠ acceleration.
- Theory ~64× is word-op reduction, not wall-clock.
- Thread scaling uses OpenMP over output rows; memory-bound shapes may plateau early.
- Ternary bitplane path is for ±1 activations; full-precision X still uses dequant FP.
