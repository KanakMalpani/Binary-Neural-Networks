# 34 — Compute speedup (OpenMP / pack / ternary)

**Date:** 2026-07-24  
**Machine:** Windows x64, MSVC `/O2 /openmp`, hardware `__popcnt64`  
**Protocol:** `scripts/benchmark.py` — weights pre-packed once; warmup=5; reps=10; `BNN_NUM_THREADS=4` for headline compute; thread sweep 1/2/4/8.

Thesis lock unchanged: **32× is packing density**, not a guaranteed e2e wall-clock claim. Numbers below are **kernel** measurements.

## Before → after (this pass)

| Item | Before | After |
|------|--------|-------|
| Native GEMM | Single-thread (or OpenMP stub without thread API) | OpenMP over output rows + `BNN_NUM_THREADS` / `--threads` |
| Pack | Multiply-sum over 64 bits | `numpy.packbits` little-endian uint64 |
| Ternary compute | Dequant → FP GEMM only | Bitplane `ternary_gemm_u64` (native) + NumPy fallback |
| Compile | MSVC vcvars list | + vswhere, `/DEF` exports, Linux `-fopenmp` path |
| Bench | Fixed sizes, no scaling curve | Warmups, pack-vs-compute, thread curve, ternary |

## Headline speedups vs NumPy FP32 (compute-only, 4 threads)

| Shape (B×N×M) | Compute ms | NumPy FP32 ms | S_compute | Err |
|---------------|----------:|--------------:|----------:|----:|
| 128×2048×2048 | 1.79 | 9.43 | **5.28×** | 0 |
| 64×4096×4096 | 2.92 | 23.49 | **8.05×** | 0 |
| 32×8192×8192 | 6.04 | 84.34 | **13.96×** | 0 |

Prior committed golden (single-thread era) at 64×4096×4096 was ~3.7× vs NumPy FP32.

## Thread scaling (compute-only ms → speedup vs 1 thread)

| Shape | 1 | 2 | 4 | 8 |
|-------|--:|--:|--:|--:|
| 128×2048×2048 | 5.22 (1.00×) | 2.90 (1.80×) | 1.98 (2.63×) | 1.80 (2.91×) |
| 64×4096×4096 | 7.36 (1.00×) | 5.04 (1.46×) | 3.33 (2.21×) | 2.51 (2.93×) |
| 32×8192×8192 | 16.53 (1.00×) | 10.27 (1.61×) | 6.53 (2.53×) | 5.69 (2.91×) |

Plateau near ~3× on this machine is expected (memory bandwidth / OpenMP overhead). Prefer `BNN_NUM_THREADS=4`–`8`; avoid blindly matching logical CPU count.

## Ternary bitplane vs dequant FP (same shapes, err=0)

| Shape | Bitplane ms | Dequant FP ms | S |
|-------|------------:|--------------:|--:|
| 128×2048×2048 | 3.31 | 21.54 | **6.50×** |
| 64×4096×4096 | 4.56 | 54.64 | **11.99×** |
| 32×8192×8192 | 11.20 | 188.70 | **16.85×** |

Bitplane path requires **±1 activations**. Full-precision X still uses dequant FP (honest pedagogy).

## Pack once vs compute

| Shape | Pack W ms | Pack X ms | Compute ms |
|-------|----------:|----------:|-----------:|
| 128×2048×2048 | 5.31 | 0.22 | 1.79 |
| 64×4096×4096 | 20.13 | 0.33 | 2.92 |
| 32×8192×8192 | 90.67 | 0.39 | 6.04 |

Weights are deploy-time; fair inference benches **must not** re-pack W every call.

## How to reproduce

```bash
python -m bnn.kernels.compile_native --force
set BNN_NUM_THREADS=4
python scripts/benchmark.py --reps 10 --warmup 5 --threads 1,2,4,8
bnn validate-native   # err_nat=0
```

Artifacts: `results/benchmark.json`, `results/benchmark.md`.

## Other gaps closed here

- `binary_gemm_set_num_threads` / `get` / `openmp_enabled` + `.def` exports (MSVC)
- Linux/GCC `-fopenmp` in `compile_native.py`
- Eval-only `fuse_bireal_bn_` (BN → binary α/bias)
- DataLoader `num_workers=0` on Windows (`BNN_NUM_WORKERS` override)
- G11 (OpenMP polish) no longer “deferred non-goal” for this lab — shipped with measured curve
