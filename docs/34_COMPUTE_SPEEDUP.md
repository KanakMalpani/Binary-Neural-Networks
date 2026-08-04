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

## Headline speedups vs NumPy FP32 (compute-only, committed `results/benchmark.json`)

| Shape (B×N×M) | Compute ms | NumPy FP32 ms | S_compute | Err |
|---------------|----------:|--------------:|----------:|----:|
| 128×2048×2048 | 0.61 | 7.34 | **11.99×** | 0 |
| 64×4096×4096 | 0.60 | 14.37 | **23.86×** | 0 |
| 32×8192×8192 | 0.94 | 27.35 | **29.25×** | 0 |

Prior single-thread-era golden at 64×4096×4096 was ~3.7× vs NumPy FP32.

## Thread scaling (compute-only ms → speedup vs 1 thread)

Committed artifact: [`results/benchmark.json`](../results/benchmark.json) /
[`results/benchmark.md`](../results/benchmark.md) (W13.T04). Soft CI check:
`bnn.profile.check_committed_bench_soft_floors` requires ≥2 thread points per
published shape. Re-run does **not** invent new golden shapes.

| Shape | 1 | 2 | 4 | 8 |
|-------|--:|--:|--:|--:|
| 128×2048×2048 | 1.93 (1.00×) | 1.19 (1.62×) | 0.85 (2.28×) | 1.31 (1.48×) |
| 64×4096×4096 | 2.22 (1.00×) | 1.66 (1.34×) | 1.07 (2.08×) | 0.74 (2.99×) |
| 32×8192×8192 | 3.69 (1.00×) | 2.52 (1.46×) | 1.65 (2.23×) | 1.00 (3.68×) |

Plateau / non-monotonic 8-thread on smaller shapes is expected (memory bandwidth /
OpenMP overhead). Prefer `BNN_NUM_THREADS=4`–`8`; avoid blindly matching logical
CPU count. Numbers above are wall-clock on the commit machine — floats need not
match bit-identically elsewhere; conclusions must.

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
