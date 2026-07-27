# macOS notes (W14.T05)

| Field | Value |
|-------|-------|
| **Native kernel** | **Supported** — `python -m bnn.kernels.compile_native` (Clang `.so`); runtime NEON on Apple Silicon, AVX2/scalar on Intel |
| **NEON / SIMD** | **Delivered** — [`41_PORTABLE_SIMD_KERNEL.md`](41_PORTABLE_SIMD_KERNEL.md); spike [`spikes/ARM_NEON_SPIKE.md`](spikes/ARM_NEON_SPIKE.md) |
| **CI** | `portability` job: `macos-latest` (arm64) + `macos-15-intel` (x86_64) |
| **Fallback** | NumPy packed GEMM if native build/load fails — correctness preserved |

## Install

```bash
python -m pip install -U pip
pip install -e ".[dev]" -c constraints.txt
python -m bnn.kernels.compile_native
bnn repro
```

**OpenMP is off by default on macOS.** Linking Homebrew `libomp` into a
process that already loaded PyTorch's OpenMP runtime aborts with
`OMP: Error #15` (duplicate libomp). The native build still uses NEON / AVX2;
threading is the only thing disabled. Force OpenMP only if you know a single
runtime is present: `BNN_FORCE_OPENMP=1 python -m bnn.kernels.compile_native`
or `--openmp`. If `.so` fails to load, NumPy path remains correct.

## Accelerate

PyTorch may use Accelerate BLAS for FP baselines. Packed BNN path uses the
portable native kernel (or NumPy fallback) — do not conflate Accelerate FP GEMM
with XNOR-popcount wall-clock.
