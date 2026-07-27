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

Optional: Homebrew `libomp` improves OpenMP scaling (`brew install libomp`).
If `.so` fails to load, NumPy path remains correct.

## Accelerate

PyTorch may use Accelerate BLAS for FP baselines. Packed BNN path uses the
portable native kernel (or NumPy fallback) — do not conflate Accelerate FP GEMM
with XNOR-popcount wall-clock.
