# Spike note — ARM NEON packed GEMM (W2.T04)

| Field | Value |
|-------|-------|
| **Status** | **DELIVERED** — see [`../41_PORTABLE_SIMD_KERNEL.md`](../41_PORTABLE_SIMD_KERNEL.md) |
| **Date** | 2026-07-25 (note) · delivered with portable SIMD |
| **Blocker** | ~~No aarch64 CI runner~~ — resolved via GitHub `ubuntu-24.04-arm` + `macos-latest` |
| **Acceptance** | Met: `vcntq_u8`/`vpadalq_u8` NEON path + aarch64/macOS CI asserting err=0 |

## Intent (historical)

Edge phones / Apple Silicon are the honest BNN deployment story. At spike-write time the matrix was:

- **Windows x64** — MSVC OpenMP DLL (primary lab path)
- **Linux x64** — GCC `.so` (CI hard gate as of v0.3)
- **macOS / ARM** — NumPy correctness fallback only ← **superseded**

## Delivered (current)

1. `#ifdef` / runtime NEON path in `bnn/kernels/binary_gemm.c` using
   `vcntq_u8` / `vpadalq_u8` (128-bit words).
2. Scalar / `__builtin_popcountll` fallback kept for correctness.
3. `compile_native` on Darwin + aarch64 Linux (Clang/GCC `-O3 -shared -fPIC`, OpenMP when present).
4. CI: `portability` job on `ubuntu-24.04-arm` and `macos-latest` (plus forced `BNN_KERNEL=scalar`).
5. Dual-metric bench only — no theory-as-latency claims.

## Interim acceptance (met → closed)

- Documented in `docs/COMPATIBILITY_MATRIX.md` (native NEON on arm64)
- Tests pass via native **and** NumPy path
- ROADMAP W2.T04 flipped to `[x]`

## Non-goals

- Stock phone NPU 1-bit (vendor INT8/INT4) — see `docs/20`
- Claiming NEON speedups without measured wall-clock on target silicon
