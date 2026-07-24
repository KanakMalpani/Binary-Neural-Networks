# Spike note — ARM NEON packed GEMM (W2.T04)

| Field | Value |
|-------|-------|
| **Status** | **Deferred** (documented spike) |
| **Date** | 2026-07-25 |
| **Blocker** | No aarch64 CI runner / Apple Silicon self-hosted agent in this lab |
| **Acceptance leftover** | Portable NEON (or SVE) popcount GEMM + CI job on aarch64 with err=0 |

## Intent

Edge phones / Apple Silicon are the honest BNN deployment story. Today:

- **Windows x64** — MSVC OpenMP DLL (primary lab path)
- **Linux x64** — GCC `.so` (CI hard gate as of v0.3)
- **macOS / ARM** — **NumPy correctness fallback** only

## Spike plan (when hardware available)

1. Add `#ifdef __ARM_NEON` path in `bnn/kernels/binary_gemm.c` using
   `vcntq_u8` / pairwise sums (or ACLE popcount) for 128-bit words.
2. Keep scalar / `__builtin_popcountll` fallback for correctness.
3. `compile_native` on Darwin: `clang -O3 -shared -fPIC` (+ `-fopenmp` if present).
4. CI: self-hosted aarch64 **or** GitHub `ubuntu-24.04-arm` when available to org.
5. Dual-metric bench only — no theory-as-latency claims.

## Interim acceptance

- Documented in `docs/COMPATIBILITY_MATRIX.md`
- Tests pass via NumPy path on any arch
- This spike note linked from ROADMAP W2.T04 as `[~]` deferred-with-plan

## Non-goals

- Stock phone NPU 1-bit (vendor INT8/INT4) — see `docs/20`
- Claiming NEON speedups without measured wall-clock on target silicon
