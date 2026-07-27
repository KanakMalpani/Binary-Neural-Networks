# Moonshot note — AVX-512 VPOPCNTDQ (W2.T05)

| Field | Value |
|-------|-------|
| **Status** | **DELIVERED** — see [`../41_PORTABLE_SIMD_KERNEL.md`](../41_PORTABLE_SIMD_KERNEL.md) |
| **Date** | 2026-07-25 (note) · delivered with portable SIMD |
| **Blocker** | ~~Needs dispatch harness~~ — resolved: runtime cpuid+xgetbv dispatch, AVX-512 never required |
| **Acceptance** | Met: runtime dispatch, AVX2 + scalar fallbacks, cross-ISA err=0 gate |

## Intent (historical)

`VPOPCNTDQ` can accelerate packed popcount on Ice Lake / Zen4-class CPUs.
This was framed as a **perf** moonshot, not a correctness gate (NumPy / scalar /
`__popcnt64` already err=0).

## Delivered (current)

- Runtime dispatch: AVX-512 VPOPCNTDQ → AVX2 → NEON → scalar in one portable build
- No `-march=native`; AVX-512 is used when present, **never required** to install/build/run
- Cross-ISA err=0 in `tests/test_native_gemm.py` + CI `BNN_KERNEL=scalar` rerun
- ROADMAP W2.T05 flipped to `[x]`

## Explicit non-claim

Do not advertise “AVX-512 32×” without published wall-clock on listed shapes
(`docs/BENCH_SHAPES.md` + `docs/FAIR_EVAL_PROTOCOL.md`).
