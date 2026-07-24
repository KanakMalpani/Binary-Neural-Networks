# Moonshot note — AVX-512 VPOPCNTDQ (W2.T05)

| Field | Value |
|-------|-------|
| **Status** | **Open moonshot** (not blocking v0.3 / Phase C) |
| **Date** | 2026-07-25 |
| **Blocker** | Needs AVX-512_VPOPCNTDQ CPU + dispatch harness; not on default GHA runners |
| **Acceptance leftover** | Optional runtime dispatch + dual-metric bench vs AVX2/scalar |

## Intent

`VPOPCNTDQ` can accelerate packed popcount on Ice Lake / Zen4-class CPUs.
This is a **perf** moonshot, not a correctness gate (NumPy / scalar / `__popcnt64`
already err=0).

## When to pick up

- After Linux native CI stays green
- When a machine with AVX-512_VPOPCNTDQ is available for dual-metric evidence
- Must keep fallback paths; never require AVX-512 for install

## Explicit non-claim

Do not advertise “AVX-512 32×” without published wall-clock on listed shapes
(`docs/BENCH_SHAPES.md` + `docs/FAIR_EVAL_PROTOCOL.md`).
