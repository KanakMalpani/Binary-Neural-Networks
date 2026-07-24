# Architecture Decision Record

## Status

Accepted (2026-07-23) for this repository’s MVP.

## Context

Need a working path to reduce neural net inference cost via extreme low-bit methods,
with honest speedups on this CPU-only machine, and a product thesis that survives
adversarial gap analysis.

## Decision

1. **Train** Bi-Real-inspired / BinaryNet-style models in PyTorch with STE (simulation).
2. **Infer** matmuls with **MSVC-native packed XNOR+popcount** (`binary_gemm_u64`).
3. **Position** CPU/edge inference as the primary win surface; treat BitNet b1.58 as the
   LLM production analogue; treat INT8/FP8 as the datacenter default.
4. Keep stem/head FP; BN after binary; optional TernaryLinear for BitNet pedagogy.

## Alternatives considered

| Alternative | Why rejected / deferred |
|-------------|-------------------------|
| Pure “everything ±1” including first/last | Accuracy collapse; tiny compute share |
| Larq-only stack | TF/Keras; archived; less transparent for teaching kernels |
| Brevitas-only | Heavy; still needs export for real speed |
| torchao INT4 as the demo | Correct for GPU LLMs, wrong pedagogical proof of XNOR |
| NumPy-only popcount | Measured **slower** than BLAS (~0.04×) |
| MinGW gcc DLL | 32-bit only → WinError 193 on 64-bit Python |
| Claim 32× wall-clock | Contradicted by Amdahl + measurements (2–5× kernel) |

## Consequences

- Demo proves **real** CPU speedup (2.4–5.0×) and 32× compression.
- Does **not** prove GPU supremacy of BNNs (explicitly out of scope).
- LLM path is **documented**, not fully reproduced (needs bitnet.cpp + large models).

## Accepted non-goals (2026-07-23 gap closure)

These do **not** reopen the perfected thesis; ADR unchanged on stack choice.

| Non-goal | Why accepted | Substitute evidence |
|----------|--------------|---------------------|
| OpenMP / AVX2 multi-thread kernel polish | Single-thread `__popcnt64` already proves CPU XNOR thesis | `results/benchmark.*` (2–9×) |
| Full ImageNet Bi-Real train in-repo | GPU-days; not needed for product decision | CIFAR-10 Bi-Real proxy + Bi-Real/ReActNet papers |
| Board RAPL / Windows energy API | No portable RAPL in stdlib on this host | `results/energy_bound.*` (measured latency × assumed P + lit) |
| Stock vendor NPU native 1-bit | Vendors document INT8/INT4/FP16, not XNOR | `docs/20_NPU_VENDOR_CLOSURE.md` → INT8-first |
