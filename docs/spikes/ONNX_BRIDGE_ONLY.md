# Spike: ONNX / ORT path for packed BNN (W5.T07 / M3)

| Field | Value |
|-------|-------|
| **Status** | CLOSED — bridge-only (no in-tree ORT custom op) |
| **Date** | 2026-08-04 |
| **Related** | W5.T05 `.bnnpack` v2, W5.T06 safetensors, M3 |

## Question

Should the lab ship an ONNX Runtime **custom op** for XNOR-popcount / ternary
bitplane GEMM so `.bnnpack` graphs run under ORT / ORT-Web?

## Spike findings

1. **Portable artifact already exists:** `.bnnpack` v2 (+ optional safetensors
   export) carries packed Linear / ternary / Conv2d tensors with hashes.
   Consumers load via `bnn.codec` on CPU — thesis-aligned path.
2. **ORT custom ops** need a maintained C++/CUDA extension, ABI pins per ORT
   version, and a second correctness surface. Cost is XL vs lab size; no
   consumer demand measured with dual metrics yet.
3. **Honest deploy bridges** already documented:
   - torchao / bitsandbytes for GPU INT4/FP8 (not binary 32×)
   - GGUF / bitnet.cpp for LLM serve (`scripts/bridges/`, docs/23)
   - `.bnnpack` encode/decode for packed CPU/edge pedagogy
4. **Thin path considered and declined for v1.0:** export FP dequant graph to
   ONNX (Conv/MatMul only). That loses the packed-kernel story and invites
   fake “binary ONNX” marketing. Prefer explicit bridges.

## Decision (unchanged, spike refreshed)

**Bridge-only.** Do **not** land a full ORT custom op in-tree for v1.0.

Preferred consumer order:

1. `bnn encode` / `bnn decode` / `bnn.codec` for packed CPU
2. Safetensors side-car when HF tooling expects it (`export_bnnpack_safetensors`)
3. Documented bridges (torchao, GGUF, bitnet.cpp) for production serving
4. Revisit ORT custom ops only if a named consumer funds measured dual-metric
   acceptance (latency + compression; no GPU 32× from `sign()`)

## Non-goals

- Claiming ONNX “binary inference” without packed kernels
- Vendoring ORT extension build matrices into default CI

## Exit

Spike complete. ROADMAP W5.T07 remains “decision executed (explicit defer)”;
this note is the living consumer doc.
