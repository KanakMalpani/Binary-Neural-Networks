# Honest Failure Analysis

Every “binary = 32× faster / same accuracy” claim fails somewhere. This document catalogs
**failure modes, evidence, and mitigations** used by this repo’s architecture.

---

## F1 — Accuracy collapse from information loss

**Failure:** `sign(x)` destroys magnitude. Deep stacks of binary layers lose capacity;
ImageNet gaps of **10–20+ points** were common in early BNNs.

**Why:** Each layer’s output is only a Hamming distance; residual high-precision signal is gone.

**Mitigations (proven):**
- **Bi-Real / ReActNet:** full-precision identity shortcuts around binary convs
- Keep **first and last layers** in higher precision (standard in Larq guides)
- **Scaling factors** (XNOR-Net α) to restore dynamic range
- **Ternary weights** (BitNet b1.58) with explicit 0 for gating
- Longer schedules (often 200+ epochs on ImageNet-class tasks)

**This repo:** FP first/last layers; Bi-Real-style residuals on the binary CNN; channel scales.

---

## F2 — Gradient approximation (STE) mismatch

**Failure:** `sign` has zero derivative almost everywhere → plain backprop dies.
Straight-Through Estimator (STE) pretends \(\partial\mathrm{sign}/\partial x \approx 1_{|x|\le1}\).
Gradient mismatch → unstable or suboptimal minima.

**Mitigations:**
- Clip STE to \([-1,1]\) (BinaryNet)
- Bi-Real’s piecewise-polynomial surrogate
- ReActNet RSign / RPReLU (learnable thresholds)
- SURGE (2026): learnable dual-path surrogate gradients
- Weight clip on latent weights (`|w| ≤ 1`)

**This repo:** classic clipped STE + weight clip; documented as a remaining research risk.

---

## F3 — BatchNorm sensitivity

**Failure:** BNNs without BN almost never train. With BN, running stats / momentum can
destabilize because binary activations are discrete and noisy.

**Mitigations:**
- BN (with affine) after every binary layer — non-negotiable in practice
- Slightly lower momentum (Larq often suggests ~0.9)
- Prefer Adam over SGD for latent weights

**This repo:** BatchNorm after binary layers; Adam; BN momentum 0.9.

---

## F4 — First / last layer exceptions

**Failure:** Binarizing the stem (raw pixels / embeddings) or classifier logits destroys
accuracy for little compute savings (those layers are small).

**Mitigation:** Always leave first and last layers FP16/FP32 (or at least ≥8-bit).

**This repo:** enforces this in `BinaryMLP` / `BinaryCNN`.

---

## F5 — Training instability & slow convergence

**Failure:** Loss oscillates; needs more epochs than FP; LR schedules transfer poorly.

**Mitigations:** Adam; smaller LR; optional FP pretrain → binarize; gradient clipping;
progressive quantization (train FP → fake-quantize → hard binary).

---

## F6 — “Fake binary” in frameworks (kernel support gap)

**Failure:** PyTorch `w.sign() @ x.sign()` still uses **FP32 GEMM**. No packing →
**no speedup**, often slowdown (extra sign kernels).

Evidence: arXiv:1911.04477 — custom XNOR kernel beats naive control, but **loses to
cuDNN FP** on GPU unless kernels match vendor quality.

**Mitigations:**
- Packed bit kernels (this repo’s `bnn.kernels.packed`)
- Larq Compute Engine / bitnet.cpp / T-MAC for production
- Never claim wall-clock wins from simulation alone

**This repo:** separates **sim mode** (trainable) from **packed mode** (measured speed).

---

## F7 — GPU Tensor Core reality

**Failure:** On A100/H100, FP16/BF16/FP8/INT8 Tensor Cores dominate. Binary popcount
kernels rarely beat them end-to-end for large matmuls unless extremely carefully engineered.

**When binary/ternary still wins on GPU:**
- Memory-bound decode with packed weights (BitNet-style)
- Huge batch from smaller footprint (BitNet 70B: 8.9× throughput)

**When it loses:**
- Compute-bound large-batch training
- Unoptimized custom CUDA vs cuBLAS/cuDNN

**This environment:** CPU-only PyTorch — the honest place to demo packing wins.

---

## F8 — Amdahl’s law / non-binary ops

**Failure:** Softmax, LayerNorm/RMSNorm, attention scores, embeddings, data loading —
remain FP. End-to-end speedup << layer speedup.

**Mitigation:** Quantize the dominant matmuls; accept residual FP overhead; report both
**kernel** and **end-to-end** numbers (this repo’s benchmark does).

---

## F9 — Deployment mismatch (train graph ≠ inference graph)

**Failure:** Training stores latent FP weights; forgetting to pack/export binary weights
ships a fat FP model.

**Mitigation:** Explicit export (`pack_binary_linear`); dual checkpoints; CI check that
packed size ≈ FP_size/32 for binary layers.

---

## F10 — Ternary vs binary confusion

**Failure:** Marketing “1-bit LLM” when BitNet b1.58 is **ternary** (~1.58 bits).
Pure ±1 LLMs historically lagged FP quality; ternary closed the gap.

**Mitigation:** Use the right tool:
- Extreme edge CNN → binary (+ shortcuts)
- LLM quality parity → ternary BitLinear

**This repo:** implements both `BinaryLinear` and `TernaryLinear`.

---

## Failure → architecture checklist

| Risk | Gate in our solution |
|------|----------------------|
| Accuracy | Bi-Real residuals, FP stem/head, scales |
| Gradients | STE + clip |
| BN | Always present |
| Fake speed | Packed XNOR benchmark required |
| GPU myth | Document CPU vs GPU; no false 32× claim |
| Amdahl | Report e2e + kernel |
| Export | Pack utilities + size asserts |
| LLM path | Document BitNet, don’t pretend MNIST BNN solves LLMs |
