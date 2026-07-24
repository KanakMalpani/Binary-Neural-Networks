# Chosen Solution Architecture

## Verdict in one paragraph

**For this workspace (CPU PyTorch, reproducible research):** train a **Bi-Real-style
binary CNN/MLP** with STE, keep first/last layers full-precision, then run inference with
**bit-packed XNOR + popcount** kernels. That is the only path that simultaneously
(1) stays accurate enough, (2) demonstrates *real* wall-clock gains on CPU, and
(3) teaches the failure modes that kill “fake binary” PyTorch demos.

**For production LLMs in 2026:** prefer **BitNet b1.58 + bitnet.cpp** (ternary weights)
on CPU/edge, and **FP8/INT4 (torchao / vLLM)** on datacenter GPUs — not classic BNNs.

---

## What we binary-ize

| Component | Precision | Why |
|-----------|-----------|-----|
| Stem / first layer | FP32 | Pixels/embeddings need magnitude |
| Hidden dense / conv layers | **Binary weights + binary activations** (±1) | Max packing / XNOR benefit |
| Residual shortcuts | FP32 | Bi-Real capacity recovery |
| BatchNorm + scale | FP32 | Stabilize & restore range |
| Classifier / last layer | FP32 | Softmax/logits need resolution |

Optional module: **TernaryLinear** (BitNet-style absmean) for LLM-oriented experiments.

## What we do *not* claim

- 32× end-to-end on CUDA without custom kernels
- Training speedup (training stays FP latent + STE)
- ImageNet SOTA from this small MNIST demo

## Training recipe

1. **Latent weights** \(W \in \mathbb{R}\), forward uses \(\mathrm{sign}(W)\) (or ternary).
2. **STE** for activations and weights; clip latent weights to \([-1,1]\).
3. **Adam**, lr ≈ 1e-3 (MLP) / 1e-3 (CNN), BN momentum 0.9.
4. **MNIST** for fast closed-loop validation (seconds–minutes on CPU).
5. Export packed weights; verify size ≈ 1/32 of FP for binary tensors.

## Inference path

```
Train (sim):   FP activations → sign → FP GEMM with ±1  (correct grads, no speed)
Infer (packed): bitpack(x), bitpack(W) → XNOR → popcount → scale → BN/add
```

## Why not Larq/Brevitas as hard dependency?

- Larq is TF/Keras and archived; Brevitas is heavy for a from-scratch teaching repo.
- We need **transparent** packing kernels to prove speedup with evidence in `results/`.
- Dependencies: `torch`, `numpy` (see `pyproject.toml`); torchvision optional.
- Kernels: MSVC x64 DLL on Windows; NumPy fallback elsewhere.
- Package entry: `pip install -e .` → `import bnn` / `bnn` CLI.

## Repo map

```
docs/           research synthesis (this + siblings)
bnn/            STE, layers, models, packed kernels
scripts/        train.py, benchmark.py, export_check.py
results/        measured JSON + markdown tables
checkpoints/    trained weights
```

## Success criteria (definition of done)

1. FP32 vs Binary accuracy on MNIST within a small gap (target: binary ≥ 95% if FP ≥ 97%).
2. Packed binary GEMM **beats** FP32 NumPy/PyTorch matmul on CPU for wide layers (e.g. 4096×4096).
3. Documented theoretical op counts vs measured wall-clock (no conflation).
4. Failure modes and mitigations written and enforced in code where possible.
