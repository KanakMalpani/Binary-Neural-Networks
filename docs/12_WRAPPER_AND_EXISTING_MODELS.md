# Wrapping Existing Models: Deep Answer

*Generated: 2026-07-23 | Confidence: High (primary sources + local wrap demo)*

## Executive summary

**Yes, you can wrap existing models — but “transparent + faster + accurate” is only
true for some wrappers.**

| Goal | Best wrapper today | Drop-in? | Retrain? |
|------|-------------------|----------|----------|
| Faster **GPU** LLM / diffusion | torchao / AWQ / GPTQ / bitsandbytes INT4–FP8 | Mostly yes | No (PTQ) or light calib |
| Smaller **CPU** LLM | GGUF/llama.cpp Q4_K / bitnet.cpp (native BitNet) | Via export | No / yes for BitNet |
| Extreme 1-bit CNN | Larq / QAT Bi-Real — **not** PTQ wrap | No | **Yes** |
| This repo’s XNOR speed | `binary_xnor` wrap of wide Linears | API yes | Accuracy needs QAT |

**Verdict:** A **transparent speed wrapper** for popular PyTorch/HF models is
**partial**: INT4/FP8 PTQ wrappers work in production; **binary/ternary XNOR wrappers
are not honest drop-ins** for pretrained FP checkpoints without QAT/distillation.
Weight-only ternary conversion of arbitrary HF LLMs **without** continued training
**collapses quality** (HF BitNet docs; BitDistill 2025).

---

## 1. Can we implement a wrapper over current models to make them faster?

### First principles

A wrapper that only changes storage format without changing the **compute kernel**
does not speed up matmul. Speed needs either:

1. **Less DRAM traffic** + a kernel that streams packed weights (batch-1 decode), or  
2. **Faster arithmetic** on hardware that likes that format (Tensor Core INT4/FP8, or CPU popcount).

\[
S_{\mathrm{e2e}} = \frac{1}{(1-f)+f/S_k}
\]

For Transformers, MLP/FFN Linears often dominate \(f\) (~50–70% of matmul bytes);
attention + softmax + norms stay higher precision → Amdahl caps e2e.

### Wrapper taxonomy (verdict matrix)

| Approach | Drop-in? | Retrain? | Hardware | Typical speedup | Accuracy risk | Fits “wrapper”? |
|----------|----------|----------|----------|-----------------|---------------|-----------------|
| **bitsandbytes INT8/NF4** | Yes (`load_in_4bit`) | No | NVIDIA GPU | Memory ~2–4×; tokens/s often **flat or −20–40%** vs AWQ | Low–med | **Yes** (memory) |
| **AWQ / GPTQ INT4** | Yes (prequant or calib) | Calib only | GPU (+vLLM) | ~**1.3–1.9×** latency, ~**50%** VRAM (e.g. Qwen3-8B AWQ) | Low on ≥7B | **Yes** (prod) |
| **torchao INT4/FP8** | Yes (`quantize_`) | Optional QAT | GPU/CPU/XPU | Llama-3-8B INT4 ~**1.89×**, ~58% less mem; FP8 ~1.5–1.7× H100 | Low with AWQ/QAT | **Yes** |
| **GGUF Q4_K + llama.cpp** | Via convert | No | CPU/GPU | Often **best CPU** path for normal LLMs | Low–med | **Yes** (export) |
| **bitnet.cpp** | For BitNet checkpoints | Native / distill | CPU (+GPU path) | Lit. **1.4–6×** CPU | N/A if native BitNet | **Yes** if model is BitNet |
| **HF BitLinear replace + PTQ absmean** | API replace | **Needs QAT/FT** | CPU/GPU | Speed only with kernels | **High** if PTQ-only | **Partial** |
| **Full W+A BNN wrap** | API possible | **Requires QAT** | CPU/NPU/ASIC | 2–18× with kernels | **Very high** without QAT | **No** as transparent |
| **This repo `binary_xnor`** | Yes for `nn.Linear` | Recommended | CPU | Wide layers: **several×** (local) | High on FP pretrained | **Demo / research** |
| **This repo `ternary_weight_only`** | Yes | Recommended | Any (FP GEMM) | **Size** ~8–16× theoretical; speed **not** without ternary kernel | High PTQ-only | **Size prototype** |

Sources: HF Transformers quantization guide; torchao docs / HF Hub recipes; BitNet b1.58
(arXiv:2402.17764); HF “Fine-tuning LLMs to 1.58bit”; BitDistill (arXiv:2510.13998);
prior repo measurements (`results/benchmark.md`).

### What cannot be a transparent wrapper

1. **Activation+weight binary nets** — need STE/QAT; PTQ sign() destroys features.  
2. **GPU XNOR beating Tensor Cores** — rare on commodity NVIDIA.  
3. **Naïve BitNet conversion of Llama/Qwen** — HF: BitNet “can’t be quantized on the fly”;
   abrupt ternary wipeout of pretrained info unless gradual λ / distill / continued pretrain.

### Calculated benefit example (local LLM, batch-1)

7B FP16 weights ≈ \(7\times10^9 \times 2\) B ≈ **14 GB**.  
INT4 weight-only ≈ **3.5 GB** (~4×).  
If decode is 80% weight-bandwidth bound and INT4 kernel achieves \(S_k=3\):

\[
S_{\mathrm{e2e}} \approx 1/(0.2+0.8/3) \approx 2.14\times
\]

Matches order of published ~1.3–1.9× server INT4 numbers (kernel + non-matmul overhead).

Binary 1-bit weights ≈ **0.875 GB** (~16× vs FP16) — **but** only if activations pack too
or a ternary/add kernel exists; otherwise you dequant and lose the win.

---

## 2. What more can we do? (prioritized)

| Priority | Extension | Why | Effort |
|----------|-----------|-----|--------|
| P0 | **Use industry PTQ wrappers** (torchao/AWQ/bnb) for real HF models on GPU | Immediate user value | Low (deps) |
| P0 | **OpenMP/AVX** on `binary_gemm_u64` | Push local \(S\) toward BLAS-competitive | Med |
| P1 | **Hybrid wrap**: quantize FFN Linears only; keep attn/embed/lm_head FP | Better accuracy/Amdahl tradeoff | Med |
| P1 | **QAT / distillation recipe** after ternary/binary wrap (BitDistill-style) | Makes wrap usable | High |
| P1 | **End-to-end packed BinaryMLP** forward (done partially via wrapper) | Close “sim vs packed” gap | Low–med |
| P2 | **HF `from_pretrained` → export GGUF / bitnet.cpp** guide + thin CLI | Real CPU LLM path | Med |
| P2 | **Ternary LUT/add CPU kernel** (T-MAC / Litespark class) | Speed for weight-only ternary | High |
| P3 | Speculative decoding + INT4; 2:4 sparsity + FP8 | Orthogonal wins | Med |
| P3 | Edge: ORT / OpenVINO / ExecuTorch INT8 | Vision/mobile | Med |

---

## 3. How this benefits normally used models

### Local LLM inference (CPU)

- **Normal models (Llama, Qwen, Phi):** wrap via **GGUF Q4/Q5** or torchao CPU INT8/INT4 — not classic BNN.  
- **BitNet-native models:** **bitnet.cpp** — largest extreme-low-bit win (memory + tokens/s).  
- **Benefit:** fit larger ctx / more concurrency in RAM; lower $/token on CPU boxes.

### GPU serving

- Prefer **FP8 / INT4 AWQ+vLLM/SGLang** (1.3–1.9×, ~½ VRAM).  
- Binary XNOR wrappers: **not recommended**.

### Vision classifiers on CPU/edge

- INT8 PTQ (OpenVINO, ORT, TFLite) = standard wrapper.  
- Full BNN: train Bi-Real/ReActNet (Larq CE) — **retrain**, then deploy packed.  
- Benefit: mJ/frame and DRAM for cameras/NPUs.

### Diffusion / multimodal

- Same as GPU LLMs: **weight INT8/FP8**, keep attention sensitive layers higher bit;
  UNet/DiT Linear/Conv dominate — PTQ wrappers exist; 1-bit rare.

### Hybrid pattern (recommended product shape)

```
Keep: embeddings, lm_head/classifier, norms, softmax/attn scores (FP16/BF16/INT8)
Wrap: MLP/FFN (and optionally out_proj) → INT4 / ternary / binary_xnor
```

This maximizes \(f\) under quantization while protecting quality-critical ops.

---

## 4. Local prototype (this repo)

| Artifact | Role |
|----------|------|
| `bnn/wrapper.py` | `wrap_linear_modules`, `PackedBinaryXNORLinear`, `TernaryWeightOnlyLinear` |
| `scripts/wrap_existing_demo.py` | Before/after size + e2e + layer microbench |
| `results/wrap_demo.json` | Measured numbers |

### Measured (`binary_xnor`, hidden=4096, batch=32, 2 middle Linears)

| Metric | Value |
|--------|------:|
| Weight compression (replaced) | **32×** |
| Model bytes | 147.3 MB → **13.1 MB** |
| E2E latency | 21.1 → 12.6 ms (**1.67×**) |
| Layer gemm_only vs torch Linear | **2.58×** |
| Output cosine vs FP (no QAT) | **0.28** (not drop-in accurate) |

### Measured (`ternary_weight_only`, same shape, batch=64)

| Metric | Value |
|--------|------:|
| Theoretical 2-bit compression (replaced) | **16×** |
| Model bytes | 147.3 → **46.6 MB** |
| Output cosine vs FP | **0.91** (better than binary_xnor) |
| Speed vs FP | **slower** here (dequant + FP GEMM; need bitnet.cpp-class kernels) |

**Limitation:** Production HF LLM wrap → **torchao/AWQ/GGUF/bitnet.cpp**, not this kernel alone.

---

## 5. Decision flowchart

```
Need faster existing model?
├─ NVIDIA/AMD server GPU? → torchao FP8 or AWQ INT4 + vLLM
├─ Local CPU LLM (normal arch)? → GGUF Q4_K_M / llama.cpp
├─ Already BitNet checkpoint? → bitnet.cpp
├─ Edge vision, can retrain? → Bi-Real/ReActNet + Larq CE / this BNN recipe
└─ Research / CPU XNOR demo on Linear MLP? → this repo binary_xnor wrap
```

---

## 6. New gaps

| ID | Gap | Mitigation | Status |
|----|-----|------------|--------|
| G16 | Expect HF→1-bit transparent speed | Docs + cosine 0.28 demo; use INT4/BitNet | Closed |
| G17 | Ternary PTQ without ternary kernel | Size-only mode; bitnet.cpp for speed | Closed |
| G18 | Python pack overhead hides kernel win | gemm_only metric; large-N e2e | **CLOSED** |
| G19 | Float dequant cache erased size win | Store int8 ternary only | Closed |
