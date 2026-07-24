# Perfected Concept: Extreme Low-Bit Inference (not “make everything ±1”)

## Naive claim (rejected)

> “Make neural networks binary (±1) and get **exponentially** (≈32×) faster time.”

**Why this fails as a product thesis:**
1. On commodity NVIDIA GPUs, FP16/FP8/INT8 Tensor Cores usually beat naive binary.
2. PyTorch `sign()` BNNs are *simulations* — still FP GEMM → often **slower**.
3. Pure ±1 accuracy collapses without Bi-Real/ReActNet/BitNet-class recipes.
4. “Exponential” conflates bit-packing density (32× memory) with wall-clock (Amdahl).

## Sharpened claim (kept)

> **Compress the dominant matmul traffic to 1–1.58 bits and execute it with packed
> integer kernels on the hardware where memory bandwidth (not Tensor Cores) is the
> bottleneck — CPU / edge / NPU — to cut inference latency and energy by several×
> while preserving task accuracy via architecture-aware training.**

### Positioning

| Dimension | Choice |
|-----------|--------|
| **Primary job** | Faster / cheaper **inference** |
| **Primary hardware** | **CPU / mobile / NPU** (secondary: custom ASIC) |
| **Primary workloads** | (A) edge vision BNNs, (B) ternary LLMs (BitNet b1.58) |
| **Non-goal** | Faster GPU training; drop-in 32× on CUDA without kernels |
| **Success metrics** | Wall-clock latency, tokens/s or img/s, model bytes, accuracy Δ |

### What is binary vs not

| Keep higher precision | Extreme low-bit |
|-----------------------|-----------------|
| First / last layers, embeddings | Hidden Linear/Conv weights |
| BatchNorm / RMSNorm scales | Hidden activations (CNN BNNs) or INT8 acts (BitNet) |
| Softmax, attention scores (unless separately quantized) | Residual streams stay FP in Bi-Real |
| Training optimizer states | — |

### Thesis vs alternatives

| Approach | Best when | Avoid when |
|----------|-----------|------------|
| **1-bit BNN (weights+acts)** | Edge CNN, ASIC, max energy | Datacenter GPU, quality-critical ImageNet without ReActNet |
| **Ternary BitNet b1.58** | Local LLM on CPU | Need SOTA chat quality at tiny scale without enough tokens |
| **INT8 / FP8 / INT4** | NVIDIA / AMD servers | Extreme edge memory (KB–MB) |
| **Sparsity alone** | Structured 2:4 on sparse TC | Unstructured sparse on CPU |

**Product decision for this repo:** ship a **CPU-native binary GEMM proof** + **trainable Bi-Real/BitLinear recipes**, and document BitNet as the LLM production path — not pretend MNIST BNN replaces GPT serving on H100.
