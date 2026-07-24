# State of the Art Survey (2016 → 2026)

## Classic CNN BNNs (vision)

| Method | Year | Weights | Activations | Key idea | Practical note |
|--------|------|---------|-------------|----------|----------------|
| **BinaryConnect** | 2015 | ±1 | FP | Binary weights only | Memory win, limited compute win |
| **BinaryNet / BNN** | 2016 | ±1 | ±1 | STE + binary GEMM kernel | Foundational; GPU kernel demo ~7× vs unoptimized |
| **XNOR-Net** | 2016 | ±1 | ±1 | Channel-wise scaling α | Better ImageNet than plain BNN |
| **DoReFa-Net** | 2016 | low-bit | low-bit | Multi-bit QAT, bit-conv | Flexible bitwidths |
| **ABC-Net** | 2017 | multi-binary bases | multi-binary | Approximate full-precision with K binary bases | Accuracy↑, speedup↓ |
| **Bi-Real Net** | 2018 | ±1 | ±1 | **FP residual shortcuts** every block | Huge accuracy recovery; still standard recipe |
| **ReActNet** | 2020 | ±1 | ±1 | RSign / RPReLU distribution reshape + distillation | SOTA-class 1-bit CNNs on ImageNet (~within 3% of FP MobileNet) |
| **SURGE** | 2026 (ICML) | ±1 | ±1 | Learnable surrogate gradients (DPGC) | Addresses STE mismatch |

### Vision accuracy reality (ImageNet top-1, approximate)

| Model family | FP baseline | Best binary-ish | Gap |
|--------------|-------------|-----------------|-----|
| ResNet-18 class | ~69–70% | Bi-Real ~56%, later methods ~60%+ | Still material |
| Compact MobileNet-scale + ReActNet tricks | ~72% | ReActNet ~71% claimed in paper | Near-parity *with* architecture changes |

**Lesson:** Naive “sign everything” collapses accuracy. Bi-Real shortcuts + activation reshape + longer training close most of the gap for CNNs.

## LLM / Transformer era (2023–2026) — where practice moved

| Method | Year | What is quantized | Status in practice |
|--------|------|-------------------|--------------------|
| **BitNet** | 2023 | 1-bit weights, higher-bit activations | Proof that Transformers can train with BitLinear |
| **BitNet b1.58** | 2024 | Ternary \(\{-1,0,1\}\) weights, ~8-bit acts | Matches FP16 LLaMA-class from ~3B+ at same tokens; Microsoft **bitnet.cpp** |
| **BitNet a4.8** | 2024 | 4-bit activations for 1-bit LLMs | Further activation compression |
| **bitnet.cpp** | 2024–2026 | Inference kernels CPU (+ GPU path) | ARM **1.37–5.07×**, x86 **2.37–6.17×** vs FP; energy −55–82% |
| **BitNet-b1.58-2B-4T** | 2025 | Official 2B HF model | Production-usable artifact |
| **Litespark Inference** | 2026 | SIMD ternary kernels for CPUs | Claims up to **~18×** Apple Silicon, **~96×** some x86 vs naive PyTorch |
| **Sparse-BitNet** | 2026 | 1.58-bit + N:M sparsity | Extra ~1.3× with sparse tensor cores |
| **BitEmbed** | 2026 | Ternary embedding encoders | Extends BitNet to retrieval |

### BitNet b1.58 headline numbers (paper)

At 3B params (100B tokens pretrain):

- Memory: **3.55×** less than FP16 LLaMA twin
- Latency: **2.71×** faster
- PPL / zero-shot: **matches** FP16 twin
- At 70B: throughput **8.9×** higher (batch capacity from memory)

Ternary (not pure ±1) matters: the **0** enables feature filtering and closes the quality gap.

## Quantization-aware training & “almost binary”

| Approach | Bits | When to use |
|----------|------|-------------|
| **PTQ INT8 / INT4** (torchao, bitsandbytes, AWQ, GPTQ) | 4–8 | Default production path for LLMs on GPU today |
| **QAT** (Brevitas, torchao QAT) | 4–8 | Recover accuracy when PTQ fails at ≤4-bit |
| **Binary / ternary from-scratch** | 1 / 1.58 | Max efficiency on CPU/edge; needs special kernels |
| **LUT methods** (T-MAC, ternary LUT ASICs) | 1–2 | Replace mul with table / conditional add |

**2026 industry truth:** For NVIDIA datacenter GPUs, **INT4/FP8 via torchao / vLLM / TensorRT** usually beats “research BNN simulation.” For **CPU / mobile / NPU / custom silicon**, **1-bit / 1.58-bit** is the frontier for extreme latency/energy.

## Tooling landscape

| Tool | Stack | Role | Caveat |
|------|-------|------|--------|
| **Larq + Larq Compute Engine** | TF/Keras | Train BNNs + deploy packed ARM kernels | Larq repo archived 2026; still usable |
| **Brevitas** | PyTorch | Flexible QAT including 1-bit | Need export path for real speed |
| **torchao** | PyTorch | FP8/INT4/INT8, QAT, sparsity | Not a full 1-bit BNN stack; best for 4–8 bit GPU |
| **bitnet.cpp** | C++/CUDA | Ternary LLM inference | Best open path for BitNet-style LLMs |
| **This repo** | Pure PyTorch + NumPy packed kernels | Teach + measure real CPU XNOR speedups | Educational / research scaffold |

## What “works” in 2024–2026 practice (decision tree)

```
Need faster NN?
├─ Datacenter GPU serving LLM?
│   └─ Prefer FP8 / INT4 (torchao, vLLM, TensorRT). Binary rarely wins on CUDA TC.
├─ CPU / edge / mobile / NPU?
│   ├─ LLM → BitNet b1.58 + bitnet.cpp (or Litespark-class ternary kernels)
│   └─ CNN → Bi-Real / ReActNet recipe + Larq CE or custom packed XNOR
└─ New silicon / max energy efficiency?
    └─ Design for binary/ternary datapath from day one (LUT / CIM / BGEMM)
```

## Citations (core)

- Courbariaux et al., *Binarized Neural Networks*, 2016
- Rastegari et al., *XNOR-Net*, ECCV 2016
- Liu et al., *Bi-Real Net*, ECCV 2018 / IJCV
- Liu et al., *ReActNet*, ECCV 2020
- Wang et al., *BitNet*, 2023; Ma et al., *BitNet b1.58*, 2024 (arXiv:2402.17764)
- Microsoft bitnet.cpp technical reports 2024–2025 (arXiv:2410.16144, 2502.11880)
- Bannink et al., *Larq Compute Engine*, MLSys
- Litespark Inference, arXiv:2605.06485 (2026)
