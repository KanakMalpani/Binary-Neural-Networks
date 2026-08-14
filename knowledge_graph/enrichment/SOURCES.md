# Enrichment sources (gap-parallel research)

Generated: 2026-08-04 · Researcher: `gap-parallel`

Primary papers and tools used to fill under-cited gaps relative to `docs/` (esp. `02_SOTA_SURVEY`, `03_FAILURE_ANALYSIS`, `13_TRAINING_QAT_DISTILL`, `14_HARDWARE_AND_ENERGY`, `16_ECOSYSTEM_AND_TOOLING`, `20_NPU_VENDOR_CLOSURE`, `23_BITNET_CPP_BRIDGE`).

## Vision BNNs (ImageNet ladder)

| Topic | Citation | ID / URL | Key number(s) |
|-------|----------|----------|---------------|
| Bi-Real Net | Liu et al., ECCV 2018 / IJCV | [arXiv:1808.00278](https://arxiv.org/abs/1808.00278), [1811.01335](https://arxiv.org/abs/1811.01335) | ResNet-18 **56.4%** / 79.5%; ResNet-34 **62.2%** / 83.9% |
| Real-to-Binary | Martinez et al., ICLR 2020 | [arXiv:2003.11535](https://arxiv.org/abs/2003.11535), [OpenReview](https://openreview.net/forum?id=BJg4NgBKvH) | ResNet-18 **65.4%** / 86.2%; CIFAR-100 **76.2%** |
| ReActNet | Liu et al., ECCV 2020 | [arXiv:2003.03488](https://arxiv.org/abs/2003.03488), [code](https://github.com/liuzechun/ReActNet) | ReActNet-A **69.4%** @ 0.87e8 OPs; -B **70.1%**; -C **71.4%**; real twin 72.4% |

## BitNet family & runtimes

| Topic | Citation | ID / URL | Key number(s) |
|-------|----------|----------|---------------|
| BitNet | Wang et al., 2023 | [arXiv:2310.11453](https://arxiv.org/abs/2310.11453) | BitLinear 1-bit weights |
| BitNet b1.58 | Ma et al., 2024 | [arXiv:2402.17764](https://arxiv.org/abs/2402.17764) | 3B: **3.55×** mem, **2.71×** latency vs FP16 twin; 70B throughput **8.9×**; ~**71.4×** arithmetic energy (7nm model) |
| BitNet a4.8 | Wang/Ma/Wei, 2024 | [arXiv:2411.04965](https://arxiv.org/abs/2411.04965) | W1.58A4 hybrid; ~**55%** active params; 3-bit KV OK |
| BitNet v2 | Wang/Ma/Wei, 2025 | [arXiv:2504.18415](https://arxiv.org/abs/2504.18415) | Native 4-bit acts + Hadamard |
| bitnet.cpp | Wang et al., 2024 | [arXiv:2410.16144](https://arxiv.org/abs/2410.16144) | ARM **1.37–5.07×**, x86 **2.37–6.17×**; energy −**55.4–70%** (M2), −**71.9–82.2%** (i7) |
| bitnet.cpp ACL | 2025 | [arXiv:2502.11880](https://arxiv.org/abs/2502.11880) | Edge ternary kernel follow-up |
| BitDistill | Wu et al., 2025 | [arXiv:2510.13998](https://arxiv.org/abs/2510.13998) | Task FP→1.58; ~**10×** mem, **~2.65×** CPU; SubLN+CPT+MiniLM attn distill |
| BitDistiller (distinct) | Du et al., 2024 | [arXiv:2402.10631](https://arxiv.org/abs/2402.10631) | Sub-4-bit self-distill — **not** BitDistill |
| HF checkpoint | Microsoft | [bitnet-b1.58-2B-4T](https://hf.co/microsoft/bitnet-b1.58-2B-4T) | Official 2.4B / 4T tokens |

## Binary transformers (non-LLM)

| Topic | Citation | ID / URL | Key number(s) |
|-------|----------|----------|---------------|
| BiBERT | Qin et al., 2022 | [arXiv:2203.06390](https://arxiv.org/abs/2203.06390) | Full W1A1E1; **56.3×** FLOPs / **31.2×** size; +20.4 pp vs BinaryBERT W1A1 avg GLUE |

## 2026 literature overlay (cite only — not lab goldens)

Added 2026-08-15 (`knowledge_graph/enrichment/literature_2026.json`). Paper figures stay **their** numbers.

| Topic | Citation | ID / URL | Notes |
|-------|----------|----------|-------|
| ScaleQ-1.58 / AYOT | Wang et al., 2026 | [arXiv:2608.01078](https://arxiv.org/abs/2608.01078) | Ternary PTQ + reasoning-trace calibration. **Not reproduced here.** |
| BitEmbed | Li et al., 2026 | [arXiv:2606.25674](https://arxiv.org/abs/2606.25674) | BitNet-style text embedders. Named in `docs/15`. **Not a lab golden.** |
| VibeVoice-ASR-BitNet | Xu et al., 2026 | [arXiv:2607.21075](https://arxiv.org/abs/2607.21075) | Their ASR stack. Lab audio is synthetic; ASR is a non-goal. |
| Litespark Inference | Dade et al., 2026 | [arXiv:2605.06485](https://arxiv.org/abs/2605.06485) | SIMD ternary vs naive PyTorch. Local numbers: `gap_litespark_local`. |

## Sparse + low-bit hybrids

| Topic | Citation | ID / URL | Key number(s) |
|-------|----------|----------|---------------|
| Sparse-BitNet | Zhang et al., 2026 | [arXiv:2603.05168](https://arxiv.org/abs/2603.05168) | BitNet more N:M-friendly; up to **~1.30×** with 6:8 sparse TC |
| Q-Sparse | Wang et al., 2024 | [arXiv:2407.10969](https://arxiv.org/abs/2407.10969) | Fully sparsely-activated LLMs |

## FPGA / tooling

| Topic | Citation | ID / URL | Key number(s) |
|-------|----------|----------|---------------|
| FINN | Umuroglu et al., FPGA 2017 | [arXiv:1612.07119](https://arxiv.org/abs/1612.07119), [docs](https://xilinx.github.io/finn/) | MNIST **12.3M FPS** (~583k FPS/W wall); CIFAR **21.9k FPS** (~1.87k FPS/W wall) on ZC706 |
| Brevitas | AMD/Xilinx | [github.com/Xilinx/brevitas](https://github.com/Xilinx/brevitas) | PyTorch QAT → FINN |
| Larq CE | Plumerai | [github.com/larq/compute-engine](https://github.com/larq/compute-engine) | ARM BGEMM; Larq upstream archived ~2026 |

## PTQ stacks (contrast)

| Topic | Citation | ID / URL | Notes |
|-------|----------|----------|-------|
| AWQ | Lin et al., 2023 | [arXiv:2306.00978](https://arxiv.org/abs/2306.00978) | Activation-aware INT4; salient ~1% channels |
| GPTQ | Frantar et al., 2022 | [arXiv:2210.17323](https://arxiv.org/abs/2210.17323) | Second-order weight PTQ; calib-domain sensitivity |
| GPU XNOR reality | — | [arXiv:1911.04477](https://arxiv.org/abs/1911.04477) | Custom XNOR can lose to cuDNN FP |

## Lab-local (already covered; used for gap mapping)

- `docs/02_SOTA_SURVEY.md`, `03_FAILURE_ANALYSIS.md`, `09_GAP_REGISTER.md`
- `docs/13_TRAINING_QAT_DISTILL.md`, `14_HARDWARE_AND_ENERGY.md`, `16_ECOSYSTEM_AND_TOOLING.md`
- `docs/20_NPU_VENDOR_CLOSURE.md`, `23_BITNET_CPP_BRIDGE.md`

## Research tooling used

- Academia MCP: `arxiv_search`, `arxiv_download`
- Exa: `web_search_exa`, `web_fetch_exa`
- Tavily: `tavily_search`
- Hugging Face Hub: `hub_repo_search`
