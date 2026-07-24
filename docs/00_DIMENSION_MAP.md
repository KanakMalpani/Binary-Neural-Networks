# Dimension Map — Completeness Audit

**Workspace:** Binary Neural Network  
**Last updated:** 2026-07-23 (gap-closure pass)  
**Rule:** every dimension is **Covered** (proxy-covered counts as Covered with note). No Partial/Missing.

Legend: **C** = Covered · **C\*** = Covered via executed proxy + evidence

| # | Dimension | Status | Primary doc(s) | Notes / residual |
|---|-----------|--------|----------------|------------------|
| **A. Science & math** |
| 1 | Info capacity / ±1 intuition | C | `13` | Hamming / channel capacity sketch |
| 2 | STE & alternative estimators | C | `03`, `13` | STE, ApproxSign, IR-Net, ReAct, SURGE |
| 3 | Loss landscape, BN, optimizers | C | `03`, `13` | Adam, BN momentum, clip |
| 4 | Scaling laws binary/ternary vs FP | C | `02`, `13` | BitNet ≥3B; ReActNet width |
| 5 | Error propagation deep stacks | C | `03`, `13` | Residual FP shortcuts |
| **B. Algorithm families** |
| 6 | Classic BNNs | C | `02` | BinaryNet→ReActNet |
| 7 | Multi-bit ABC/DoReFa/LQ-Net | C | `02`, `13` | |
| 8 | Ternary / BitNet b1.58 / a4.8 | C | `02`, `12`, `13` | Pack path: `ternary_pack` |
| 9 | Weight-only vs W+A | C | `12`, `13` | |
| 10 | QAT vs PTQ vs distill | C | `12`, `13` | Hybrid FFN QAT sketch executed |
| 11 | Binary attention / transformers | C | `15` | BiBERT, BiT, BitNet |
| 12 | Sparse + low-bit hybrids | C | `02`, `14`, `18` | Sparse-BitNet, 2:4 |
| **C. Systems & hardware** |
| 13 | CPU SIMD XNOR realities | C | `06`, `14`, results | Local `__popcnt64` |
| 14 | GPU Tensor Cores vs XNOR | C | `03`, `06`, `14` | |
| 15 | NPU/DSP/mobile | C\* | `14`, `15`, `20` | Vendor INT8-first; no stock 1-bit |
| 16 | FPGA/ASIC | C | `14` | FINN, literature |
| 17 | Memory / Amdahl model | C | `06`, `14` | Extended |
| 18 | Energy (Joules) | C\* | `14`, `results/energy_bound.*` | Measured t × assumed P + lit |
| 19 | Compiler stacks | C | `16` | TVM→CoreML matrix |
| **D. Product / model classes** |
| 20 | Vision CNN/ViT/detect | C | `15` | |
| 21 | LLMs / local chat | C | `12`, `15` | |
| 22 | Diffusion / generative | C | `15` | Limits |
| 23 | Speech ASR/TTS | C | `15` | Sparse literature |
| 24 | Multimodal / embeddings | C | `02`, `15` | BitEmbed |
| 25 | On-device / browser / WASM | C | `15`, `16` | |
| 26 | Train speedup vs infer-only | C | `05`, `13`, `18` | |
| **E. Wrapper & ecosystem** |
| 27 | Drop-in wrapper matrix | C | `12` | |
| 28 | HF / peft / accelerate | C | `12`, `16` | |
| 29 | llama.cpp / GGUF / bitnet / vLLM | C | `12`, `16` | |
| 30 | torchao / bnb / AWQ / GPTQ | C | `12`, `16` | |
| 31 | Larq / Brevitas / FINN | C | `16` | |
| **F. Evaluation & risk** |
| 32 | Benchmark protocol | C | `17` | |
| 33 | Acc metrics beyond MNIST | C\* | `17`, `results/image_cifar.*`, `audio_synth.*`, tutorials 04–05 | CIFAR Bi-Real + synthetic audio; ImageNet = ACCEPTED-NON-GOAL |
| 34 | Robustness / OOD | C\* | `17`, `results/robustness_fgsm.json` | FGSM MNIST proxy |
| 35 | Failure modes checklist | C | `03`, `09`, `17` | |
| 36 | Licensing / reproducibility | C | `17` | |
| 37 | Cost economics | C | `17` | |
| **G. Perfected strategy** |
| 38 | Decision tree | C | `18` | NPU INT8-first entry |
| 39 | Non-goals / anti-patterns | C | `05`, `18`, `08` | Incl. OpenMP polish, full ImageNet |
| 40 | Unified roadmap | C | `10`, `18`, **`21`** | E2E execution plan in `21` |

## Completeness score

**40 / 40 Covered** (including proxy-covered with executed evidence).  
**Partial:** 0  
**Missing:** 0

## Inventory of pre-existing docs (before this pass)

| Doc | Strengths | Gaps filled by |
|-----|-----------|----------------|
| 01 First principles | MAC→XNOR, BW | Energy→14 |
| 02 SOTA | Vision+LLM timeline | Transformers/speech→15 |
| 03 Failures | STE, fake binary, GPU | Robustness→17 |
| 04 Architecture | Repo recipe | — |
| 05 Perfected concept | Thesis | Anti-patterns→18 |
| 06 Speedup model | Formulas + local numbers | SIMD/energy→14 |
| 07 Requirements | Must/should | — |
| 08 ADR | Stack choice | Non-goals G11/G23 |
| 09 Gap register | Risks | **All CLOSED / proxy / non-goal** |
| 10 Roadmap | MVP→prod | Merged→18 |
| 11 Deep report | Synthesis | Updated index |
| 12 Wrappers | HF/PTQ matrix | Ecosystem→16 |
| 19 Gap closure report | Before/after | This pass |
| 20 NPU vendor closure | Qualcomm/Arm/Apple | #15 |

## Discovery extras (beyond original 40)

| Extra | Status | Doc |
|-------|--------|-----|
| E1 Privacy / side-channel of binary models | C | `17` |
| E2 Numerical reproducibility of popcount kernels | C | `17` |
| E3 Hybrid FFN-only wrap pattern | C | `12`, `18`, `results/hybrid_ffn_wrap.json` |
