# Decision Tree, Anti-Patterns, Unified Roadmap

## 38. Decision tree — technique × user goal

```
START: What is the primary goal?
│
├─ Maximize quality on NVIDIA GPU server
│   └─ BF16/FP8 train → FP8 or AWQ-INT4 serve (vLLM/SGLang/TensorRT)
│
├─ Fit / speed local CPU LLM (chat)
│   ├─ Checkpoint is BitNet? → bitnet.cpp
│   └─ Else → GGUF Q4_K_M (llama.cpp) or torchao CPU INT8/4
│
├─ Edge vision camera (power/latency)
│   ├─ Can retrain? → Bi-Real/ReActNet → LCE (ARM) or FINN (FPGA)
│   └─ Cannot retrain? → INT8 TFLite/OpenVINO/ORT
│
├─ Phone / NPU deploy
│   ├─ Stock SDK (QNN / CoreML / Ethos+Vela) → **INT8 (or INT4 wt)** — not native 1-bit
│   ├─ Need BitNet ternary on Hexagon → budget **custom** kernels
│   └─ Need classic W+A binary CNN → CPU LCE / this repo / FPGA FINN
│
├─ Convert existing HF LLM to extreme 1.58-bit
│   └─ BitDistill / gradual-λ QAT — NOT absmean PTQ alone
│
├─ Research / teach XNOR kernels
│   └─ This repo: train BNN + packed CPU GEMM + wrapper demo
│
└─ Diffusion / high-fidelity generative
    └─ INT8/FP8 weight PTQ; avoid full BNN
```


## 39. Non-goals & anti-patterns

| Anti-pattern | Why bad |
|--------------|---------|
| “sign() in PyTorch = 32× faster” | Fake binary; often slower |
| Binary everything including stem/head | Acc collapse, tiny compute share |
| Expect CUDA BNN > Tensor Core FP8 | Usually false |
| PTQ Llama → ternary, ship chat | Quality wipe without distill |
| Advertise \(R_{\mathrm{arith}}=64×\) as e2e | Amdahl |
| Re-pack weights every forward in benches | False slowdowns |
| Keep float dequant buffers “for speed” while claiming 32× size | Lies about footprint |
| Claim training throughput win from BNN | Training uses FP latents |

**Non-goals of this repo:** replace vLLM; ImageNet SOTA; full BitNet pretrain;
OpenMP/AVX polish; board RAPL meters (proxy-closed).

## 40. Unified roadmap (priority × impact × feasibility)

**Execution plan (phases, task IDs, Done Definition):**  
→ **[`docs/21_E2E_ROADMAP_COMPLETE_REPO.md`](21_E2E_ROADMAP_COMPLETE_REPO.md)** (canonical).  
`docs/10_ROADMAP.md` is a short pointer only.

Score 1–5; **PIF = priority** (strategy view — detail lives in `21`).

| Item | Impact | Feas. | Effort | PIF | Status |
|------|--------|-------|--------|-----|--------|
| Docs completeness (gap-closure) | 5 | 5 | 3 | High | **Done** → `19` |
| Packed CPU GEMM + benches | 5 | 5 | 3 | High | **Done** |
| Linear wrap demo | 4 | 5 | 2 | High | **Done** |
| Hybrid FFN wrap + STE sketch | 4 | 5 | 2 | High | **Done** |
| CIFAR-10 Bi-Real proxy | 4 | 5 | 3 | High | **Done** (`cifar10_proxy`) |
| Energy bound to measured latency | 3 | 5 | 1 | High | **Done** (`energy_bound`) |
| Ternary 2-bit pack path | 3 | 5 | 2 | Med | **Done** |
| NPU vendor 1-bit closure | 4 | 5 | 2 | High | **Done** → `20` |
| Package / tests / CI / CLI (P0) | 5 | 5 | 3 | High | **Next** → `21` P0 |
| Eval suite + golden floors (P5) | 5 | 5 | 2 | High | Planned → `21` |
| HF / GGUF / bitnet / GPU INT4 guides (P4) | 5 | 4 | 2 | High | Planned → `21` |
| OpenMP/AVX2 kernel | 4 | 4 | 3 | — | **ACCEPTED-NON-GOAL** (optional P7) |
| torchao INT4 demo (if CUDA env) | 5 | 2* | 2 | Med | *CPU-only; doc lane in P4 |
| FINN/Brevitas path | 3 | 2 | 5 | Low | Doc only (P7) |
| ImageNet Bi-Real | 4 | 1 | 5 | — | **ACCEPTED-NON-GOAL** |
| BitDistill reproduction | 5 | 1 | 5 | Low | Cite + recipe (P2 sketch) |

\* Feasibility low **on this workstation** (no CUDA); high elsewhere.

## Perfected strategy (unchanged thesis)

Extreme low-bit **inference** on **bandwidth-bound** hardware with **real kernels** and
architecture-aware training — not universal 32× binary marketing.

Update ADR only if thesis changes; this roadmap **extends** ADR without contradiction.
