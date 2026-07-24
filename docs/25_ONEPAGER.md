# Decision tree — one pager

Honest routing for extreme low-bit / quantized inference. This lab proves
**packed CPU/edge** kernels; it does **not** replace GPU INT4/FP8 stacks.

```
GPU server quality?     → FP8 / AWQ-INT4 + vLLM (NOT classic BNN)
CPU local LLM?          → BitNet? bitnet.cpp : GGUF Q4_K_M
Edge vision retrain?    → Bi-Real + this repo / LCE/FINN ; else INT8
Phone NPU stock SDK?    → INT8/INT4 (docs/20) ; no stock 1-bit
Research XNOR kernels?  → this repo (`bnn`)
Diffusion fidelity?     → INT8/FP8 PTQ ; avoid full BNN
```

| Claim | Reality |
|-------|---------|
| Weight pack **32×** | Exact bit-pack compression |
| ~64× word ops | Theoretical — not wall-clock e2e |
| Kernel speedups | Real on large GEMMs with native DLL |
| STE train | Simulation; not a throughput win |

Anti-patterns: fake PyTorch binary speedups; binary stem/head; advertising
arithmetic reduction as end-to-end latency.

```bat
bnn recommend --goal <gpu-server|cpu-llm|edge-vision|npu-phone|research-xnor|diffusion>
bnn repro
```

Full tree: [`18_DECISION_TREE_AND_COMPLETE_ROADMAP.md`](18_DECISION_TREE_AND_COMPLETE_ROADMAP.md)  
Reproduce: [`../REPRODUCIBILITY.md`](../REPRODUCIBILITY.md)
