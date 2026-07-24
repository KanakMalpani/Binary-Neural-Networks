# Decision tree — one pager

```
GPU server quality?     → FP8 / AWQ-INT4 + vLLM (NOT classic BNN)
CPU local LLM?          → BitNet? bitnet.cpp : GGUF Q4_K_M
Edge vision retrain?    → Bi-Real + LCE/FINN ; else INT8
Phone NPU stock SDK?    → INT8/INT4 (docs/20) ; no native 1-bit
Research XNOR kernels?  → this repo (bnn)
Diffusion fidelity?     → INT8/FP8 PTQ ; avoid full BNN
```

Anti-patterns: fake PyTorch binary speedups; binary stem/head; advertise 64× arithmetic as e2e.

Full tree: `docs/18_DECISION_TREE_AND_COMPLETE_ROADMAP.md`  
Recommend: `bnn recommend --goal <gpu-server|cpu-llm|edge-vision|npu-phone|research-xnor|diffusion>`
