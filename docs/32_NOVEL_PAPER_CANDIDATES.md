# Novel paper candidates (local research vault)

**Date:** 2026-07-24  
**Policy:** Selective — only credible novel systems contributions from this lab; not XNOR-Net/BitNet surveys.

Vault root: `C:\00 Research Papers`  
Series index: `C:\00 Research Papers\BINARY_NEURAL_SERIES_README.md`

## Created folders (3)

| Paper | Path | Novel claim (1 sentence) |
|-------|------|--------------------------|
| **B1 — Honest speedup accounting** | `C:\00 Research Papers\Stop Claiming 32x Honest Speedup Accounting for Binary Neural Networks\` | Dual-metric reporting (op-count/compression vs wall-clock) plus fake-binary negative control and Amdahl/energy proxies — never advertise 32×/64× as latency. |
| **B2 — Packed XNOR productization** | `C:\00 Research Papers\Packed XNOR on Commodity CPUs Reproducible Productization and Golden Gates\` | Commodity CPU packed XNOR recipe (MSVC/portable) with fair pre-pack benches and machine-checkable golden gates for third parties/AI agents. |
| **B3 — When not to binarize** | `C:\00 Research Papers\When Not to Binarize Decision Tree for Hybrid Low-Bit Wrapping\` | Goal×hardware decision tree + hybrid FFN wrap evidence for honest wrapping vs INT4/FP8/GGUF / “do not binarize.” |

Each folder contains: `README.md`, `01_idea.md`, `00_source/` (`SOURCES.md` + extracted notes).

## Lab evidence mapped into papers

| Evidence | Primary paper |
|----------|---------------|
| `docs/06`, `results/benchmark.json`, fake-binary trap | B1 |
| `results/energy_bound.json`, \(E=P\cdot t\) proxy | B1 |
| `docs/35`, `bnn/math/` XNOR↔dot proofs + Amdahl calculators | B1 (math spine) |
| `results/math_ste_compare.json` (STE / ApproxSign / EDE) | B1 (learning math) |
| `AGENTS.md`, `docs/30`, `tests/golden_floors.json`, `bnn repro` | B2 |
| Multimodal canaries (MNIST / CIFAR / audio synth) | B2 (canaries, not SOTA) |
| `docs/18` decision tree, `docs/12` wrapper taxonomy | B3 |
| `results/hybrid_ffn_wrap.json`, `wrap_demo.json` | B3 |
| `docs/20` NPU vendor closure | B3 |

## Rejected standalone angles

- Literature-only survey of BNNs  
- Cross-modality “SOTA lab” from proxy CIFAR + synthetic audio  
- Energy methodology alone (folded into B1)  
- Any GPU 32× claim from `sign()` (forbidden by thesis lock)

## Status

Idea vault only — no venue LaTeX yet. Drafting should preserve thesis lock: packed CPU/edge kernels + honest STE simulation; no fake GPU wins.
