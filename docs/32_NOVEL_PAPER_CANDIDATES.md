# Novel paper candidates (local research vault)

**Date:** 2026-08-04  
**Policy:** Selective — only credible novel systems contributions from this lab; not XNOR-Net/BitNet surveys.  
**Tasks:** W12.T01 (vault links) · W12.T05 (triage → ship or defer)

## Local vault links (W12.T01)

| Resource | Path |
|----------|------|
| Vault root | `C:\00 Research Papers` |
| Series index | `C:\00 Research Papers\BINARY_NEURAL_SERIES_README.md` |
| B1 folder | `C:\00 Research Papers\Stop Claiming 32x Honest Speedup Accounting for Binary Neural Networks\` |
| B2 folder | `C:\00 Research Papers\Packed XNOR on Commodity CPUs Reproducible Productization and Golden Gates\` |
| B3 folder | `C:\00 Research Papers\When Not to Binarize Decision Tree for Hybrid Low-Bit Wrapping\` |

Each folder contains: `README.md`, `01_idea.md`, `00_source/` (`SOURCES.md` + extracted notes).

Lab publication plan + claims whitelist: [`docs/PUBLICATION_PLAN.md`](PUBLICATION_PLAN.md).  
Figure / claims machine check: `bnn bridge figures`.

## Created folders (3)

| Paper | Path | Novel claim (1 sentence) |
|-------|------|--------------------------|
| **B1 — Honest speedup accounting** | `C:\00 Research Papers\Stop Claiming 32x Honest Speedup Accounting for Binary Neural Networks\` | Dual-metric reporting (op-count/compression vs wall-clock) plus fake-binary negative control and Amdahl/energy proxies — never advertise 32×/64× as latency. |
| **B2 — Packed XNOR productization** | `C:\00 Research Papers\Packed XNOR on Commodity CPUs Reproducible Productization and Golden Gates\` | Commodity CPU packed XNOR recipe (MSVC/portable) with fair pre-pack benches and machine-checkable golden gates for third parties/AI agents. |
| **B3 — When not to binarize** | `C:\00 Research Papers\When Not to Binarize Decision Tree for Hybrid Low-Bit Wrapping\` | Goal×hardware decision tree + hybrid FFN wrap evidence for honest wrapping vs INT4/FP8/GGUF / “do not binarize.” |

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
| `docs/23`–`24`, `bnn bridge …` | B3 (bridge honesty) |

## Triage → ship or defer (W12.T05)

| Candidate | Decision | Rationale |
|-----------|----------|-----------|
| **B1** Honest speedup accounting | **SHIP** (tech-report primary) | Unique dual-metric + fake-binary control; evidence already in committed `results/benchmark.json` |
| **B2** Packed XNOR productization | **SHIP** (companion / systems note) | Matches repo thesis + `bnn repro` / golden floors; citable artifact path |
| **B3** When not to binarize | **SHIP** (short companion) | Decision tree + bridges CLI; prevents false BNN wins vs INT4/bitnet.cpp |
| Literature-only BNN survey | **DEFER permanently** | No novel systems claim |
| Cross-modality “SOTA lab” | **DEFER permanently** | Proxies only; forbidden as golden |
| Standalone energy methodology | **DEFER** (folded into B1) | Insufficient alone |
| GPU 32× from `sign()` | **REJECT** | Thesis lock |

Drafting must preserve thesis lock: packed CPU/edge kernels + honest STE simulation; no fake GPU wins.

## Rejected standalone angles

- Literature-only survey of BNNs  
- Cross-modality “SOTA lab” from proxy CIFAR + synthetic audio  
- Energy methodology alone (folded into B1)  
- Any GPU 32× claim from `sign()` (forbidden by thesis lock)

## Status

Vault linked + triage closed for v1.0 narrative. Venue LaTeX still optional; claims must pass `bnn bridge figures` against goldens.
