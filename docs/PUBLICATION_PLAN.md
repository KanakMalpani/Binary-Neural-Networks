# Publication plan (W12.T02)

| Field | Value |
|-------|-------|
| **Status** | In-repo B1 skeleton (not submitted; **arXiv is human**) |
| **Date** | 2026-08-15 |
| **Venue (candidates)** | Tech report on GitHub + optional workshop (MLSys / edge-AI) |
| **B1 skeleton** | [`docs/papers/B1_STOP_CLAIMING_32X.md`](papers/B1_STOP_CLAIMING_32X.md) |
| **Paper vault** | `C:\00 Research Papers` — see [`docs/32_NOVEL_PAPER_CANDIDATES.md`](32_NOVEL_PAPER_CANDIDATES.md) |
| **Papers with Code** | Official-code URL ready; **register after arXiv** (human) |

## Claims whitelist (must match goldens)

Allowed to claim in any write-up — each row ties to `tests/golden_floors.json` and/or committed `results/*.json`:

| ID | Claim | Evidence gate |
|----|-------|---------------|
| C1 | Aligned uint64 binary pack compression **32.00×** (theory, not latency) | `compression_exact_when_uint64_pack` + `results/wrap_demo.json` / benchmark theoretical |
| C2 | Native XNOR-popcount GEMM **err = 0** vs ±1 FP when DLL/`.so` present | `native_err_max: 0` + `results/benchmark.json` |
| C3 | Dual-metric culture: theory vs wall-clock; never GPU 32× from `sign()` | Thesis lock + floors `notes` |
| C4 | MNIST binary MLP within floors | `mnist.binary_mlp_min_acc` vs `results/train_results.json` |
| C5 | CIFAR Bi-Real proxy within floors | `image_cifar.binary_bireal_min_acc` vs `results/image_cifar.json` |
| C6 | Audio synth binary CNN within floors | `audio_synth.binary_cnn_min_acc` vs `results/audio_synth.json` |
| C7 | Linux + Windows CI; Linux native `.so` validated in Actions | CI matrix / WC-R2 |

Machine-check the whitelist:

```bat
bnn bridge figures --out results/figures_manifest.json
```

Forbidden:

- GPU e2e 32× from STE/`sign()`
- Invented bench shapes as “the” golden
- Bit-identical floats across machines as a pass criterion
- Production ASR / full ImageNet SOTA as delivered

## Figure pipeline (W12.T03)

```bat
bnn bridge figures --plot-dir results/figures
bnn pareto --from-results --out results/pareto_from_results.json --plot results/pareto_from_results.png
python scripts/pareto_report.py --demo --out results/pareto_demo.json --plot results/pareto_demo.png
```

Prefer figures generated from committed `results/*.json` + Pareto / figures-manifest schemas.
Manual polish OK; source JSON must stay in repo. **No invented goldens.**

## Bridges (CLI)

```bat
bnn bridge list
bnn bridge gpu --probe
bnn bridge cpu-llm
```

See [`docs/23_BITNET_CPP_BRIDGE.md`](23_BITNET_CPP_BRIDGE.md) and [`docs/24_GPU_INT4_FP8_LANE.md`](24_GPU_INT4_FP8_LANE.md).

## Papers with Code (code link)

Official implementation (no arXiv ID yet):
<https://github.com/KanakMalpani/Binary-Neural-Networks>

| Field | Value |
|-------|-------|
| Title | Stop Claiming 32×: Honest Speedup Accounting for Binary Neural Networks |
| Skeleton | [`docs/papers/B1_STOP_CLAIMING_32X.md`](papers/B1_STOP_CLAIMING_32X.md) |
| Citation | root [`CITATION.cff`](../CITATION.cff) (`bnn-lab` **1.0.0**) |
| Tasks | Quantization / model compression — **not** ImageNet SOTA, **not** ASR product |
| Human | After arXiv: add paper by ID, mark this repo **official code** |

Do **not** upload to arXiv or click PwC submit from CI / this PR.

## Citation

See root `CITATION.cff` (version aligned to release tag). Cite the B1 skeleton path until an arXiv ID exists.
