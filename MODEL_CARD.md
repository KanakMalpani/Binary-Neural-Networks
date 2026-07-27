# Model card — Binary Neural Network Optimiser (lab / beta)

**Task:** W10.T01 · **Status:** draft toward world-class WC-D3  
**Version:** see `bnn.__version__` (currently 0.2.x lab → target 1.0)

## Model / product summary

This repository is an **optimiser toolkit** for packing neural net weights (and
optionally activations) to binary / ternary formats for **CPU / edge** inference
with real XNOR-popcount kernels — not a single pretrained foundation model.

| Artifact | Description |
|----------|-------------|
| STE zoo | MNIST MLP, CIFAR Bi-Real proxy, Tiny ViT, seq2seq reverse |
| Wrap / optimise | `bnn.optimise` / `bnn wrap` hybrid policies |
| Codec | `.bnnpack` v1 portable packed Linear weights |

## Intended use

- Research and education on packed binary/ternary CPU kernels
- Hybrid wrap of wide FFN layers where native GEMM helps
- Honest comparison of **theory compression** vs **wall-clock**

## Out of scope / failure modes

| Failure | Why |
|---------|-----|
| GPU 32× from `sign()` | Forbidden — use INT4/FP8 stacks |
| Drop-in HF LLM quality without QAT | PTQ cosine often collapses; refuse without metrics |
| Small GEMMs / attn projections | Packing overhead dominates; hybrid policy skips |
| Stock phone NPU 1-bit | Vendors ship INT8/INT4 (`docs/20`) |
| Production ASR | Audio lane is synthetic pedagogy |

## Metrics honesty

- **32×** = aligned uint64 binary pack ratio (exact), not e2e latency
- Native GEMM **err = 0** vs ±1 FP when DLL/SO present
- Golden floors: `tests/golden_floors.json` + committed `results/*.json`

## Ethical considerations

Binary/ternary compression can silently degrade decision quality. Always report
`effectiveness` / refuse drop-in claims below threshold unless `--force` is
explicit. Dual-use: same as any ML toolkit — do not deploy without evaluation
on the target domain.

## Citation

See [`CITATION.cff`](CITATION.cff) when present; otherwise cite the GitHub repo
https://github.com/KanakMalpani/Binary-Neural-Networks
