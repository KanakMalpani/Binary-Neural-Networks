# Publication plan (W12.T02)

| Field | Value |
|-------|-------|
| **Status** | Draft plan (not submitted) |
| **Date** | 2026-07-25 |
| **Venue (candidates)** | Tech report on GitHub + optional workshop (MLSys / edge-AI) |

## Claims whitelist (must match goldens)

Allowed to claim in any write-up:

1. Aligned uint64 binary pack compression **32.00×** (theory, not latency)
2. Native XNOR-popcount GEMM **err = 0** vs ±1 FP when DLL/`.so` present
3. Dual-metric culture: theory vs wall-clock; never GPU 32× from `sign()`
4. MNIST / CIFAR / audio accuracies **within** `tests/golden_floors.json`
5. Linux + Windows CI; Linux native `.so` validated in Actions

Forbidden:

- GPU e2e 32× from STE/`sign()`
- Invented bench shapes as “the” golden
- Bit-identical floats across machines as a pass criterion
- Production ASR / full ImageNet SOTA as delivered

## Figure pipeline (W12.T03)

```bat
python scripts/pareto_report.py --demo --out results/pareto_demo.json --plot results/pareto_demo.png
```

Prefer figures generated from committed `results/*.json` + Pareto schema.
Manual polish OK; source JSON must stay in repo.

## Citation

See root `CITATION.cff` (version aligned to release tag).
