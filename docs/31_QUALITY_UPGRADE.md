# 31 — Quality upgrade report (before → after)

**Date:** 2026-07-24  
**Version:** 0.2.0  
**Goal:** Make the repo feel like a serious open-source lab others (humans + AIs) trust.

## Multipliers shipped

| # | Area | Before | After |
|---|------|--------|-------|
| 1 | **Third-party repro** | Ad-hoc scripts, thin floors | `bnn repro`, `REPRODUCIBILITY.md`, `AGENTS.md`, golden floors v2 |
| 2 | **Install/DX** | Minimal pyproject | Versioned package, constraints, keywords/urls, extras, `python -m bnn` |
| 3 | **CLI** | Basic subcommands | `--version`, epilog/thesis, exit codes, fail-loud validate-native |
| 4 | **Determinism** | Partial `manual_seed` | `set_repro_seed` (CPU + deterministic algs) on train/wrap paths |
| 5 | **Golden gates** | Bench + MNIST only | MNIST + image + audio + wrap + live compression pytest |
| 6 | **Safety** | pickle/load unchecked | Path guards, `weights_only` prefer, CIFAR structure checks, no pickle NPZ |
| 7 | **Kernel robustness** | assert-based | Typed validation (dtype/shape/n words), clear errors |
| 8 | **Tests/CI** | Pytest + soft native | Pip cache, `not slow`, repro gates must PASS on Win+Linux |
| 9 | **Docs navigation** | Flat dump of 00–29 | `docs/README.md` index, rewritten README, accurate API, honest one-pager |
| 10 | **Results honesty** | SUMMARY could mis-label wrap | Dual-reporting table; theory ≠ wall-clock; cosine/QAT caveats |

## Acceptance checklist

- [x] `bnn repro` exits 0 on author machine
- [x] pytest green (incl. new CLI/paths/determinism/golden tests)
- [x] README / AGENTS / REPRODUCIBILITY excellent
- [x] CI workflow improved (cache + repro fail-hard)
- [x] Pushed to GitHub (see commit SHA in git log)
- [x] This report written

## How others reproduce (≤10 steps)

1. `git clone https://github.com/KanakMalpani/Binary-Neural-Networks.git`
2. `cd Binary-Neural-Networks`
3. `python -m pip install -U pip`
4. `pip install -e ".[dev]" -c constraints.txt`
5. Windows: `python -m bnn.kernels.compile_native`
6. `bnn repro`
7. Confirm `REPRO: PASS`
8. (Optional) read `results/SUMMARY.md`
9. (Optional) `bnn repro --mode full` for short smokes
10. Do **not** invent new benches — compare to `tests/golden_floors.json`

## Guarantees

| Identical | Tolerance-gated |
|-----------|-----------------|
| Compression **32×** | Accuracies within floors (±pp) |
| Native/NumPy GEMM **err = 0** (path applies) | Soft speedup floors (machine-dependent) |
| Thesis / decision tree | Wall-clock latencies |

Thesis lock unchanged. Datasets stay out of git.
