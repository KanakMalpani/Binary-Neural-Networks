# Allowed benchmark shapes (W7.T02)

**Rule:** do **not** invent alternate shapes and treat them as the golden.
Compare only to `tests/golden_floors.json` and committed `results/*.json`.

## Published shapes (canonical)

| Lane | Shape / protocol | Golden |
|------|------------------|--------|
| Pack compression | Aligned uint64 binary pack | `export-check` / floors 32.00× |
| Native GEMM | Packed vs ±1 FP | err = 0 |
| Kernel microbench | Documented in `bnn bench` / `results/` | wall-clock soft |
| Wrap demo | Wide MLP hidden=4096 class | `results/wrap_demo.json` |
| Ultra / optimise | TinyBlock d=512/ff=2048 + wide probe 2048/8192 | `results/ultra_wrap.json` |
| MNIST STE | `bnn train` default | `results/mnist_*.json` |
| CIFAR Bi-Real | `train-image` / cifar floors | `results/cifar_*.json` |
| Audio synth | `train-audio` | `results/audio_*.json` |
| Seq2seq reverse | `train-seq2seq` | `results/seq2seq_*.json` |

## Allowed

- Re-running the **same** shapes on new hardware (floats may differ)
- Adding **optional** probes clearly labeled non-golden
- Soft latency budgets that do not change pass/fail goldens without CHANGELOG + justification

## Forbidden

- New matrix sizes sold as “the” compression or speedup golden
- Claiming GPU e2e 32× from theory
- Quietly editing floors to pass a regression
