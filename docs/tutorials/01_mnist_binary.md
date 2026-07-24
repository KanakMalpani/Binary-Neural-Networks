# Tutorial 01 — MNIST binary train + bench (≈15 min)

**Master guide:** [`../GUIDE_E2E.md`](../GUIDE_E2E.md) · **Next:** [02_wrap_linear.md](02_wrap_linear.md)

## Goal

Train a binary MLP on MNIST and confirm packed kernel correctness/speed.

## Steps

```bat
pip install -e ".[dev]" -c constraints.txt
python -m bnn.kernels.compile_native
bnn validate-native
bnn export-check
bnn train --epochs 3 --seed 42 --model binary_mlp
bnn bench --reps 5
:: Or verify committed goldens without retraining:
bnn repro
```

## Expect

- Native GEMM err = 0 (Windows MSVC DLL); NumPy path err = 0 everywhere
- Compression **32×** exact
- `binary_mlp` test acc ≥ ~95% when FP is ≥97% (see `tests/golden_floors.json`)

## Notes

Training uses STE (not faster than FP). Inference wins need packed kernels.
Full agent/human repro: [`REPRODUCIBILITY.md`](../../REPRODUCIBILITY.md).
