# Tutorial 01 — MNIST binary train + bench (≈15 min)

## Goal

Train a binary MLP on MNIST and confirm packed kernel correctness/speed.

## Steps

```bat
cd "C:\Users\mrkan\CRAZZY\Binary Neural Network"
pip install -e ".[dev]"
python -m bnn.kernels.compile_native
bnn validate-native
bnn export-check
bnn train --epochs 3 --seed 42 --model binary_mlp
bnn bench --reps 5
```

## Expect

- Native GEMM err = 0
- Compression ≈ 32×
- `binary_mlp` test acc ≥ ~95% when FP is ≥97%

## Notes

Training uses STE (not faster than FP). Inference wins need packed kernels.
