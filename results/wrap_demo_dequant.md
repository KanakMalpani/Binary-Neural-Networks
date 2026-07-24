# Wrap existing model demo

- Mode: `binary_weight_only_dequant`
- Replaced layers: ['3', '5']
- Weight compression (replaced only): **32.00×**
- Model bytes FP → wrapped: 40083496 → 7577648
- Latency FP → wrapped: 7.00 ms → 8.98 ms (**0.78×**)
- Output cosine vs FP: 0.7201
- Native kernel: False
- Note: Packed storage but dequant GEMM — often slower.
