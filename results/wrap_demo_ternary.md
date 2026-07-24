# Wrap existing model demo

- Mode: `ternary_weight_only` hidden=4096 batch=64
- Replaced: ['3', '5']
- Weight compression (replaced): **16.00×**
- Model bytes: 147275816 → 46612528
- E2E latency: 23.69 ms → 38.42 ms (**0.62×**)
- Output cosine vs FP: **0.9091**

E2E may lose to torch when stem/head/ReLU dominate or Python pack overhead is large; layer gemm_only shows true kernel ROI. Cosine<<1 for binary_xnor without QAT is expected (not a transparent accuracy-preserving wrap).
