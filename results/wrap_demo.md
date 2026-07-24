# Wrap existing model demo

- Mode: `binary_xnor` hidden=4096 batch=32
- Replaced: ['3', '5']
- Weight compression (replaced): **32.00×**
- Model bytes: 147275816 → 13090856
- E2E latency: 21.55 ms → 18.65 ms (**1.16×**)
- Output cosine vs FP: **0.2832**
- Layer micro: torch Linear 12.76 ms | wrapped fwd 7.64 ms | gemm_only 6.01 ms
- **Kernel speedup (gemm_only vs torch Linear): 2.12×**

E2E may lose to torch when stem/head/ReLU dominate or Python pack overhead is large; layer gemm_only shows true kernel ROI. Cosine<<1 for binary_xnor without QAT is expected (not a transparent accuracy-preserving wrap).
