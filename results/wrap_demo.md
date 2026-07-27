# Wrap existing model demo

- Mode: `binary_xnor` hidden=4096 batch=64
- Replaced: ['3', '5']
- Weight compression (replaced): **32.00×**
- Model bytes: 147275816 → 17285160
- E2E latency: 34.14 ms → 7.09 ms (**4.82×**)
- Output cosine vs FP: **0.3102**
- Layer micro: torch Linear 14.12 ms | wrapped fwd 1.56 ms | gemm_only 2.05 ms
- **Kernel speedup (gemm_only vs torch Linear): 6.90×**

E2E may lose to torch when stem/head/ReLU dominate or Python pack overhead is large; layer gemm_only shows true kernel ROI. Cosine<<1 for binary_xnor without QAT is expected (not a transparent accuracy-preserving wrap).
