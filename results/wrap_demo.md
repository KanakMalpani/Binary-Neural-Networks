# Wrap existing model demo

- Mode: `binary_xnor` hidden=4096 batch=64 qat_steps=200
- Replaced: ['3', '5']
- Weight compression (replaced): **32.00×**
- Model bytes: 147275816 → 17285160
- E2E latency: 50.24 ms → 18.94 ms (**2.65×**)
- Output cosine vs FP: **0.9990** (drop_in_ok=True, forced=False)
- Layer micro: torch Linear 19.21 ms | wrapped fwd 4.91 ms | gemm_only 3.72 ms
- **Kernel speedup (gemm_only vs torch Linear): 5.17×**

MSE STE QAT (fold learned alpha into Linear magnitudes) on Sequential middles 3/5, then packed binary_xnor. Cosine and e2e are measured; compression is theoretical pack ratio. Never claim GPU 32× from sign()/STE.
