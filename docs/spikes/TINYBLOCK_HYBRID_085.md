# Spike note — Ultra TinyBlock hybrid AND-gate (d=512 / ff=2048)

| Field | Value |
|-------|-------|
| **Status** | **FAIL-CLOSED** — cosine passes; e2e not stably ≥1.5× |
| **Date** | 2026-08-15 |
| **Shape** | committed `ultra_wrap` TinyBlock `d=512` / `ff=2048` / batch=64 |
| **AND-gate** | hybrid/binary cosine ≥ 0.85 **and** e2e ≥ 1.5× vs FP, **without `--force`** |
| **Ternary** | 0.991 cosine / 0.73× e2e — **does not count** |

`wrap_demo` hidden=4096 already shipped (PR #39: cosine 0.999 / e2e 2.65×). This
spike does **not** retarget that bench.

## Intent

Close the remaining wrap gap on **the same TinyBlock shape**: MSE STE + fold α
already recovers cosine (~0.9997). The leftover question is whether packed FFN
XNOR can also hold **stable** ≥1.5× e2e vs FP without `--force`.

## Recipe (same shape; no new bench)

```text
light_qat_recover(..., logit_loss="mse", fold_alpha=True, steps=200)
  layer_names=["ffn_fc1", "ffn_fc2"]
wrap_model(..., mode="binary_xnor", policy="hybrid_ffn")  # no --force
```

32× remains uint64 pack compression, not GPU from `sign()`.

## Measured (this host, AVX-512 native DLL, torch intra-op=8)

Paired in-process timings (FP then wrap in the same process). Isolated =
one trial per subprocess after a saved QAT+wrap checkpoint.

| Protocol | Recipe | cosine | e2e vs FP | drop_in | forced | AND |
|----------|--------|--------|-----------|---------|--------|-----|
| committed `ultra_wrap.json` primary | PTQ hybrid | 0.699 | 1.61× (that regen) | false | false | no |
| in-process 3× warmup40/reps80 | PTQ hybrid | 0.6988 | median **1.398×** [1.218, 1.403] | false | false | **no** |
| in-process 3× warmup40/reps80 | MSE + fold α, 200 | **0.9997** | median **1.386×** [1.350, 1.390] | true | false | **no** (e2e) |
| short-rep 3× warmup3/reps12 | MSE + fold α, 200 | 0.9997 | median 1.407× [1.216, 1.746] | true | false | **no** (noisy) |
| isolated subprocess 7× warmup40/reps80 | MSE + fold α, 200 | 0.9997 | median **1.207×** [0.804, 2.091]; 2/7 ≥1.5 | true | false | **no** |
| isolated subprocess 7× demo warmup5/reps25 | MSE + fold α, 200 | 0.9997 | median **1.175×** [0.552, 6.149]; 1/7 ≥1.5 | true | false | **no** |

QAT does not change the packed kernel. Short-rep 2× outliers are load/autograd
noise, not a gate pass.

Raw JSON: [`WRAP_HYBRID_085_MEASURED.json`](WRAP_HYBRID_085_MEASURED.json)
(`tinyblock_recheck` block).

## Why e2e stalls on this shape

TinyBlock keeps embed / attn / lm_head in FP (`hybrid_ffn`). On this host a
layer split (warmup 20 / reps 40) was:

| Part | ms |
|------|----|
| embed+ReLU | 0.46 |
| attn+ReLU | 0.33 |
| FFN FP (`fc2(relu(fc1))`) | 1.70 |
| FFN wrapped | 0.90 (**1.89×** vs FP FFN) |
| lm_head | 0.06 |
| pack activations (fc1) | 0.04 |
| native scaled GEMM (fc1) | 0.35 |

Non-FFN is ~0.85 ms. With FFN wrap at 0.90 ms, predicted e2e ≈ 1.46× against a
~2.5 ms FP forward — under 1.5× even if wrap overhead went to zero on the GEMM
epilogue. Hitting 1.5× would need FFN wrap ≳2.0× **and** a quiet host. This
box did not provide that stably (OpenMP default 16 threads vs torch 8 also
oversubscribes; native=8 still median <1.5).

## What was not done

- AND-gate **not** lowered
- No new golden shape / width
- `results/ultra_wrap.json` and `tests/golden_floors.json` **not** updated
- Ternary 0.73× e2e not counted
- `bnn/kernels/packed.py` not edited (thesis: 32× is pack compression)
- No GPU 32× from `sign()`

## Integrator

Leave Ultra TinyBlock hybrid as `REFUSE` in committed `ultra_wrap.json`.
`wrap_demo` hidden=4096 remains the published AND-gate. ROADMAP twins stay
with the integrator.
