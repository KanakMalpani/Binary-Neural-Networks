# Spike note — wrap hybrid 0.85 cosine **and** 1.5× e2e (Wave S2 / W3)

| Field | Value |
|-------|-------|
| **Status** | **PASS on `wrap_demo`** (same hidden=4096 / layers 3+5). Ultra TinyBlock still **FAIL-CLOSED** on e2e (see [`TINYBLOCK_HYBRID_085.md`](TINYBLOCK_HYBRID_085.md)) |
| **Date** | 2026-08-15 |
| **AND-gate** | cosine ≥ 0.85 **and** e2e ≥ 1.5× vs FP, hybrid/binary, **without `--force`** |
| **Ternary** | 0.991 cosine / 0.73× e2e — **does not count** |

## Intent

PTQ hybrid wrap sat at cosine ~0.70 (`ultra_wrap`) or ~0.31 (`wrap_demo`) while
XNOR e2e could already beat FP. The missing lever was a short QAT/distill
recipe on a **committed** shape, not a new bench.

## Recipe that met the gate

On `scripts/wrap_existing_demo.py` (hidden=4096, batch=64, replace Sequential
middles `3` and `5` only):

1. `light_qat_recover(..., logit_loss="mse", fold_alpha=True, steps=200)`
   targeting layers `3` and `5` (W+A `BinaryLinear` STE).
2. Fold learned per-out-channel `alpha` into restored Linear row magnitudes so
   wrap absmean calib matches QAT (Xavier leftover `alpha` was a pack mismatch).
3. `PackedBinaryXNORLinear` wrap — packed CPU XNOR, no `--force`.

```bash
python scripts/wrap_existing_demo.py --mode binary_xnor --hidden 4096 --batch 64 --qat-steps 200
```

## Measured table

Same calib/eval batch as the demo protocol (`torch.randn` after `set_repro_seed(0)`).

| Shape | Recipe | cosine | e2e vs FP | drop_in | forced | AND |
|-------|--------|--------|-----------|---------|--------|-----|
| wrap_demo 4096 | PTQ binary_xnor | 0.310 | ~4.8× (committed) | n/a | n/a | no |
| **wrap_demo 4096** | **MSE + fold α, 200** | **0.999** | **2.65×** (golden regen) | **true** | **false** | **yes** |
| ultra TinyBlock 512/2048 | PTQ hybrid | 0.699 | committed 1.61×; this host median 1.10× | false | false | no |
| ultra TinyBlock | legacy KL, no fold, 200 | 0.521 | noisy | false | false | no |
| ultra TinyBlock | MSE + fold α, 200 | 0.9997 | in-process median **1.386×**; isolated median **1.207×** (was ~1.25×) | true | false | **no** (e2e) |
| ultra TinyBlock | cosine loss + fold, 200 | 0.9999 | noisy | true | false | no* |
| ultra TinyBlock | weight-only STE + XNOR wrap | 0.561 | noisy | false | false | no |
| ultra TinyBlock | ternary + FP distill | 0.991 | 0.73× | true | true | **no** |

\*Ultra e2e on this host was not stably ≥1.5× even for the published PTQ wrap.
2026-08-15 recheck (AVX-512, torch=8): PTQ paired median 1.398×; MSE+fold paired
median 1.386× (range 1.350–1.390); isolated subprocess median 1.207× (2/7 ≥1.5).
QAT does not change the packed kernel; do not treat a noisy 2× from 12-rep
timings as the win. TinyBlock write-up:
[`TINYBLOCK_HYBRID_085.md`](TINYBLOCK_HYBRID_085.md).

Raw ultra sweep: [`WRAP_HYBRID_085_MEASURED.json`](WRAP_HYBRID_085_MEASURED.json).

## What was not done

- No new golden **shape**
- AND-gate **not** lowered
- `results/ultra_wrap.json` left as the no-QAT hybrid snapshot (e2e not stable here)
- Ternary 0.73× e2e not counted
- No GPU 32× from `sign()`

## Integrator

`results/wrap_demo.json` + `tests/golden_floors.json` `wrap_demo.cosine_min` /
`e2e_speedup_min` are the published AND-gate evidence. ROADMAP twins stay with
the integrator.
