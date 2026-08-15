# QAT recipe + per-layer mode search + distill + BN fuse (WC-O)

Closes WC-O polish: runnable QAT/distill recipes, per-layer
**binary / ternary / skip** search, unified calibrate, always-on effectiveness
+ policy reasons, drop-in honesty tests, and BN fuse on the wrap path.

Accuracy tools only. Thesis lock: compression numbers stay theoretical pack
ratios; latency stays wall-clock; never claim GPU 32× from `sign()`.

---

## 1. Why a search and not a threshold

`score_layer_sensitivity` (W3.T05) wraps **one layer at a time** and ranks the
damage. That is a good diagnostic and a poor decision procedure, because layers
interact — two layers that each look harmless can be bad together, and the
per-layer view cannot see it.

`search_layer_modes` (W3.T06) measures the **whole model** at every step:

1. Start maximally aggressive — every eligible Linear binary.
2. Measure cosine against the FP32 teacher.
3. While below the quality floor, try relaxing each layer one step
   (`binary → ternary → skip`) and keep the single relaxation that buys the most
   quality.
4. Stop when the floor is met, or when nothing left helps.

Cost is `O(L)` probes per relaxation rather than the `3**L` of exhaustive search,
which is what makes it usable on a real stack.

```python
import torch
from bnn.wrap import search_layer_modes

report = search_layer_modes(model, calib_inputs, quality_floor=0.90)
print(report.binary, report.ternary, report.skipped)
print(report.final_cosine, report.compression())

from bnn.wrap import wrap_model
wrapped = wrap_model(model, mode="binary_xnor", exclude_exact=report.skipped)
```

### Measured trade-off (toy 3-Linear stack, 16×128 calibration)

| `quality_floor` | final cosine | theoretical compression | assignment |
|---|---|---|---|
| 0.00 | 0.271 | **32.0×** | 3 binary |
| 0.90 | 0.950 | 1.71× | 1 ternary, 2 skip |
| 0.999 | 1.000 | 1.00× | 3 skip |

Compression falls monotonically as the floor rises — enforced by a test, because
a search that ever reported *more* compression at *higher* quality would be lying.

The 0.00 row is the honest headline: **32× is available, at cosine 0.27.** That
is why the search exists, and why "32×" alone is never a result.

---

## 2. Unified calibrate (W3.T01)

One entrypoint dispatches on the argument type:

```python
from bnn.wrap import calibrate, CalibConfig

alpha = calibrate(weight_tensor, CalibConfig(method="absmean", per_channel=True))
report = calibrate(model, CalibConfig(method="percentile"), policy="hybrid_ffn")
print(report.to_dict()["n_layers"], report.scales_by_name().keys())
```

`calibrate_linear_scales` / `calibrate_model` remain the explicit helpers;
`wrap_model(..., calib=CalibConfig(...))` still applies scales at pack time.

---

## 3. Effectiveness + policy reasons + drop-in (W3.T02–T04)

Every `wrap_model` report now carries:

- `effectiveness` — measured dict after `attach_effectiveness`, or an explicit
  **unmeasured stub** (`measured=False`, `drop_in_ok=False`) so the field is
  never missing.
- `policy_reason` — non-empty string (auto recommender text or
  `policy=… mode=…`).

Drop-in honesty:

```python
from bnn.wrap import measure_agreement, attach_effectiveness, drop_in_ok

eff = measure_agreement(teacher_logits, student_logits, drop_in_threshold=0.85)
attach_effectiveness(report, eff)          # refuse if below threshold
attach_effectiveness(report, eff, force=True)  # claim only with --force
```

Unmeasured stubs refuse drop-in unless `force=True`.

---

## 4. QAT + distill (W3.T07 / W3.T08)

PTQ alone does not recover binary FFN quality. Two recovery APIs:

### Light STE (`light_qat_recover`)

```python
from bnn.wrap.qat import light_qat_recover

report = light_qat_recover(
    model,
    calib_x,
    teacher=fp32_reference,
    steps=200,
    lr=1e-3,
)
```

### Multi-batch distill (`distill_binary_student`) — beyond `distill_sketch.py`

```python
from bnn.wrap import distill_binary_student, DistillConfig

d = distill_binary_student(
    student, teacher, batches,
    cfg=DistillConfig(steps=80, temperature=2.0, lr=5e-3),
)
print(d.cosine_before, d.cosine_after, d.cosine_uplift)
```

Runnable demo:

```bash
python scripts/distill_wrap_demo.py --steps 80
```

Or CLI optimise with light QAT:

```bash
bnn optimise --policy auto --qat-steps 200 --force
```

### Order of operations

1. `fuse_bn_for_wrap_(model)` or `wrap_model(..., fuse_bn=True)` (W3.T09)
2. `search_layer_modes` → which layers can be binary
3. `distill_binary_student` / `light_qat_recover` → recover remaining binary layers
4. `wrap_model(..., exclude_exact=report.skipped)` → pack
5. `bnn profile` → confirm wall-clock win

---

## 5. BN fuse on the optimiser / wrap path (W3.T09)

```python
from bnn.wrap import fuse_bn_for_wrap_, wrap_model

fuse_bn_for_wrap_(model)          # Linear→BN1d pairs + BiRealBlock
# or
wrap_model(model, policy="hybrid_ffn", fuse_bn=True)
```

Eval-only fold. Does not change the thesis compression story.

**Residual for integrator:** wire `OptimiseConfig.fuse_bn` /
`OptimiseConfig.distill_steps` into `bnn/optimise.py` (outside Lane A ownership).

---

### Measured AND-gate (W3 / Wave S2)

Hybrid/binary wrap + short STE QAT on the **committed** `wrap_demo` shape
(hidden=4096, Sequential middles `3`/`5`, batch=64, no `--force`):

| Recipe | cosine | e2e vs FP | `drop_in_ok` | AND |
|---|---|---|---|---|
| PTQ `binary_xnor` (legacy golden) | 0.31 | ~4.8× | n/a | no (cosine) |
| **MSE STE + fold α, 200 steps** | **0.999** | **2.65×** | **true** | **yes** |
| Ultra TinyBlock PTQ hybrid | ~0.70 | host-noisy; committed ~1.61×; recheck paired median 1.40× | false | no |
| Ultra TinyBlock MSE+fold 200 | **0.9997** | paired median **1.386×**; isolated median **1.207×** | true | no (e2e) |
| Ternary + FP distill | 0.991 | 0.73× | true (forced) | **no** (e2e) |

Recipe (same shape; do not invent a new bench):

```bash
python scripts/wrap_existing_demo.py --mode binary_xnor --hidden 4096 --batch 64 --qat-steps 200
```

`light_qat_recover(..., logit_loss="mse", fold_alpha=True)` bakes per-out-channel
STE `alpha` into restored Linear magnitudes so wrap absmean calib matches QAT.
Packed path stays CPU XNOR — never GPU 32× from `sign()`.

Full table: [`docs/spikes/WRAP_HYBRID_085.md`](spikes/WRAP_HYBRID_085.md).
TinyBlock fail-closed recheck: [`docs/spikes/TINYBLOCK_HYBRID_085.md`](spikes/TINYBLOCK_HYBRID_085.md).
Do not update `results/ultra_wrap.json` until **both** gates hold without `--force`.

---

## Non-claims

- Toy numbers demonstrate mechanisms, not production accuracy.
- Distill / light QAT are recovery aids, not BitDistill-scale pipelines.
- Compression figures are theoretical pack ratios. Use `bnn profile` /
  `bnn bench` for wall-clock.
