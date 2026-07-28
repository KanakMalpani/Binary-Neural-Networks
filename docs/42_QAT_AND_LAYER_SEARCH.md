# QAT recipe + per-layer mode search (W3.T06 / W3.T07)

Closes the two open WC-O polish items: a runnable QAT recipe, and an actual
**search** over per-layer binary / ternary / skip instead of a single global mode.

Both are accuracy tools. Neither changes the thesis: compression numbers stay
theoretical pack ratios, latency stays wall-clock, and the two are never mixed.

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

## 2. QAT recipe

PTQ alone does not recover binary FFN quality. `light_qat_recover` is a short STE
fine-tune that runs in seconds on a toy stack; it is **not** BitDistill-scale.

```python
from bnn.wrap.qat import light_qat_recover

report = light_qat_recover(
    model,                    # mutated in place; FFN Linears swapped to BinaryLinear
    calib_x,                  # a representative batch
    teacher=fp32_reference,   # distillation target (or pass loss_fn=...)
    steps=200,
    lr=1e-3,
)
# Layers come back as plain nn.Linear with the learned latent weights,
# so the model is immediately packable.
```

Then wrap and pack:

```python
from bnn.wrap import wrap_model
wrapped = wrap_model(model, mode="binary_xnor", policy="hybrid_ffn")
```

Or in one step from the CLI:

```bash
bnn optimise --policy auto --qat-steps 200 --force
```

### Recipe parameters that actually matter

| Knob | Toy stack | Real model | Why |
|---|---|---|---|
| `steps` | 40–200 | 10³–10⁵ | The dominant factor; a few dozen steps only removes the worst PTQ damage |
| `teacher` | optional | **required** | Distilling FP32 logits beats any label loss for recovery |
| `lr` | `1e-3` | `1e-5`–`1e-4` | Latent weights are already near a good solution; large steps destroy them |
| target layers | `ffn`/`mlp`/`fc1`/`fc2` | same | Wrapping attention is the documented accuracy trap — leave Q/K/V FP |
| `calib_x` | random | **real data** | Random activations do not exercise the distribution the scales were fit to |

`light_qat_recover` refuses to run without `teacher=` or `loss_fn=`. The old
self-argmax cross-entropy fallback was removed because it was a no-op at best.

### Order of operations

Search **before** QAT, not after:

1. `search_layer_modes` → which layers can be binary at all
2. `light_qat_recover` → recover the ones that stay binary
3. `wrap_model(..., exclude_exact=report.skipped)` → pack
4. `bnn profile` → confirm the wall-clock win is real

Doing QAT first wastes training on layers the search would have skipped.

---

## Non-claims

- The toy numbers above are a **3-Linear stack with random calibration data**;
  they demonstrate the mechanism, not production accuracy.
- `light_qat_recover` is a recovery aid, not a training pipeline. Production
  binary LLMs need BitDistill-scale distillation on real corpora.
- Compression figures are theoretical pack ratios throughout. Use `bnn profile`
  or `bnn bench` for anything wall-clock.
