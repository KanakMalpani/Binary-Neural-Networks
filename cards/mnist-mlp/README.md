---
license: mit
library_name: bnn-lab
tags:
  - binary-neural-network
  - xnor
  - cpu
  - mnist
  - canary
  - lab-demo
  - not-sota
pipeline_tag: other
---

# bnn-lab MNIST MLP `.bnnpack` (canary)

**Canary / lab demo — not ImageNet SOTA, not a MNIST leaderboard submission.**

Packed **hidden** `BinaryLinear` layers from `bnn.models.BinaryMLP` (`hidden=512`).
FP stem (`784→512`) and FP head (`512→10`) are **not** in this pack (standard
BNN practice; encode skips them).

## This file vs published MNIST numbers

This Hub artifact is a **codec canary** (seed-0 `build_model("binary_mlp")`
BinaryLinear blobs). It is **not** the 3-epoch trained checkpoint
(`checkpoints/binary_mlp.pt` is gitignored). Do not report this file's
inference accuracy as the lab MNIST result.

Published STE train floors (CPU, seed **42**, 3 epochs) from
[`results/train_results.json`](https://github.com/KanakMalpani/Binary-Neural-Networks/blob/main/results/train_results.json)
and [`tests/golden_floors.json`](https://github.com/KanakMalpani/Binary-Neural-Networks/blob/main/tests/golden_floors.json)
`mnist`:

| Model | Recorded test acc | Floor |
|-------|-------------------|-------|
| `fp32_mlp` | **97.67** | `fp32_mlp_min_acc`: 96.0 |
| `binary_mlp` | **96.36** | `binary_mlp_min_acc`: **95.0** |
| `ternary_mlp` | **97.16** | `ternary_mlp_min_acc`: 95.0 |

Gap gate: `gap_max_pp_fp_vs_binary`: **3.0** pp when FP ≥ `fp_for_gap_gate` 97.0.
Retrain + re-encode: `bnn train --model binary_mlp --epochs 3 --seed 42` then
`python scripts/encode_hf_canaries.py --only mnist-mlp --from-checkpoint checkpoints/binary_mlp.pt`.

## What 32× means

**32× is uint64 pack compression** of aligned BinaryLinear weights
(`in_features % 64 == 0`), not a GPU speedup from `sign()`. Training with STE
is simulation.

## Load

```python
from huggingface_hub import hf_hub_download
from bnn.codec import decode_file, packed_module_fp_err

path = hf_hub_download(
    "KanakMalpani/bnn-lab-mnist-mlp-canary", filename="model.bnnpack"
)
modules, meta = decode_file(path)
for name, mod in modules.items():
    print(name, mod.in_features, mod.out_features, packed_module_fp_err(mod))
```

Tutorial: [`docs/tutorials/08_HF_OPTIMISER.md`](https://github.com/KanakMalpani/Binary-Neural-Networks/blob/main/docs/tutorials/08_HF_OPTIMISER.md).

## License

MIT (same as [bnn-lab](https://pypi.org/project/bnn-lab/)). MNIST itself is not
bundled (lab `data/` is gitignored).
