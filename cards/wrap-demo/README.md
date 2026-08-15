---
license: mit
library_name: bnn-lab
tags:
  - binary-neural-network
  - xnor
  - cpu
  - quantization
  - canary
  - lab-demo
  - not-sota
pipeline_tag: other
---

# bnn-lab wrap-demo `.bnnpack` (canary)

**Canary / lab demo — not ImageNet SOTA.** Packed binary Linear weights for the
committed `wrap_demo` shape (wide Sequential, **hidden=4096**, layers **3** and
**5**). Format: [`.bnnpack` v2](https://github.com/KanakMalpani/Binary-Neural-Networks)
(`weights_only` load, content hashes).

## What 32× means

**32× is uint64 pack compression** of the replaced Linear weights, not a GPU
speedup from `sign()` / STE. Wall-clock is a separate metric.

## Dual-metric AND-gate (same shape — not this file's bytes)

This Hub file is a **shape/codec canary** (PTQ pack of seed-0 random-init
middles). It is **not** the QAT checkpoint behind the published JSON.

On **wrap_demo hidden=4096**, after MSE + fold_α QAT (200 steps), committed
[`results/wrap_demo.json`](https://github.com/KanakMalpani/Binary-Neural-Networks/blob/main/results/wrap_demo.json):

| Metric | Published | Floor (`tests/golden_floors.json` → `wrap_demo`) |
|--------|-----------|--------------------------------------------------|
| Weight compression (replaced) | **32.0×** | `weight_compression_exact`: 32.0 |
| Output cosine vs FP | **0.9990** | `cosine_min`: **0.85** |
| E2E speedup vs FP Sequential | **2.65×** | `e2e_speedup_min`: **1.5** |
| Status | `OK` (`drop_in_ok=true`, `forced=false`) | AND-gate without `--force` |

PTQ-only cosine on this shape is bounded by `cosine_max_without_qat`: **0.5**
(legacy ~0.31). Do not quote PTQ as drop-in.

## Not this pack — Ultra TinyBlock still REFUSE

[`results/ultra_wrap.json`](https://github.com/KanakMalpani/Binary-Neural-Networks/blob/main/results/ultra_wrap.json)
hybrid/binary TinyBlock: cosine **~0.70** (`0.6988`), status
`REFUSE_DROP_IN_CLAIM`. Ternary on that suite meets cosine but **loses**
wall-clock (~0.73×). Ternary 0.991 / 0.73× does **not** count as the AND-gate.

## Load

```python
from huggingface_hub import hf_hub_download
from bnn.codec import decode_file, packed_module_fp_err

path = hf_hub_download("KanakMalpani/bnn-lab-wrap-demo", filename="model.bnnpack")
modules, meta = decode_file(path)
mod = modules["3"]  # PackedBinaryXNORLinear, in=4096 out=4096
print(meta["canary"])
print("GEMM err", packed_module_fp_err(mod))  # 0
```

Needs `pip install bnn-lab huggingface_hub` (or `pip install -e ".[hf]"` from
the repo). Tutorial: [`docs/tutorials/08_HF_OPTIMISER.md`](https://github.com/KanakMalpani/Binary-Neural-Networks/blob/main/docs/tutorials/08_HF_OPTIMISER.md).

Offline encode (no Hub):

```bat
python scripts/encode_hf_canaries.py --only wrap-demo --out-dir results/hf_canaries
```

## Files

- `model.bnnpack` — packed layers `3` and `5` (binary XNOR, ~4 MiB packed vs
  128 MiB FP32 for those two Linears).
- This card.

## License

MIT (same as [bnn-lab](https://pypi.org/project/bnn-lab/)).
