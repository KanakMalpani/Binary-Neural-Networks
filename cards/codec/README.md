---
license: mit
library_name: bnn-lab
tags:
  - binary-neural-network
  - xnor
  - cpu
  - canary
  - lab-demo
  - not-sota
pipeline_tag: other
---

# bnn-lab codec `.bnnpack` (canary)

**Canary / lab demo — not ImageNet SOTA.** One random `256×256` Linear encoded
to `.bnnpack` v2 so strangers can download a **noun** (the format), round-trip
decode, and check **GEMM err = 0**.

Equivalent local command:

```bat
bnn encode --source random --in-features 256 --out-features 256 --out results/codec_canary.bnnpack
bnn decode --pack results/codec_canary.bnnpack
```

## What 32× means

**32× is exact uint64 pack compression** when `in_features % 64 == 0`
(`256/64 = 4` words). Not a GPU claim from `sign()`. Native XNOR GEMM **err = 0**
when the platform DLL/so is present; NumPy packed path is also err = 0
(`tests/golden_floors.json`: `native_err_max` 0.0, `compression_exact_when_uint64_pack` 32.0).

## Load

```python
from huggingface_hub import hf_hub_download
from bnn.codec import decode_file, packed_module_fp_err

path = hf_hub_download("KanakMalpani/bnn-lab-codec-canary", filename="model.bnnpack")
modules, meta = decode_file(path)
mod = modules["linear"]
assert packed_module_fp_err(mod) == 0.0
print(meta, mod.in_features, mod.out_features)
```

## License

MIT (same as [bnn-lab](https://pypi.org/project/bnn-lab/)).
