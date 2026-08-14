# 08 — Hugging Face optimiser path

**Master guide:** [`../GUIDE_E2E.md`](../GUIDE_E2E.md) · **Prev:** [07](07_OPTIMISER_QUICKSTART.md)

**Goal:** load a tiny HF model → hybrid wrap → report (optional encode).  
**Requires:** `pip install -e ".[hf]"`. Network needed for first download.

## Warning (read first)

PTQ wrap of Transformers **without** real calibration data + QAT is pedagogical.
Attention / embeds are usually skipped (`hybrid_ffn`). For production LLMs on GPU
prefer INT4/FP8 (torchao / vLLM / AWQ). For BitNet-style 1.58-bit serve, prefer
**bitnet.cpp** — see `docs/23`–`24` and `scripts/bridges/`.

## CLI / script

```bat
pip install -e ".[hf]" -c constraints.txt
python scripts/hf_tiny_wrap_demo.py --out results/hf_tiny_wrap.json
```

Default model: `hf-internal-testing/tiny-random-BertModel` (tiny, CI-friendly).

## Python (product API)

```python
import torch
from transformers import AutoModelForSequenceClassification
from bnn.optimise import OptimiseConfig, optimise_model

name = "hf-internal-testing/tiny-random-BertModel"
model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2)
# Dummy batch for agreement metrics (real users: real calib loader)
batch = {
    "input_ids": torch.randint(0, 1000, (2, 16)),
    "attention_mask": torch.ones(2, 16, dtype=torch.long),
}

# HF forwards need kwargs — wrap modules first, then measure with a thin adapter
# if you need logits agreement. Minimal path: wrap only:
from bnn import wrap_model

wrapped, report = wrap_model(model, policy="hybrid_ffn", min_in_features=32)
print("replaced:", report.replaced)
print("compression (theory):", report.compression)
print(report.policy_reason)
```

For the full `optimise_model` path on a **plain** `nn.Module` with tensor inputs,
see [07_OPTIMISER_QUICKSTART.md](07_OPTIMISER_QUICKSTART.md). HF models that only
accept dict kwargs should use `wrap_model` (+ optional encode of packed Linears).

Prefer `bnn optimise` for the local toy / ultra demo path (not `bnn wrap --ultra`
in new docs).

## Optional pytest

```bat
pytest -q -m "hf" tests/test_hf_optimiser.py
```

Marked `hf` + `slow` so default `bnn repro` / CI stay offline-friendly.

## Encode packed layers

After wrap, portable artifacts:

```bat
bnn encode --source mlp --out results/toy.bnnpack
bnn decode --pack results/toy.bnnpack
```

Encoding an arbitrary HF tree into `.bnnpack` is still evolving (`.bnnpack` v2 =
W5.T05). Today: encode BinaryLinear / PackedBinaryXNORLinear layers that the
wrap path produced when names are Linear-shaped.

## Hub `.bnnpack` canaries (load path)

Tiny **canaries / lab demos — not ImageNet SOTA.** 32× is uint64 pack
compression, not GPU from `sign()`. Collection:
[bnn-lab `.bnnpack` canaries](https://huggingface.co/collections/KanakMalpani/bnn-lab-bnnpack-canaries-6a7f84448bdcaba4b5950eba)
([`../HUB_BNNPACK.md`](../HUB_BNNPACK.md)).

Wrap AND-gate is true on **wrap_demo hidden=4096** (cosine ≥ 0.85 **and** e2e
≥ 1.5× after MSE+fold_α QAT — `results/wrap_demo.json`). Ultra TinyBlock hybrid
still **REFUSE** (~0.70 cosine) in `results/ultra_wrap.json`. Do not overclaim.

```python
from huggingface_hub import hf_hub_download
from bnn.codec import decode_file, packed_module_fp_err

path = hf_hub_download(
    "KanakMalpani/bnn-lab-wrap-demo",
    filename="model.bnnpack",
)
modules, meta = decode_file(path)
layer = modules["3"]  # PackedBinaryXNORLinear, 4096×4096
print(meta["note"])
print("GEMM err", packed_module_fp_err(layer))  # 0
```

Also published:

- [`KanakMalpani/bnn-lab-mnist-mlp-canary`](https://huggingface.co/KanakMalpani/bnn-lab-mnist-mlp-canary)
  — hidden BinaryLinear codec canary; MNIST floors stay in
  `tests/golden_floors.json` / `results/train_results.json` (binary_mlp ≥ 95.0,
  recorded 96.36).
- [`KanakMalpani/bnn-lab-codec-canary`](https://huggingface.co/KanakMalpani/bnn-lab-codec-canary)
  — random 256×256 Linear, GEMM err = 0.

Offline encode (no Hub; packs gitignored):

```bat
python scripts/encode_hf_canaries.py --out-dir results/hf_canaries
```

`hf_hub_download` needs `pip install -e ".[hf]"`. `load_bnnpack` uses
`weights_only=True` and may soft-warn when the cache path sits outside lab
`results/` / `checkpoints/` / `data/` — treat Hub files as untrusted inputs.

## Honesty checklist

- [ ] Did not claim e2e 32× from compression alone
- [ ] Documented skipped attn / embed layers
- [ ] Pointed GPU users to INT4/FP8 bridges when appropriate
- [ ] Hub cards quoted floors; canary not ImageNet SOTA; Ultra wrap still REFUSE
