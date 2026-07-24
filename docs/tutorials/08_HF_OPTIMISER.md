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

## Honesty checklist

- [ ] Did not claim e2e 32× from compression alone
- [ ] Documented skipped attn / embed layers
- [ ] Pointed GPU users to INT4/FP8 bridges when appropriate
