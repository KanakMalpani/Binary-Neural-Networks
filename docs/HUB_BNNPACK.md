# Hub `.bnnpack` canaries

**Canaries / lab demos — not ImageNet SOTA.** Tiny packed artifacts so
`.bnnpack` is a **noun** on the Hugging Face Hub (llama.cpp won in part because
GGUF files exist in public). 32× is **uint64 pack compression**, not a GPU
speedup from `sign()`.

In-repo cards: [`cards/`](../cards/README.md). Encode:
`python scripts/encode_hf_canaries.py --out-dir results/hf_canaries`.
Load path: [tutorial 08](tutorials/08_HF_OPTIMISER.md).

## Collection

Public collection: **[bnn-lab `.bnnpack` canaries](https://huggingface.co/collections/KanakMalpani/bnn-lab-bnnpack-canaries-6a7f84448bdcaba4b5950eba)**
under [`KanakMalpani`](https://huggingface.co/KanakMalpani).

| Hub repo | What | Size class |
|----------|------|------------|
| [`KanakMalpani/bnn-lab-wrap-demo`](https://huggingface.co/KanakMalpani/bnn-lab-wrap-demo) | Packed Sequential middles **3** and **5**, **hidden=4096** | ~4 MiB packed (128 MiB FP32 for those two Linears) |
| [`KanakMalpani/bnn-lab-mnist-mlp-canary`](https://huggingface.co/KanakMalpani/bnn-lab-mnist-mlp-canary) | `BinaryMLP` hidden BinaryLinear layers, hidden=512 | tens of KiB |
| [`KanakMalpani/bnn-lab-codec-canary`](https://huggingface.co/KanakMalpani/bnn-lab-codec-canary) | Random `256×256` Linear encode | ~2 KiB packed |

Create + upload (write token required):

```bat
pip install -e ".[hf]" -c constraints.txt
python scripts/encode_hf_canaries.py --out-dir results/hf_canaries --upload --collection
```

If Hub create/upload fails (auth / org / rate-limit), the in-repo cards and
encode script still land. Packs stay gitignored (`*.bnnpack`); do not commit
`data/` datasets.

## Floors (quote these — do not invent)

From [`tests/golden_floors.json`](../tests/golden_floors.json) and committed
`results/*.json`:

**wrap_demo** (`results/wrap_demo.json`, hidden=4096, layers 3+5, MSE+fold_α
QAT 200 steps):

| Metric | Published | Floor |
|--------|-----------|-------|
| Weight compression (replaced) | 32.0× | `weight_compression_exact` 32.0 |
| Cosine vs FP | 0.9990 | `cosine_min` **0.85** |
| E2E speedup | 2.65× | `e2e_speedup_min` **1.5** |
| Status | `OK` without `--force` | AND-gate |

PTQ-only cosine on that shape is bounded by `cosine_max_without_qat` **0.5**.
The Hub wrap-demo pack is a **shape/codec canary** (PTQ bytes), not the QAT
checkpoint.

**ultra_wrap** (`results/ultra_wrap.json`) hybrid/binary TinyBlock: cosine
**~0.70**, status `REFUSE_DROP_IN_CLAIM`. Do not overclaim. Ternary on that
suite meets cosine and **loses** wall-clock (~0.73×) — that does not count.

**mnist** (`results/train_results.json`, seed 42, 3 epochs):

| Model | Recorded | Floor |
|-------|----------|-------|
| `fp32_mlp` | 97.67 | 96.0 |
| `binary_mlp` | 96.36 | **95.0** |
| `ternary_mlp` | 97.16 | 95.0 |

The Hub MNIST pack is a **codec canary** of untrained hidden BinaryLinears
unless you pass `--from-checkpoint`.

**codec**: `compression_exact_when_uint64_pack` **32.0**, `native_err_max` **0.0**.

## Honesty checklist

- [ ] Cards say canary / lab demo, not ImageNet SOTA
- [ ] 32× is pack compression, not GPU from `sign()`
- [ ] wrap_demo AND-gate quoted from committed JSON; Ultra TinyBlock still REFUSE
- [ ] No `data/` datasets uploaded

CIFAR Bi-Real weights are **not** in this drop (checkpoints gitignored; ImageNet
SOTA is a non-goal).
