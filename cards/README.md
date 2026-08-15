# Hub `.bnnpack` canaries (in-repo cards)

Lab **canaries**, not ImageNet SOTA. These Markdown files are the source of
truth for Hugging Face model cards. Encode + upload:

```bat
python scripts/encode_hf_canaries.py --out-dir results/hf_canaries --upload
```

`--upload` needs a Hugging Face write token (`hf auth login`). Packs are
**not** committed (`*.bnnpack` is gitignored). Collection notes:
[`docs/HUB_BNNPACK.md`](../docs/HUB_BNNPACK.md)
([live collection](https://huggingface.co/collections/KanakMalpani/bnn-lab-bnnpack-canaries-6a7f84448bdcaba4b5950eba)).
Load path: [tutorial 08](../docs/tutorials/08_HF_OPTIMISER.md).

| Id | Suggested Hub repo | Card |
|----|--------------------|------|
| wrap-demo | [`KanakMalpani/bnn-lab-wrap-demo`](https://huggingface.co/KanakMalpani/bnn-lab-wrap-demo) | [`wrap-demo/README.md`](wrap-demo/README.md) |
| mnist-mlp | [`KanakMalpani/bnn-lab-mnist-mlp-canary`](https://huggingface.co/KanakMalpani/bnn-lab-mnist-mlp-canary) | [`mnist-mlp/README.md`](mnist-mlp/README.md) |
| codec | [`KanakMalpani/bnn-lab-codec-canary`](https://huggingface.co/KanakMalpani/bnn-lab-codec-canary) | [`codec/README.md`](codec/README.md) |

## Honesty (do not weaken)

- **32×** is uint64 pack compression of replaced Linear weights, not GPU from `sign()`.
- Wrap AND-gate is true on **wrap_demo hidden=4096** (cosine ≥ 0.85 **and** e2e ≥ 1.5× after MSE+fold_α QAT). See `results/wrap_demo.json`.
- Ultra TinyBlock hybrid still **REFUSE** (~0.70 cosine) in `results/ultra_wrap.json`. Do not overclaim.
- Quote floors from `tests/golden_floors.json` / committed `results/*.json` only.
