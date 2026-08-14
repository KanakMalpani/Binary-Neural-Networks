---
title: bnn-lab wrap paradox
emoji: ⚖️
colorFrom: blue
colorTo: yellow
sdk: gradio
sdk_version: 6.15.1
app_file: app.py
python_version: "3.12"
short_description: Pack 32× ≠ drop-in. CPU wrap paradox, dual metrics.
---

# bnn-lab — wrap paradox (CPU)

Live [`optimise_model`](https://github.com/KanakMalpani/Binary-Neural-Networks) demo on a **tiny** MLP or CNN. Hardware is **cpu-basic** (2 vCPU / 16 GB). There is **no GPU** and **no ZeroGPU**.

## Thesis lock

**32× is uint64 pack compression (size), not GPU latency from `sign()`.** Dual metrics stay separate: pack ratio vs wall-clock vs cosine. A wrap whose cosine collapses is **REFUSE**, not a drop-in replacement.

## Three columns (live)

| Column | What the library does |
|--------|------------------------|
| **FP32** | Unwrapped teacher. Cosine = 1. Baseline size and e2e ms. |
| **Binary packed** | `policy=hybrid_ffn`, `mode=binary_xnor`, no QAT. Pack ~32× on replaced FFN weights. Cosine is usually junk → **REFUSE**. |
| **Ternary + QAT** | Short FP MSE distill, then `policy=ternary_wo`. Pack ~16×. Cosine often meets 0.85; e2e often **loses** to FP (no ternary GEMM). |

Live numbers are **measured on this Space**, not copied from JSON. A later wrap win can show up after bumping `bnn-lab` in `requirements.txt`.

## Published goldens (committed shapes — labels only)

These are **snapshots** from the repo, not this host's live run. Default **auto** is the hybrid path, not the legacy 0.31 wrap.

| Snapshot | Cosine | E2E vs FP | Status |
|----------|--------|-----------|--------|
| Default auto / hybrid (`results/ultra_wrap.json` primary, d=512 ff=2048 batch=64) | ~0.70 | ~1.61× | **REFUSE** (`drop_in_ok: false`) |
| Legacy `results/wrap_demo.json` (binary_xnor, no QAT, hidden=4096 batch=64) | 0.31 | ~4.8× | not drop-in (cosine junk) |
| Ternary + QAT (`results/ultra_wrap.json` `ternary_accurate_path`) | ~0.991 | ~0.73× | cosine OK; **slower** than FP |

Sources: [ultra_wrap.json](https://github.com/KanakMalpani/Binary-Neural-Networks/blob/main/results/ultra_wrap.json), [wrap_demo.json](https://github.com/KanakMalpani/Binary-Neural-Networks/blob/main/results/wrap_demo.json).

The live MLP uses the same **TinyBlock** architecture as `ultra_wrap` (d=512, ff=2048) with a smaller batch so cpu-basic stays interactive. It does **not** re-run the 4096-wide `wrap_demo` shape.

## Deploy (human)

Hugging Face returns **HTTP 402** for Gradio `cpu-basic` on free accounts (PRO required). After [huggingface.co/pro](https://huggingface.co/pro):

```bash
hf repos create KanakMalpani/bnn-lab-wrap-paradox --type space --space-sdk gradio --flavor cpu-basic --public
hf upload KanakMalpani/bnn-lab-wrap-paradox . --type space
```

Do **not** attach ZeroGPU. This demo is CPU wrap + dual metrics; 32× is pack size, not GPU from `sign()`.

## Local

```bash
pip install -r requirements.txt
python app.py
```

From a clone of the lab (editable `bnn` instead of the PyPI pin):

```bash
pip install -e ".[dev]" -c constraints.txt
python demo/space/app.py
```

## Links

- GitHub: [KanakMalpani/Binary-Neural-Networks](https://github.com/KanakMalpani/Binary-Neural-Networks)
- PyPI: [`bnn-lab==1.0.0`](https://pypi.org/project/bnn-lab/1.0.0/)
- Guide: [docs/GUIDE_E2E.md](https://github.com/KanakMalpani/Binary-Neural-Networks/blob/main/docs/GUIDE_E2E.md)
