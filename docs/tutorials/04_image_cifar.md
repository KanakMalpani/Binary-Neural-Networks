# Tutorial 04 — Image lane (CIFAR-10 Bi-Real)

**Goal:** Train FP vs Bi-Real binary CNNs on CIFAR-10, write results, understand honesty limits.

## Quick run

```bat
bnn train-image --epochs 8 --subset 30000 --seed 0 --approx-sign
```

Or:

```bat
python scripts/train_image.py --epochs 8 --train-subset 30000 --seed 0 --approx-sign
```

Committed golden: `results/image_cifar.json` (verify without retrain: `bnn repro`).

Outputs: `results/image_cifar.json` (+ `.md`) and refreshes `results/cifar10_proxy.json` for the eval suite.

## What you get

| Model | Role |
|-------|------|
| `fp32_cifar_cnn` | FP twin baseline |
| `binary_cifar_bireal` | FP stem/head + binary blocks + residuals |
| optional `tiny_vit_binary` | `--include-vit` — binary FFN tokens, FP attention |

## STE choice

Default: clipped Sign STE. Pass `--approx-sign` for Bi-Real ApproxSign backward (often better for deeper nets).

## Packed inference note

- **Linear / ViT FFN:** use `bnn.wrapper.wrap_model` / `PackedBinaryXNORLinear` for real CPU XNOR speedups.
- **Conv:** `wrap_conv_modules` packs weights (~32× size) but forward is **dequant + FP conv** — no native binary-conv DLL yet. Do not claim 32× wall-clock for Conv.

## ImageNet

Full ImageNet Bi-Real train is an **ADR ACCEPTED-NON-GOAL**. Stub: `bnn.vision.check_imagenet_folder`. CIFAR is the in-repo image evidence.

## Smoke test

```bat
pytest tests\test_vision_smoke.py -q
```
