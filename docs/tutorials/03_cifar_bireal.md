# Tutorial 03 — CIFAR-10 Bi-Real proxy

**Master guide:** [`../GUIDE_E2E.md`](../GUIDE_E2E.md) · **Prev:** [02](02_wrap_linear.md) · **Next:** [04](04_image_cifar.md)

## Goal

Compare FP32 vs Bi-Real-style binary CNN on CIFAR-10 (ImageNet substitute).

```bat
pip install datasets   :: first time — dumps to data/cifar10_hf
bnn train-cifar --epochs 5 --subset 20000
```

Full 50k: `--subset 0` (longer). Recipe notes in `docs/13_TRAINING_QAT_DISTILL.md`.

For the richer image lane (ApproxSign, golden `image_cifar.json`), prefer
[04_image_cifar.md](04_image_cifar.md).

## Expect

Short schedules leave an accuracy gap (see `results/cifar10_proxy.json`). Full ImageNet train is an ADR non-goal; use published Bi-Real/ReActNet for scale-up.
