# Tutorial 03 — CIFAR-10 Bi-Real proxy

## Goal

Compare FP32 vs Bi-Real-style binary CNN on CIFAR-10 (ImageNet substitute).

```bat
pip install datasets   # first time — dumps to data/cifar10_hf
bnn train-cifar --epochs 5 --subset 20000
```

Full 50k: `--subset 0` (longer). Recipe notes in `docs/13_TRAINING_QAT_DISTILL.md`.

## Expect

Short schedules leave an accuracy gap (see `results/cifar10_proxy.json`). Full ImageNet train is an ADR non-goal; use published Bi-Real/ReActNet for scale-up.
