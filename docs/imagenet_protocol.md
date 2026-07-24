# ImageNet protocol stub (P7.T2)

Full ImageNet Bi-Real **training** is an ADR **ACCEPTED-NON-GOAL** for this repo.

## What this repo provides

1. **CIFAR-10 Bi-Real** as the in-repo image evidence: `bnn train-image`.
2. Folder layout check: `bnn.vision.check_imagenet_folder(root)`.

## If you bring your own ImageNet-style folder

```
<root>/train/<class>/*.jpg
<root>/val/<class>/*.jpg
```

Use the same Bi-Real recipe (FP stem/head, binary blocks, BN, ApproxSign optional),
start from CIFAR hyperparameters, and expect multi-GPU days — not a laptop MVP.

## Production vision on GPU

Prefer INT8/FP8 TorchAO / TensorRT — not classic BNN for datacenter ImageNet.
