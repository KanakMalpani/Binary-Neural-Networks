# ImageNet protocol (M6 / W6.T07)

Full ImageNet Bi-Real **training** is an ADR **ACCEPTED-NON-GOAL** for this repo
and is **never** a `bnn repro` / CI accuracy gate. This document is the **dataset
contract** plus how to run the protocol runner.

## What this repo provides

1. **CIFAR-10 Bi-Real** as the in-repo image evidence: `bnn train-image`
   (floors in `tests/golden_floors.json` → `results/image_cifar.json`).
2. **ResNet-BiReal CIFAR reference** (W4.T05): `bnn.vision.ResNetBiRealCIFAR` /
   `bnn train-image --include-resnet`.
3. **ImageNet folder protocol runner** (smoke / proxy OK):
   `python scripts/imagenet_protocol.py --mode smoke`.
4. Layout + contract helpers: `bnn.vision.check_imagenet_folder`,
   `bnn.vision.IMAGENET_DATASET_CONTRACT`.

## Dataset contract (ImageFolder)

```
<root>/train/<class>/*.{jpg,jpeg,png,...}
<root>/val/<class>/*.{jpg,jpeg,png,...}
```

| Field | Full ImageNet (reference) | Proxy minimum (accepted) |
|-------|---------------------------|--------------------------|
| Classes | 1000 | ≥2 |
| Images / class / split | ~1.2k train avg | ≥1 |
| Input size | 224 | 32 / 64 / 224 OK for smoke |
| SOTA top-1 gate | **No** | **No** |

Contract schema id: `bnn_imagenet_folder_contract_v1`
(see `bnn.vision.IMAGENET_DATASET_CONTRACT` or
`python scripts/imagenet_protocol.py --mode contract`).

**Do not** invent new ImageNet golden shapes for CI. Do not commit datasets under
`data/`.

## Protocol runner

```bash
# Print / write contract JSON
python scripts/imagenet_protocol.py --mode contract

# Validate a local ImageNet-style tree (requires images in class dirs)
python scripts/imagenet_protocol.py --mode check --root /path/to/ImageNet

# Scaffold tiny proxy under data/_imagenet_proxy_smoke (gitignored via data/)
python scripts/imagenet_protocol.py --mode proxy --root data/_imagenet_proxy_smoke

# Default: proxy + 1 STE train step on ResNet-BiReal (CIFAR or ImageNet stem)
python scripts/imagenet_protocol.py --mode smoke \
  --out results/imagenet_protocol_smoke.json
```

Smoke pass criteria: layout proxy `ok`, logits finite after one Adam step.
**Not** criteria: ImageNet top-1, multi-day schedules, bit-identical floats.

For ImageNet-stem smoke (7×7 + max-pool), use a larger spatial size:

```bash
python scripts/imagenet_protocol.py --mode smoke --imagenet-stem --image-size 224 --width 16
```

## If you bring your own full ImageNet

Use the same Bi-Real recipe (FP stem/head, binary residual units, BN,
ApproxSign optional), start from CIFAR hyperparameters, and expect multi-GPU
days — not a laptop MVP. Prefer reporting dual metrics (packed size + wall-clock)
over claiming theoretical 32× as latency.

## Production vision on GPU

Prefer INT8/FP8 TorchAO / TensorRT — not classic BNN for datacenter ImageNet.
