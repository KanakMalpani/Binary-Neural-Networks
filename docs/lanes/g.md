# Lane G — Moonshot vision / ImageNet protocol

| Field | Value |
|-------|-------|
| Branch | `lane/g-vision` |
| Base | `main` @ `5910978` |
| Owns | `bnn/vision/**`, `scripts/train_image*`, `scripts/imagenet*`, `docs/imagenet_protocol.md`, `tests/test_vision*`, this note |
| Tasks | **W4.T05**, **W6.T07 / M6** |

## Status

| ID | Item | Lane status | Notes |
|----|------|-------------|-------|
| W4.T05 | ResNet-BiReal CIFAR reference | `[x]` delivered | `ResNetBiRealCIFAR` / `BiRealBasicBlock`; train via `--include-resnet` |
| W6.T07 | ImageNet folder protocol | `[x]` runner (was stub) | `scripts/imagenet_protocol.py` + contract |
| M6 | ResNet-18 Bi-Real ImageNet protocol runner | `[x]` smoke/proxy | No SOTA gate; no invented goldens |

## Acceptance

- [x] ResNet-BiReal CIFAR builds, forward, 1-step train; optional full train compares to existing `image_cifar` floors (no new golden keys).
- [x] ImageNet dataset contract documented + machine-readable.
- [x] Protocol runner: `contract` / `check` / `proxy` / `smoke`.
- [x] Tests under `tests/test_vision*.py` (no network; synthetic / proxy PNG).
- [x] Thesis lock: no GPU 32× from `sign()`; ImageNet SOTA not a pass gate.

## Residuals / integrator

- Integrator should flip ROADMAP twin: W4.T05 → `[x]`; W6.T07 → `[x]` runner; M6 → delivered smoke; refresh `docs/MOONSHOT_DEFERRALS.md` rows for W4.T05 / M6.
- Optional human: long CIFAR ResNet-BiReal run with `--include-resnet` to record optional result JSON (still **not** a new golden floor unless explicitly regenerating published goldens).
- `bnn/cli.py` train-image flags touched only for `--include-resnet` wiring (Lane E owns bridge subcommands only).

## Commands

```bash
python -m pytest tests/test_vision_smoke.py tests/test_vision_resnet_imagenet.py -q
python scripts/imagenet_protocol.py --mode smoke --out results/imagenet_protocol_smoke.json
bnn train-image --include-resnet --epochs 1 --subset 512 --channels 32 --resnet-width 16
```
