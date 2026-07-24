# Image + Audio modality completion

**Date:** 2026-07-24  
**Thesis lock:** CPU packed low-bit inference; honest accuracy; no fake GPU 32× BNN.

## Status

| Gate | Status | Evidence |
|------|--------|----------|
| **I1** Image train/eval CLI + results + tutorial | **PASS** | `bnn train-image` → `results/image_cifar.*`; `docs/tutorials/04_image_cifar.md` |
| **I2** Image pytest smoke | **PASS** | `tests/test_vision_smoke.py` |
| **A1** Audio train/eval + synthetic fallback + tutorial | **PASS** | `bnn train-audio` → `results/audio_synth.*`; `docs/tutorials/05_audio.md` |
| **A2** Audio pytest smoke | **PASS** | `tests/test_audio_smoke.py` |

## Image lane

- Models: `bnn/vision/models.py` — FP CNN, Bi-Real CNN, tiny binary-ViT sketch
- Train: `scripts/train_image.py` / CLI `bnn train-image`
- Measured (30k subset, 8 ep, ApproxSign): FP **71.14%** vs binary **61.14%** (`results/image_cifar.*`)
- ApproxSign: `--approx-sign` → `bnn.ste.set_approx_sign`
- Packed Conv: `wrap_conv_modules` / `PackedBinaryConv2d` — **size win** (pad-limited <32× on tiny K); forward = dequant+FP (honest)
- ImageNet full train: ADR non-goal; stub `bnn.vision.check_imagenet_folder`

## Audio lane

- Features: numpy STFT + mel-like pool (`bnn/audio/features.py`)
- Data: synthetic tones always (CI-safe); optional NPZ cache
- Models: FP / binary CNN (+ MLP) in `bnn/audio/models.py`
- Measured (800/200, 5 ep): FP **94.5%** vs binary **96.0%** on easy tones (`results/audio_synth.*`)
- **Not production ASR** — recommend INT8 Whisper/ORT; demo proves packing/QAT pattern on audio features

## Roadmap leftovers closed this pass

- P2.T5 ApproxSign flag
- P3.T7 BinaryConv wrap (size-honest)
- P2.T4 Longer CIFAR via `train-image` defaults (30k / 8 ep)
- P7.T2 ImageNet folder stub (no full train)

OpenMP/AVX remain **ACCEPTED-NON-GOAL**.

See also: `docs/29_FINAL_COMPLETION.md`.
