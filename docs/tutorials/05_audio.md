# Tutorial 05 — Audio lane (synthetic spectrograms)

**Master guide:** [`../GUIDE_E2E.md`](../GUIDE_E2E.md) · **Prev:** [04](04_image_cifar.md) · **Next:** [06](06_encoder_decoder.md)

**Goal:** Prove STE + binary CNN pattern on **audio features**, offline-friendly.

## Honest scope

Classic BNNs are **not** production ASR/TTS. For real speech use **INT8 Whisper / ONNX Runtime / vendor NPU INT8**. This demo shows packing/QAT *patterns* on spectrogram-like inputs.

## Quick run (always works offline)

```bat
bnn train-audio --epochs 5 --n-train 800 --n-test 200 --seed 0
```

Or:

```bat
python scripts/train_audio.py --epochs 5 --seed 0
```

Committed golden: `results/audio_synth.json` (verify: `bnn repro`).

Outputs: `results/audio_synth.json` (+ `.md`).

## Pipeline

1. Synthetic musical tones (8 pitch classes) — reproducible, no download required.
2. NumPy STFT → cheap mel-like pool (`bnn/audio/features.py`).
3. FP CNN vs Bi-Real-style binary CNN (`bnn/audio/models.py`).

Optional NPZ cache under `data/audio_cache/` after first run.

## Smoke test

```bat
pytest tests\test_audio_smoke.py -q
```

No network. Safe for CI.
