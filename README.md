# Binary Neural Networks — Extreme Low-Bit Inference Lab

Cut **inference** latency and memory on **CPU / edge** by packing weights (and
optionally activations) to **1–1.58 bits** and running **real** XNOR/popcount
kernels — *not* `sign()` in PyTorch pretending to be 32× on GPU.

| | |
|--|--|
| **Reproduce** | [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) · `bnn repro` |
| **AI agents** | [`AGENTS.md`](AGENTS.md) |
| **Docs index** | [`docs/README.md`](docs/README.md) |
| **One-pager** | [`docs/25_ONEPAGER.md`](docs/25_ONEPAGER.md) |

**Python:** 3.11+ (3.12 recommended). **Native kernel:** Windows MSVC x64 only —
MinGW 32-bit DLLs fail with WinError 193; Linux/macOS use the NumPy fallback
(correctness preserved).

---

## Reproduce results (5 commands)

```bat
git clone https://github.com/KanakMalpani/Binary-Neural-Networks.git
cd Binary-Neural-Networks
pip install -e ".[dev]" -c constraints.txt
python -m bnn.kernels.compile_native
bnn repro
```

Expect `REPRO: PASS` (exit 0). Fast verify uses committed `results/*.json` +
`tests/golden_floors.json` — same **conclusions**, not bit-identical floats.
Optional longer regen: `bnn repro --mode full`.

---

## Quick start

```bat
pip install -e ".[dev]" -c constraints.txt
python -m bnn.kernels.compile_native
bnn export-check
bnn validate-native
bnn bench
bnn train --epochs 3 --seed 42
bnn eval-suite
```

### Vision + audio

```bat
bnn train-image --epochs 8 --subset 30000 --seed 0 --approx-sign
bnn train-audio --epochs 5 --seed 0
pytest tests/test_vision_smoke.py tests/test_audio_smoke.py -q
```

- Image: [`docs/tutorials/04_image_cifar.md`](docs/tutorials/04_image_cifar.md)
- Audio: [`docs/tutorials/05_audio.md`](docs/tutorials/05_audio.md) — synthetic tones;
  **not** production ASR (use INT8 Whisper/ORT for real speech)

---

## Measured on this machine (CPU) — committed goldens

| Check | Result | Kind |
|-------|--------|------|
| Weight pack compression | **32.00×** | Exact |
| Native GEMM vs ±1 FP32 | **err = 0** | Exact (when DLL loaded) |
| Speedup 64×4096×4096 vs NumPy FP32 | **~3.6–3.7×** | Wall-clock (machine-dependent) |
| MNIST binary_mlp / ternary | **96.36%** / **97.16%** (FP 97.67%) | Tolerance-gated |
| CIFAR Bi-Real vs FP CNN | **61.14%** vs **71.14%** (10 pp) | Tolerance-gated |
| Audio binary vs FP (synthetic) | **96.0%** vs **94.5%** | Tolerance-gated |
| Fake `sign`+torch Linear | Often **slower** than FP32 | Anti-pattern |

Do **not** advertise theoretical ~64× word reduction as end-to-end latency.

---

## Decision tree (where to use what)

```
GPU server quality?     → FP8 / AWQ-INT4 + vLLM  (NOT classic BNN)
CPU local LLM?          → BitNet? bitnet.cpp : GGUF Q4_K_M
Edge vision retrain?    → Bi-Real + this repo / LCE/FINN ; else INT8
Phone NPU stock SDK?    → INT8/INT4  (no stock 1-bit)
Research XNOR kernels?  → this repo (`bnn`)
Diffusion fidelity?     → INT8/FP8 PTQ ; avoid full BNN
```

```bat
bnn recommend --goal edge-vision
```

Full tree: [`docs/18_DECISION_TREE_AND_COMPLETE_ROADMAP.md`](docs/18_DECISION_TREE_AND_COMPLETE_ROADMAP.md)

---

## Wrapping existing models

**Short answer:** yes for **INT4/FP8/GGUF** (production); **partial** for
binary/ternary (size easy; accuracy + speed need QAT and real kernels).

```bat
bnn wrap --mode binary_xnor --hidden 4096 --batch 32
```

Local wrap demo: **32×** weight compression on replaced layers, modest e2e
speedup, cosine much less than 1 without QAT — expected, not a transparent
quality wrap.

Deep dive: [`docs/12_WRAPPER_AND_EXISTING_MODELS.md`](docs/12_WRAPPER_AND_EXISTING_MODELS.md)

---

## Package layout

```
bnn/           STE, layers, models, wrapper, export, determinism
bnn/vision/    CIFAR Bi-Real CNN, tiny binary ViT
bnn/audio/     STFT features, synthetic tones, FP/binary CNN
bnn/kernels/   packed XNOR GEMM (+ optional MSVC DLL)
scripts/       train / bench / wrap / repro_all
results/       committed golden JSON + SUMMARY.md
tests/         pytest + golden_floors.json
docs/          research, tutorials, completion reports
```

Public API: `import bnn` — see [`docs/api/README.md`](docs/api/README.md).
CLI: `bnn --help` · `bnn --version` · `python -m bnn repro`

---

## What this is / is not

| Is | Is not |
|----|--------|
| Honest CPU proof of binary speedups | A claim of 32× e2e everywhere |
| Trainable BNN + BitLinear pedagogy | Full BitNet LLM pretrain |
| Guidance: GPU → use FP8/INT4 | A cuDNN replacement |

---

## Contributing & quality

- [`CONTRIBUTING.md`](CONTRIBUTING.md) · [`CHANGELOG.md`](CHANGELOG.md)
- [`docs/31_QUALITY_UPGRADE.md`](docs/31_QUALITY_UPGRADE.md)
- CI: Windows + Linux (`.github/workflows/ci.yml`) — pytest, export-check, repro gates
