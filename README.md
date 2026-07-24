# Binary Neural Network — Extreme Low-Bit Inference Lab

**Perfected thesis:** Cut **inference** latency/memory on **CPU/edge** by packing
weights (and optionally activations) to **1–1.58 bits** and running **real**
XNOR/popcount (or ternary) kernels — *not* “sign() in PyTorch for 32× on GPU.”

**E2E Roadmap (complete the repo):** [`docs/21_E2E_ROADMAP_COMPLETE_REPO.md`](docs/21_E2E_ROADMAP_COMPLETE_REPO.md) · root [`ROADMAP.md`](ROADMAP.md)

**Python:** 3.11–3.14 (3.12 recommended). **Native kernel:** MSVC x64 (`python -m bnn.kernels.compile_native`) — MinGW 32-bit DLLs fail with WinError 193.

## Quick start

```bat
cd "C:\Users\mrkan\CRAZZY\Binary Neural Network"
pip install -e ".[dev]"
python -m bnn.kernels.compile_native
bnn export-check
bnn validate-native
bnn bench
bnn train --epochs 3 --seed 42
bnn eval-suite
```

### Image + audio (first-class modalities)

```bat
bnn train-image --epochs 8 --subset 30000
bnn train-audio --epochs 5
pytest tests\test_vision_smoke.py tests\test_audio_smoke.py -q
```

- Image tutorial: [`docs/tutorials/04_image_cifar.md`](docs/tutorials/04_image_cifar.md)
- Audio tutorial: [`docs/tutorials/05_audio.md`](docs/tutorials/05_audio.md) (synthetic tones; **not** production ASR — use INT8 Whisper/ORT for real speech)
- Completion: [`docs/28_IMAGE_AUDIO_COMPLETION.md`](docs/28_IMAGE_AUDIO_COMPLETION.md) · [`docs/29_FINAL_COMPLETION.md`](docs/29_FINAL_COMPLETION.md)

## Measured on this machine (CPU)

| Check | Result |
|-------|--------|
| Weight pack compression | **32.00×** |
| Native GEMM correctness | **err = 0** vs ±1 FP32 |
| Speedup 128×2048×2048 | **4.36×** vs NumPy FP32 (compute) |
| Speedup 64×4096×4096 | **3.61×** |
| Speedup 32×8192×8192 | **9.29×** ( **3.44×** vs torch FP32 ) |
| MNIST binary_mlp / ternary | **96.36%** / **97.16%** (FP32 97.67%) |
| Fake `sign`+torch linear | **slower** than FP32 |

## Wrapping existing models

**Short answer:** yes for **INT4/FP8/GGUF** (production); **partial** for binary/ternary
(size easy; accuracy+speed need QAT and real kernels).

```bat
python scripts\wrap_existing_demo.py --mode binary_xnor --hidden 4096 --batch 32
```

Local wrap demo (2 middle Linears): **32×** weight compression, e2e **~1.7×**, layer
gemm_only **~2.6×**, but cosine vs FP **~0.28** without QAT — not a transparent quality wrap.

Deep dive: `docs/12_WRAPPER_AND_EXISTING_MODELS.md`

| Want | Use |
|------|-----|
| Faster GPU HF LLM | torchao / AWQ / GPTQ / bitsandbytes |
| Faster CPU normal LLM | llama.cpp GGUF |
| Faster BitNet LLM | bitnet.cpp |
| Research XNOR wrap | `bnn/wrapper.py` |

## Docs (read in order)

| File | Content |
|------|---------|
| `docs/21_E2E_ROADMAP_COMPLETE_REPO.md` | **Master E2E plan to finish the repo** |
| `docs/22_COMPLETION_REPORT.md` | **D1–D12 completion evidence** |
| `docs/28_IMAGE_AUDIO_COMPLETION.md` | **Image + audio modality gates I1–A2** |
| `docs/29_FINAL_COMPLETION.md` | **Final done criteria + verify** |
| `docs/22_HF_TO_GGUF_GUIDE.md` | HF → GGUF checklist |
| `docs/23`–`25` | bitnet.cpp / GPU INT4-FP8 / one-pager |
| `docs/tutorials/` | MNIST, wrap, CIFAR, **image**, **audio** |
| `docs/api/README.md` | API stub |
| `docs/00_DIMENSION_MAP.md` | **Completeness checklist (40/40 Covered)** |
| `docs/19_GAP_CLOSURE_REPORT.md` | **Gap closure — 0 material OPEN** |
| `docs/09_GAP_REGISTER.md` | Closed / proxy / accepted non-goals |
| `docs/05_PERFECTED_CONCEPT.md` | Sharpened product idea |
| `docs/18_DECISION_TREE_AND_COMPLETE_ROADMAP.md` | Decision tree (strategy) |
| `docs/11_DEEP_RESEARCH_REPORT.md` | Executive synthesis |
| `docs/06_CALCULATED_SPEEDUP_MODEL.md` | Formulas + measured speedups |
| `docs/12_WRAPPER_AND_EXISTING_MODELS.md` | Wrapping HF/PyTorch models |
| `docs/13_TRAINING_QAT_DISTILL.md` | STE, QAT, distill, scaling |
| `docs/14_HARDWARE_AND_ENERGY.md` | CPU/GPU/NPU/FPGA/energy |
| `docs/15_MODEL_CLASSES_AND_DEPLOYMENT.md` | Vision/LLM/diffusion/speech |
| `docs/16_ECOSYSTEM_AND_TOOLING.md` | Tooling matrix |
| `docs/17_EVALUATION_ROBUSTNESS_ECONOMICS.md` | Eval protocol, risk, $/token |
| `docs/20_NPU_VENDOR_CLOSURE.md` | Vendor NPU: INT8-first, no stock 1-bit |
| `docs/01`–`04`, `07`–`08`, `10` | Principles, SOTA, failures, ADR; `10` → points to `21` |

```bat
python scripts\wrap_existing_demo.py --mode binary_xnor --hidden 4096 --batch 32
python scripts\energy_bound_measured.py
python scripts\train_cifar10_proxy.py --epochs 5 --train-subset 20000
python scripts\robustness_fgsm.py --epochs-quick 2
python scripts\hybrid_ffn_wrap_demo.py
python scripts\ternary_pack_demo.py
```


## Code layout

```
bnn/           STE, layers, models, wrapper, MNIST + CIFAR loaders
bnn/vision/    CIFAR Bi-Real CNN, tiny binary ViT, ImageNet folder stub
bnn/audio/     STFT features, synthetic tones, FP/binary audio CNN
bnn/kernels/   packed.py, ternary_pack.py, binary_gemm.c (MSVC DLL)
scripts/       train, train_image, train_audio, benchmark, wrap, eval suite
results/       JSON/MD (image_cifar, audio_synth, kernels, MNIST, …)
docs/          00–24 research + completion; tutorials 01–05
```

## What this is / is not

| Is | Is not |
|----|--------|
| Honest CPU proof of binary speedups | A claim of 32× e2e everywhere |
| Trainable BNN + BitLinear pedagogy | Full BitNet LLM pretrain |
| Guidance: GPU → use FP8/INT4 | A cuDNN replacement |

## CPU vs GPU (short)

- **CPU/edge:** binary/ternary packed kernels win (this repo + bitnet.cpp + Larq CE).
- **Commodity NVIDIA GPU:** use torchao/vLLM FP8–INT4; fake BNNs lose to Tensor Cores.
