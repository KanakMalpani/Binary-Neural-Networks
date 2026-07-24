# Documentation index

Start here if you are new to the repo.

## Must-read (operators & AIs)

| Doc | Why |
|-----|-----|
| [`../REPRODUCIBILITY.md`](../REPRODUCIBILITY.md) | Exact repro commands, goldens, troubleshooting |
| [`../AGENTS.md`](../AGENTS.md) | Ordered steps for coding agents |
| [`25_ONEPAGER.md`](25_ONEPAGER.md) | Executive decision tree (honest) |
| [`api/README.md`](api/README.md) | Public Python API |
| [`31_QUALITY_UPGRADE.md`](31_QUALITY_UPGRADE.md) | Latest quality leap changelog |
| [`34_COMPUTE_SPEEDUP.md`](34_COMPUTE_SPEEDUP.md) | OpenMP / pack / ternary speedups + thread curve |

## Tutorials

| # | Tutorial |
|---|----------|
| 01 | [`tutorials/01_mnist_binary.md`](tutorials/01_mnist_binary.md) |
| 02 | [`tutorials/02_wrap_linear.md`](tutorials/02_wrap_linear.md) |
| 03 | [`tutorials/03_cifar_bireal.md`](tutorials/03_cifar_bireal.md) |
| 04 | [`tutorials/04_image_cifar.md`](tutorials/04_image_cifar.md) |
| 05 | [`tutorials/05_audio.md`](tutorials/05_audio.md) |

## Research & architecture (read order)

1. [`01_FIRST_PRINCIPLES.md`](01_FIRST_PRINCIPLES.md)
2. [`05_PERFECTED_CONCEPT.md`](05_PERFECTED_CONCEPT.md)
3. [`04_ARCHITECTURE.md`](04_ARCHITECTURE.md)
4. [`06_CALCULATED_SPEEDUP_MODEL.md`](06_CALCULATED_SPEEDUP_MODEL.md)
5. [`08_ADR.md`](08_ADR.md) · [`09_GAP_REGISTER.md`](09_GAP_REGISTER.md)

## Completion / roadmap

| Doc | Content |
|-----|---------|
| [`21_E2E_ROADMAP_COMPLETE_REPO.md`](21_E2E_ROADMAP_COMPLETE_REPO.md) | Master E2E plan |
| [`22_COMPLETION_REPORT.md`](22_COMPLETION_REPORT.md) | D1–D12 evidence |
| [`28_IMAGE_AUDIO_COMPLETION.md`](28_IMAGE_AUDIO_COMPLETION.md) | Vision + audio gates |
| [`29_FINAL_COMPLETION.md`](29_FINAL_COMPLETION.md) | Final done criteria |
| [`30_REPRO_FOR_OTHER_AIS.md`](30_REPRO_FOR_OTHER_AIS.md) | Third-party repro note |
| [`../ROADMAP.md`](../ROADMAP.md) | Root roadmap pointer |

## Ecosystem bridges

| Doc | Topic |
|-----|-------|
| [`12_WRAPPER_AND_EXISTING_MODELS.md`](12_WRAPPER_AND_EXISTING_MODELS.md) | Wrapping HF/PyTorch |
| [`22_HF_TO_GGUF_GUIDE.md`](22_HF_TO_GGUF_GUIDE.md) | HF → GGUF |
| [`23_BITNET_CPP_BRIDGE.md`](23_BITNET_CPP_BRIDGE.md) | bitnet.cpp |
| [`24_GPU_INT4_FP8_LANE.md`](24_GPU_INT4_FP8_LANE.md) | GPU INT4/FP8 |
| [`20_NPU_VENDOR_CLOSURE.md`](20_NPU_VENDOR_CLOSURE.md) | Phone NPUs |

## Thesis lock (never reopen as a “win”)

Packed CPU/edge kernels are the product story. Commodity GPU → INT4/FP8.
Do not claim 32× e2e from theoretical word reduction or fake `sign()` Linear.
