# Documentation index

Start here if you are new to the repo.

## Must-read (operators & AIs)

| Doc | Why |
|-----|-----|
| [`GUIDE_E2E.md`](GUIDE_E2E.md) | **Primary User Guide** — zero → optimiser results (follow this) |
| [`../REPRODUCIBILITY.md`](../REPRODUCIBILITY.md) | Exact repro commands, goldens, troubleshooting |
| [`../AGENTS.md`](../AGENTS.md) | Ordered steps for coding agents |
| [`25_ONEPAGER.md`](25_ONEPAGER.md) | Executive decision tree (honest) |
| [`api/README.md`](api/README.md) | Public Python API |
| [`31_QUALITY_UPGRADE.md`](31_QUALITY_UPGRADE.md) | Latest quality leap changelog |
| [`34_COMPUTE_SPEEDUP.md`](34_COMPUTE_SPEEDUP.md) | OpenMP / pack / ternary speedups + thread curve |
| [`35_BINARY_MATH_EFFECTIVENESS.md`](35_BINARY_MATH_EFFECTIVENESS.md) | XNOR↔dot proofs, STE math, when binary loses |
| [`36_ENCODER_DECODER_AND_NEXT.md`](36_ENCODER_DECODER_AND_NEXT.md) | Encoder/Decoder + `.bnnpack` + bridges |

## Tutorials

Ordered path (also nested inside [`GUIDE_E2E.md`](GUIDE_E2E.md)):

| # | Tutorial |
|---|----------|
| 01 | [`tutorials/01_mnist_binary.md`](tutorials/01_mnist_binary.md) |
| 02 | [`tutorials/02_wrap_linear.md`](tutorials/02_wrap_linear.md) — prefer `bnn optimise` |
| 03 | [`tutorials/03_cifar_bireal.md`](tutorials/03_cifar_bireal.md) |
| 04 | [`tutorials/04_image_cifar.md`](tutorials/04_image_cifar.md) |
| 05 | [`tutorials/05_audio.md`](tutorials/05_audio.md) |
| 06 | [`tutorials/06_encoder_decoder.md`](tutorials/06_encoder_decoder.md) |
| 07 | [`tutorials/07_OPTIMISER_QUICKSTART.md`](tutorials/07_OPTIMISER_QUICKSTART.md) |
| 08 | [`tutorials/08_HF_OPTIMISER.md`](tutorials/08_HF_OPTIMISER.md) |

## Latest lane

| Doc | Why |
|-----|-----|
| [`GUIDE_E2E.md`](GUIDE_E2E.md) | Master end-to-end user guide |
| [`39_GUIDE_E2E_COMPLETION.md`](39_GUIDE_E2E_COMPLETION.md) | E2E guide smoke confirmation |
| [`40_ROADMAP_E2E_SESSION.md`](40_ROADMAP_E2E_SESSION.md) | v0.3.0 Phase C+D completion report |
| [`LAUNCH_CHECKLIST.md`](LAUNCH_CHECKLIST.md) | Public launch checklist |
| [`FAIR_EVAL_PROTOCOL.md`](FAIR_EVAL_PROTOCOL.md) | Dual-metric fair eval |
| [`MOONSHOT_DEFERRALS.md`](MOONSHOT_DEFERRALS.md) | Explicit v1.0 leftovers |
| [`37_WORLD_CLASS_BNN_OPTIMISER_ROADMAP.md`](37_WORLD_CLASS_BNN_OPTIMISER_ROADMAP.md) | Canonical world-class plan (twin of root `ROADMAP.md`) |
| [`38_ROADMAP_EXECUTION_LOG.md`](38_ROADMAP_EXECUTION_LOG.md) | Session execution log |
| [`SEMVER_AND_DEPRECATION.md`](SEMVER_AND_DEPRECATION.md) | Public API semver policy |
| [`adr/README.md`](adr/README.md) | Architecture Decision Records |
| [`DATASET_CARDS.md`](DATASET_CARDS.md) | MNIST / CIFAR / synth audio cards |
| [`COMPATIBILITY_MATRIX.md`](COMPATIBILITY_MATRIX.md) | OS × Python × torch matrix |
| [`36_ENCODER_DECODER_AND_NEXT.md`](36_ENCODER_DECODER_AND_NEXT.md) | Encoder/Decoder, `.bnnpack` codec, wrap-transformer, profile, bridges |

## Research & architecture (read order)

1. [`01_FIRST_PRINCIPLES.md`](01_FIRST_PRINCIPLES.md)
2. [`05_PERFECTED_CONCEPT.md`](05_PERFECTED_CONCEPT.md)
3. [`04_ARCHITECTURE.md`](04_ARCHITECTURE.md)
4. [`06_CALCULATED_SPEEDUP_MODEL.md`](06_CALCULATED_SPEEDUP_MODEL.md)
5. [`08_ADR.md`](08_ADR.md) · [`09_GAP_REGISTER.md`](09_GAP_REGISTER.md)

## Completion / roadmap

| Doc | Content |
|-----|---------|
| [`../ROADMAP.md`](../ROADMAP.md) | **Canonical** world-class BNN optimiser plan (follow when lost) |
| [`37_WORLD_CLASS_BNN_OPTIMISER_ROADMAP.md`](37_WORLD_CLASS_BNN_OPTIMISER_ROADMAP.md) | Identical twin of root `ROADMAP.md` |
| [`21_E2E_ROADMAP_COMPLETE_REPO.md`](21_E2E_ROADMAP_COMPLETE_REPO.md) | Historical lab COMPLETE (D1–D12); superseded for forward work |
| [`22_COMPLETION_REPORT.md`](22_COMPLETION_REPORT.md) | D1–D12 evidence |
| [`28_IMAGE_AUDIO_COMPLETION.md`](28_IMAGE_AUDIO_COMPLETION.md) | Vision + audio gates |
| [`29_FINAL_COMPLETION.md`](29_FINAL_COMPLETION.md) | Final done criteria |
| [`30_REPRO_FOR_OTHER_AIS.md`](30_REPRO_FOR_OTHER_AIS.md) | Third-party repro note |
| [`10_ROADMAP.md`](10_ROADMAP.md) | Thin pointer → root `ROADMAP.md` |

## Ecosystem bridges

| Doc | Topic |
|-----|-------|
| [`12_WRAPPER_AND_EXISTING_MODELS.md`](12_WRAPPER_AND_EXISTING_MODELS.md) | Wrapping HF/PyTorch |
| [`22_HF_TO_GGUF_GUIDE.md`](22_HF_TO_GGUF_GUIDE.md) | HF → GGUF |
| [`23_BITNET_CPP_BRIDGE.md`](23_BITNET_CPP_BRIDGE.md) | bitnet.cpp |
| [`24_GPU_INT4_FP8_LANE.md`](24_GPU_INT4_FP8_LANE.md) | GPU INT4/FP8 |
| [`20_NPU_VENDOR_CLOSURE.md`](20_NPU_VENDOR_CLOSURE.md) | Phone NPUs |
| [`../scripts/bridges/`](../scripts/bridges/) | Concrete torchao / llama.cpp / bitnet recipes (JSON) |

## Thesis lock (never reopen as a “win”)

Packed CPU/edge kernels are the product story. Commodity GPU → INT4/FP8.
Do not claim 32× e2e from theoretical word reduction or fake `sign()` Linear.
