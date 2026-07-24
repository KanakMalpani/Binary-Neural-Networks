# Model Classes & Deployment Surfaces

## 20. Vision — CNN / ViT / detection

| Task | Binary readiness | Practical path |
|------|------------------|----------------|
| Classification (CIFAR/ImageNet) | High with Bi-Real/ReActNet | QAT + LCE/FINN/this recipe |
| Detection / segmentation | Medium | Often hybrid: binary backbone, FP heads |
| ViT | Medium–hard | Patch embed & attn sensitive; BitLinear MLP first |

**Protocol (even if not fully trained here):** CIFAR-10 ReActNet-lite 200 epochs; ImageNet
ResNet-18 Bi-Real with published recipe; report top-1 + MACs + packed size + CPU ms.

## 21. LLMs / local chat

| Path | Use |
|------|-----|
| Normal arch (Llama/Qwen/Phi) | GGUF Q4_K / torchao INT4 / AWQ+vLLM |
| BitNet-native | bitnet.cpp / HF BitNet-b1.58-* |
| Convert FP→1.58 | BitDistill / gradual λ FT — **not** naive PTQ |

Local chat benefit: RAM fit + tokens/s; batch-1 decode is bandwidth-bound.

## 22. Diffusion / generative

BNNs struggle: diffusion needs fine continuous noise prediction; extreme quantization of UNet/DiT
weights often hurts FID badly. **Practical:** FP8/INT8 weight PTQ of UNet; keep CFG/attention
sensitive parts higher bit. Full 1-bit diffusion = research, not product default.

## 23. Speech — ASR / TTS

Sparse but growing: MSc work on **Binary Conformer** for edge ASR (accuracy–latency tradeoffs);
mainstream ASR (Whisper-class) uses FP16/INT8. TTS/codecs increasingly use **discrete tokens +
diffusion**, not BNNs. **Recommendation:** INT8 ORT/OpenVINO for ASR edge; BNN only with
custom QAT budget.

**In-repo audio demo (not ASR):** `bnn train-audio` classifies synthetic tone spectrograms with
FP vs Bi-Real-style binary CNN (`bnn/audio/`). Proves STE + packing pattern on audio features.
Production ASR → INT8 Whisper/ORT. Tutorial: `docs/tutorials/05_audio.md`.

**In-repo image demo:** `bnn train-image` CIFAR-10 Bi-Real (+ optional tiny ViT). Full ImageNet
train remains ADR non-goal (`docs/imagenet_protocol.md`).

## 24. Multimodal / embeddings

- **BitEmbed** (2026): ternary BitNet-style embedders — competitive retrieval with storage wins
- CLIP-like: prefer INT8/INT4 PTQ; binary text/vision towers need QAT
- Multimodal LLMs: same as LLM path (FFN quantize first)

## 25. On-device / browser / WASM

| Target | Stack |
|--------|-------|
| Android | TFLite / LCE / ExecuTorch |
| iOS | CoreML / ExecuTorch |
| Browser | ONNX Runtime Web / WASM SIMD (INT8 easier than 1-bit) |
| Microcontrollers | CMSIS-NN INT8; BNN if FINN/MCU-class |

1-bit in WASM is rare; prefer INT8 SIMD.

## 26. Training speedup vs inference-only

| Phase | Binary helps? |
|-------|---------------|
| Training throughput | **Almost never** (STE + FP latents + longer schedules) |
| Inference latency/energy | **Yes** with kernels |
| Memory during train | Partial (activations if quantized) |

**Thesis unchanged:** product = inference.

## Cross-class decision (snippet)

```
Vision edge + can retrain → Bi-Real/ReActNet + LCE/FINN
LLM GPU serve → FP8/INT4 + vLLM
LLM CPU local → GGUF or bitnet.cpp
Diffusion → INT8/FP8 PTQ, not BNN
Speech edge → INT8 first; Binary Conformer research; in-repo: `bnn train-audio`
Embeddings → BitEmbed / INT8
Browser → INT8 ORT-Web
```
