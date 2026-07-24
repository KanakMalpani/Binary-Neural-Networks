# Ecosystem & Tooling Matrix

## 27–31. Complete wrapper / export matrix

| Tool | Role | Drop-in HF? | Retrain? | Hardware | Typical win | Acc risk | “Wrapper”? |
|------|------|-------------|----------|----------|-------------|----------|------------|
| bitsandbytes | INT8/NF4 load | Yes | No (QLoRA yes) | NVIDIA | Memory 2–4×; speed often flat | Low–med | Yes |
| AWQ | INT4 calib | Yes | Calib | GPU+vLLM | ~1.3–1.9×, ~50% VRAM | Low | Yes |
| GPTQ / GPTQModel | INT4 calib | Yes | Calib | GPU | Similar AWQ | Low | Yes |
| torchao | INT4/FP8/QAT | Yes | Optional QAT | GPU/CPU/XPU | Llama8B INT4 ~1.89× | Low | Yes |
| Quanto / Optimum | HF PTQ helpers | Yes | No | Mixed | Convenience | Med | Yes |
| peft + bnb | QLoRA | Yes | FT | GPU | Train large on small VRAM | — | Yes |
| accelerate | device_map | Yes | — | Multi-GPU | Fit | — | Infra |
| llama.cpp / GGUF | CPU/GPU infer | Via convert | No | CPU++ | Best common local LLM | Low–med | Yes |
| bitnet.cpp | Ternary LLM | BitNet ckpt | Native/distill | CPU/GPU | 1.4–6× CPU | Low if native | Yes |
| vLLM / SGLang | Serve | Yes | No | GPU | Throughput | Low | Yes |
| Larq + LCE | BNN train+ARM | Keras | QAT | Mobile CPU | 8.5–18.5× | Med | Train+deploy |
| Brevitas | PyTorch QAT | Manual | QAT | →FINN/ORT | Flexible bits | Med | Train |
| FINN | FPGA deploy | Via Brevitas | QAT | Xilinx | Ms–µs latency vision | Task-dep | Export |
| TensorRT | NVIDIA engine | Export | No | NVIDIA | Peak serve | Low | Export |
| OpenVINO | Intel | Export | No | CPU/iGPU | INT8 | Low | Export |
| ORT | Cross | Export | No | Many | INT8 QDQ | Low | Export |
| ExecuTorch | On-device | torchao | No/QAT | Mobile | INT4/8 | Low | Export |
| TFLite | Android | Convert | No | Mobile | INT8 | Low | Export |
| CoreML | Apple | Convert | No | Apple | FP16/INT8 | Low | Export |
| TVM / IREE / MLIR | Compilers | Custom | — | Many | Research | — | Lowering |
| **This repo wrapper** | XNOR/ternary Linear | Manual | Recommended | CPU | Size 16–32×; speed if wide | High PTQ | Research |

## HF / peft / accelerate patterns

```text
from_pretrained(..., quantization_config=AwqConfig|BitsAndBytesConfig|TorchAoConfig)
+ device_map="auto"  (accelerate)
+ PeftModel for QLoRA on NF4 base
```

BitNet in Transformers: **QAT/pretrain**, not on-the-fly PTQ (`quantization/bitnet` docs).

## Licensing / dependency notes (detail in 17)

- Larq: Apache-2.0 (archived upstream)
- bitnet.cpp / llama.cpp: MIT-class
- AWQ/GPTQ weights: follow base model license
- This repo code: research scaffold (add LICENSE if redistributing)

## Residual

Vendor NPU SDKs change fast — verify annually.
