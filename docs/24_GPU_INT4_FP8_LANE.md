# GPU lane: INT4 / FP8 (not classic BNN)

On commodity **NVIDIA** servers, use industry low-bit stacks — **not** CUDA XNOR BNNs.

## Recommended

| Goal | Tooling |
|------|---------|
| Weight-only INT4 | AWQ / GPTQ / bitsandbytes / torchao |
| FP8 train/serve | TransformerEngine / torchao FP8 + vLLM / TensorRT-LLM |
| Throughput serve | vLLM / SGLang / TensorRT |

## Explicit non-goal

Classic binary networks beating Tensor Core FP8/INT8 on datacenter GPUs. See arXiv:1911.04477 and ADR `docs/08`.

## This machine

Often **CPU-only** PyTorch — run GPU recipes elsewhere; keep docs honest.

## Concrete recipe script

```bat
python scripts\bridges\torchao_int4_recipe.py --probe
```

Writes `results/bridge_gpu_torchao.json` (torchao INT4 + AWQ pointers). Does not
install CUDA deps by default.

`bnn recommend --goal gpu-server`
