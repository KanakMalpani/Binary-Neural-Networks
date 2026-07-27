#!/usr/bin/env python3
"""Recommend stack from the perfected decision tree (docs/18)."""

from __future__ import annotations

import argparse
import json

RECS = {
    "gpu-server": {
        "use": "BF16/FP8 train → FP8 or AWQ-INT4 serve (vLLM / SGLang / TensorRT / torchao)",
        "avoid": "Classic CUDA BNN / sign()-in-PyTorch for speed",
        "docs": ["docs/24_GPU_INT4_FP8_LANE.md", "docs/18_DECISION_TREE_AND_COMPLETE_ROADMAP.md"],
    },
    "cpu-llm": {
        "use": "BitNet checkpoint → bitnet.cpp; else GGUF Q4_K_M via llama.cpp",
        "avoid": "PTQ absmean ternary without distill",
        "docs": ["docs/23_BITNET_CPP_BRIDGE.md", "docs/22_HF_TO_GGUF_GUIDE.md"],
    },
    "edge-vision": {
        "use": "Retrain Bi-Real/ReActNet → Larq CE (ARM) or FINN (FPGA); else INT8 TFLite/OpenVINO",
        "avoid": "Expecting stock phone NPU to run native 1-bit XNOR",
        "docs": ["docs/20_NPU_VENDOR_CLOSURE.md", "docs/15_MODEL_CLASSES_AND_DEPLOYMENT.md"],
    },
    "npu-phone": {
        "use": "Stock QNN/CoreML/Ethos → INT8 (or INT4 weights). Custom Hexagon only if budgeting kernels",
        "avoid": "Assuming vendor 1-bit BNN support",
        "docs": ["docs/20_NPU_VENDOR_CLOSURE.md"],
    },
    "research-xnor": {
        "use": "This repo: STE train + MSVC packed GEMM + bnn wrap",
        "avoid": "Marketing 32× wall-clock from bit packing alone",
        "docs": ["README.md", "docs/05_PERFECTED_CONCEPT.md", "docs/21_E2E_ROADMAP_COMPLETE_REPO.md"],
    },
    "diffusion": {
        "use": "INT8/FP8 weight PTQ; keep UNet fidelity",
        "avoid": "Full W+A BNN for high-fidelity generative",
        "docs": ["docs/15_MODEL_CLASSES_AND_DEPLOYMENT.md"],
    },
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--goal", required=True, choices=sorted(RECS))
    args = p.parse_args()
    print(json.dumps({"goal": args.goal, **RECS[args.goal]}, indent=2))


if __name__ == "__main__":
    main()
