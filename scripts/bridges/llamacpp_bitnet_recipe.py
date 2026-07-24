#!/usr/bin/env python3
"""CPU production bridge: llama.cpp GGUF Q4 + bitnet.cpp pointers (no heavy deps).

Prints concrete build/convert steps. This lab's packed kernels are for research
XNOR / edge; production CPU LLMs use GGUF or BitNet.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

RECIPE = {
    "lane": "cpu-llm",
    "thesis": "CPU chat LLMs → GGUF Q4_K_M (llama.cpp) or BitNet → bitnet.cpp. Not sign()+torch.",
    "llama_cpp_gguf": [
        "git clone https://github.com/ggerganov/llama.cpp",
        "cmake -B build -DGGML_NATIVE=ON && cmake --build build --config Release",
        "# Convert HF → GGUF (upstream convert scripts) then:",
        "./llama-quantize model-f16.gguf model-Q4_K_M.gguf Q4_K_M",
        "./llama-cli -m model-Q4_K_M.gguf -p 'Hello' -n 64",
    ],
    "bitnet_cpp": [
        "git clone https://github.com/microsoft/BitNet",
        "# Follow upstream setup_env / build for your OS",
        "# Use native BitNet checkpoints — do not absmean-PTQ a chat Llama and ship",
        "docs/23_BITNET_CPP_BRIDGE.md",
    ],
    "this_repo_shims": [
        "bnn recommend --goal cpu-llm",
        "bnn encode / bnn decode → .bnnpack for lab packed Linear artifacts",
        "TernaryLinear + ternary_pack = pedagogy/size; speed for BitNet = bitnet.cpp",
    ],
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "bridge_cpu_llamacpp_bitnet.json",
    )
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(RECIPE, indent=2), encoding="utf-8")
    print(json.dumps(RECIPE, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
