#!/usr/bin/env python3
"""GPU production bridge recipe (torchao INT4) — docs-as-code shim.

Does NOT pull torchao by default. Prints a concrete recipe and optionally
probes whether torchao is installed. Classic BNN XNOR is not the GPU path.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


RECIPE = {
    "lane": "gpu-server",
    "thesis": "Commodity NVIDIA → INT4/FP8 (torchao / AWQ / vLLM). Not classic BNN 32x CUDA.",
    "torchao_weight_only_int4": [
        "pip install torchao  # on a CUDA machine",
        "from torchao.quantization import quantize_, int4_weight_only",
        "quantize_(model, int4_weight_only())",
        "# Serve with vLLM / TensorRT-LLM for throughput",
    ],
    "awq": [
        "pip install autoawq  # CUDA",
        "# Follow AutoAWQ docs: calibrate → save → load with vLLM AWQ",
    ],
    "this_repo": [
        "bnn recommend --goal gpu-server",
        "docs/24_GPU_INT4_FP8_LANE.md",
        "Use bnn wrap only for CPU packed binary/ternary pedagogy + edge",
    ],
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--probe", action="store_true", help="Check if torchao importable")
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "bridge_gpu_torchao.json",
    )
    args = p.parse_args()
    payload = dict(RECIPE)
    if args.probe:
        try:
            import torchao  # noqa: F401

            payload["torchao_installed"] = True
            payload["torchao_version"] = getattr(torchao, "__version__", "unknown")
        except ImportError:
            payload["torchao_installed"] = False
            payload["note"] = "Install torchao on CUDA hosts; skip on CPU-only lab machines"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
