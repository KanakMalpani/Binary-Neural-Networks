#!/usr/bin/env python3
"""CPU production bridge: llama.cpp GGUF Q4 + bitnet.cpp (pinned recipe, no heavy deps).

Prints concrete build/convert steps with **pinned** upstream SHAs/tags from
``llamacpp_bitnet_pins.json``. This lab's packed kernels are for research
XNOR / edge; production CPU LLMs use GGUF or BitNet.

Policy: do **not** vendor microsoft/BitNet as a git submodule here — clone
out-of-tree at the pinned ref. See ``docs/23_BITNET_CPP_BRIDGE.md``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PINS_PATH = Path(__file__).resolve().with_name("llamacpp_bitnet_pins.json")
SCHEMA = "bnn.llamacpp_bitnet_pins.v1"


def load_pins(path: Path = PINS_PATH) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != SCHEMA:
        raise ValueError(f"unexpected pins schema: {data.get('schema')!r} (want {SCHEMA})")
    for key in ("bitnet_cpp", "llama_cpp_gguf", "models", "policy"):
        if key not in data:
            raise ValueError(f"pins missing required key: {key}")
    bn = data["bitnet_cpp"]
    for key in ("repo", "ref"):
        if not bn.get(key):
            raise ValueError(f"bitnet_cpp.{key} required")
    gg = data["llama_cpp_gguf"]
    for key in ("repo", "ref"):
        if not gg.get(key):
            raise ValueError(f"llama_cpp_gguf.{key} required")
    if data["policy"].get("vendor_submodule") is not False:
        raise ValueError("policy.vendor_submodule must be false (recipe+pin, no giant submodule)")
    return data


def _bitnet_steps(pins: dict[str, Any]) -> list[str]:
    bn = pins["bitnet_cpp"]
    model = pins["models"]["bitnet_b158_2b_4t_gguf"]
    ref = bn["ref"]
    local = model["local_dir_hint"]
    quant = model["quant_default"]
    hf = model["hf_id"]
    return [
        f"git clone --recursive {bn['repo']} BitNet && cd BitNet",
        f"git checkout {ref} && git submodule update --init --recursive",
        "# Optional: conda create -n bitnet-cpp python=3.10 && conda activate bitnet-cpp",
        "pip install -r requirements.txt",
        f"huggingface-cli download {hf} --local-dir {local}",
        f"python setup_env.py -md {local} -q {quant}",
        "# Windows: use VS2022 Developer shell; see upstream FAQs for ClangCL",
        f"# Inference (paths vary by OS): python run_inference.py -m {local}/ggml-model-{quant}.gguf -p 'Hello' -n 64",
        "docs/23_BITNET_CPP_BRIDGE.md",
    ]


def _llama_gguf_steps(pins: dict[str, Any]) -> list[str]:
    gg = pins["llama_cpp_gguf"]
    ref = gg["ref"]
    return [
        f"git clone {gg['repo']} llama.cpp && cd llama.cpp",
        f"git checkout {ref}",
        "cmake -B build -DGGML_NATIVE=ON && cmake --build build --config Release",
        "# Convert HF → GGUF (upstream convert scripts for the model family) then:",
        "./llama-quantize model-f16.gguf model-Q4_K_M.gguf Q4_K_M",
        "./llama-cli -m model-Q4_K_M.gguf -p 'Hello' -n 64",
    ]


def build_recipe(pins: dict[str, Any]) -> dict[str, Any]:
    bn = pins["bitnet_cpp"]
    vendored = bn.get("vendored_llama_cpp_fork", {})
    return {
        "lane": "cpu-llm",
        "schema": "bnn.bridge_cpu_llamacpp_bitnet.v1",
        "thesis": (
            "CPU chat LLMs → GGUF Q4_K_M (llama.cpp) or BitNet → bitnet.cpp. "
            "Not sign()+torch."
        ),
        "pins_file": str(PINS_PATH.relative_to(ROOT).as_posix()),
        "pins": {
            "bitnet_cpp_ref": bn["ref"],
            "bitnet_cpp_repo": bn["repo"],
            "llama_cpp_gguf_ref": pins["llama_cpp_gguf"]["ref"],
            "llama_cpp_gguf_repo": pins["llama_cpp_gguf"]["repo"],
            "vendored_llama_cpp_fork_ref": vendored.get("ref"),
            "pinned_at": pins.get("pinned_at"),
            "vendor_submodule": False,
        },
        "llama_cpp_gguf": _llama_gguf_steps(pins),
        "bitnet_cpp": _bitnet_steps(pins),
        "models": pins["models"],
        "requirements": pins.get("requirements", {}),
        "this_repo_shims": [
            "bnn recommend --goal cpu-llm",
            "bnn encode / bnn decode → .bnnpack for lab packed Linear artifacts",
            "TernaryLinear + ternary_pack = pedagogy/size; speed for BitNet = bitnet.cpp",
            "python scripts/bridges/llamacpp_bitnet_recipe.py --check",
        ],
        "non_goals": [
            "Do not absmean-PTQ a chat Llama and ship as BitNet",
            "Do not claim this repo's XNOR kernels replace bitnet.cpp tokens/s",
            "Do not vendor microsoft/BitNet as a default git submodule",
        ],
    }


def probe_local_bitnet(root: Path | None) -> dict[str, Any]:
    """Optional local checkout probe (no network, no build)."""
    env = os.environ.get("BNN_BITNET_ROOT", "").strip()
    candidates: list[Path] = []
    if root is not None:
        candidates.append(root)
    if env:
        candidates.append(Path(env))
    candidates.extend(
        [
            Path.cwd() / "BitNet",
            Path.home() / "BitNet",
            Path.home() / "src" / "BitNet",
        ]
    )
    seen: set[Path] = set()
    for cand in candidates:
        try:
            resolved = cand.resolve()
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        setup = resolved / "setup_env.py"
        if setup.is_file():
            return {
                "found": True,
                "path": str(resolved),
                "has_setup_env": True,
                "has_3rdparty_llama": (resolved / "3rdparty" / "llama.cpp").exists(),
                "platform": platform.platform(),
            }
    return {
        "found": False,
        "searched": [str(c) for c in candidates],
        "hint": "Clone microsoft/BitNet at the pinned ref, or set BNN_BITNET_ROOT",
        "platform": platform.platform(),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Emit pinned llama.cpp / bitnet.cpp CPU-LLM bridge recipe (no heavy deps)."
    )
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "bridge_cpu_llamacpp_bitnet.json",
    )
    p.add_argument(
        "--check",
        action="store_true",
        help="Validate pins JSON + recipe shape; write nothing; exit non-zero on failure",
    )
    p.add_argument(
        "--probe",
        action="store_true",
        help="Also probe for a local BitNet checkout (BNN_BITNET_ROOT or ~/BitNet)",
    )
    p.add_argument(
        "--bitnet-root",
        type=Path,
        default=None,
        help="Explicit local BitNet checkout for --probe",
    )
    p.add_argument(
        "--pins",
        type=Path,
        default=PINS_PATH,
        help="Override pins JSON path",
    )
    args = p.parse_args(argv)

    try:
        pins = load_pins(args.pins)
        recipe = build_recipe(pins)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.probe or args.bitnet_root is not None:
        recipe["local_bitnet_probe"] = probe_local_bitnet(args.bitnet_root)

    if args.check:
        assert recipe["pins"]["vendor_submodule"] is False
        assert recipe["pins"]["bitnet_cpp_ref"]
        assert recipe["pins"]["llama_cpp_gguf_ref"]
        assert any("setup_env.py" in step for step in recipe["bitnet_cpp"])
        print(json.dumps({"ok": True, "pins": recipe["pins"]}, indent=2))
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(recipe, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(recipe, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
