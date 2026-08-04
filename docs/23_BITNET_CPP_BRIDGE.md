# BitNet → bitnet.cpp bridge

For **ternary / 1.58-bit LLMs**, production CPU inference is **bitnet.cpp**
([microsoft/BitNet](https://github.com/microsoft/BitNet)), not classic BNN
XNOR and not stock llama.cpp alone.

This lab **does not vendor** BitNet as a git submodule (size / recursive
llama.cpp fork). Instead: **recipe + SHA pin** under
[`scripts/bridges/llamacpp_bitnet_pins.json`](../scripts/bridges/llamacpp_bitnet_pins.json).

## Thesis lock

| Do | Don't |
|----|--------|
| Native BitNet checkpoints → bitnet.cpp | Absmean-PTQ a chat Llama and ship as “BitNet” |
| Dual-metric: tokens/s **and** energy-proxy on the same machine | Claim this repo’s packed GEMM replaces bitnet.cpp LLM serve |
| GGUF `Q4_K_M` via stock llama.cpp for **non-BitNet** HF chat | Confuse GGUF INT4 with BitNet 1.58-bit formats |

## Pinned refs (re-pin deliberately)

Read the pins file for the live SHAs/tags. Snapshot at lane ship:

| Component | Pin |
|-----------|-----|
| microsoft/BitNet | `0b341e582afbf9e1011f24744b554c96a3477eb5` (2026-07-27) |
| BitNet’s vendored llama.cpp fork | `isHuangXin/llama.cpp` @ `390c3077…` (submodule tip at that BitNet SHA) |
| Stock llama.cpp (GGUF path) | tag `b10262` |
| Reference GGUF | `microsoft/BitNet-b1.58-2B-4T-gguf` |

```bat
python scripts\bridges\llamacpp_bitnet_recipe.py --check
python scripts\bridges\llamacpp_bitnet_recipe.py
```

Writes `results/bridge_cpu_llamacpp_bitnet.json` with clone/build/quantize steps
using those pins.

## Steps (bitnet.cpp)

1. Obtain a **native** BitNet checkpoint (HF BitNet cards) **or** plan BitDistill /
   gradual-λ QAT — do not absmean-PTQ a chat Llama and ship.
2. Clone BitNet **out-of-tree** at the pinned SHA with `--recursive`:

   ```bash
   git clone --recursive https://github.com/microsoft/BitNet.git BitNet
   cd BitNet
   git checkout 0b341e582afbf9e1011f24744b554c96a3477eb5
   git submodule update --init --recursive
   ```

3. `pip install -r requirements.txt` (Python ≥3.9; CMake ≥3.22; Clang 18+ recommended).
   On Windows use a VS2022 Developer shell / ClangCL per upstream FAQs.
4. Download GGUF weights and run upstream setup (quant default `i2_s`):

   ```bash
   huggingface-cli download microsoft/BitNet-b1.58-2B-4T-gguf \
     --local-dir models/BitNet-b1.58-2B-4T
   python setup_env.py -md models/BitNet-b1.58-2B-4T -q i2_s
   ```

5. Run inference with upstream `run_inference.py` / built binaries; measure
   tokens/s and energy vs FP/GGUF baselines on the **same** machine.

## Steps (non-BitNet HF → GGUF)

See [`22_HF_TO_GGUF_GUIDE.md`](22_HF_TO_GGUF_GUIDE.md). Recipe script also emits
stock llama.cpp steps pinned to tag `b10262`.

## Local probe (optional)

```bat
set BNN_BITNET_ROOT=C:\path\to\BitNet
python scripts\bridges\llamacpp_bitnet_recipe.py --probe --out results\bridge_cpu_llamacpp_bitnet.json
```

Probe only checks for `setup_env.py` — it does **not** build or download models.

## This repo’s role

- `TernaryLinear` + `ternary_pack` = **pedagogy / size**
- Speed for LLMs = **bitnet.cpp** (or custom Hexagon kernels on Snapdragon)
- Portable lab artifacts: `bnn encode` / `bnn decode` (`.bnnpack`) — not a
  llama.cpp / bitnet.cpp replacement
- `bnn recommend --goal cpu-llm` points here

## CLI (preferred)

```bat
bnn bridge list
bnn bridge cpu-llm
bnn bridge bitnet
```

Aliases `cpu-llm` / `bitnet` / `llamacpp` all run the same recipe script
(`scripts/bridges/llamacpp_bitnet_recipe.py`).

## CI / acceptance

- Default CI: `pytest` validates pins schema + recipe (`tests/test_llamacpp_bitnet_bridge.py`).
- Full bitnet.cpp compile + model download: **local / optional** `workflow_dispatch`
  only — never a required gate (multi-GB, toolchain-heavy).
- Integrator: flip ROADMAP **W4.T06** / WC-P2 handoff clarity from
  [`docs/lanes/i.md`](lanes/i.md) when merging.

## Related

- [`12_WRAPPER_AND_EXISTING_MODELS.md`](12_WRAPPER_AND_EXISTING_MODELS.md)
- [`20_NPU_VENDOR_CLOSURE.md`](20_NPU_VENDOR_CLOSURE.md) (no stock HTP ternary)
- [`22_HF_TO_GGUF_GUIDE.md`](22_HF_TO_GGUF_GUIDE.md)
- [`24_GPU_INT4_FP8_LANE.md`](24_GPU_INT4_FP8_LANE.md) (GPU lane is INT4/FP8, not XNOR)
- [`36_ENCODER_DECODER_AND_NEXT.md`](36_ENCODER_DECODER_AND_NEXT.md)
- [`MOONSHOT_DEFERRALS.md`](MOONSHOT_DEFERRALS.md) — submodule remains deferred by design
- `bnn recommend --goal cpu-llm` / `bnn bridge bitnet`
