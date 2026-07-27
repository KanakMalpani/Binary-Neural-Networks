# End-to-end user guide — Binary Neural Network Optimiser

**Audience:** humans and AI agents who need a single, followable path from
**zero → working optimiser results** without inventing new benches or fake
GPU speedups.

**Time:** ~15 minutes for the optimiser path; ~30–60 minutes if you also train
a modality. **Pass gate:** `bnn repro` ends with `REPRO: PASS` (exit 0).

| | |
|--|--|
| **Repo** | https://github.com/KanakMalpani/Binary-Neural-Networks |
| **Agents** | [`AGENTS.md`](../AGENTS.md) (ordered install) · this guide (human-equivalent path) |
| **Roadmap** | [`ROADMAP.md`](../ROADMAP.md) · twin [`37_WORLD_CLASS_BNN_OPTIMISER_ROADMAP.md`](37_WORLD_CLASS_BNN_OPTIMISER_ROADMAP.md) |
| **Repro deep dive** | [`REPRODUCIBILITY.md`](../REPRODUCIBILITY.md) |
| **Tutorials** | [`tutorials/`](tutorials/) 01–08 (linked below) |

---

## Table of contents

1. [Who this is for / thesis (1 screen)](#1-who-this-is-for--thesis-1-screen)
2. [Prerequisites](#2-prerequisites)
3. [Install & verify](#3-install--verify)
4. [15-minute path: `bnn optimise`](#4-15-minute-path-bnn-optimise)
5. [Weight codec: encode / decode `.bnnpack`](#5-weight-codec-encode--decode-bnnpack)
6. [Train modalities](#6-train-modalities)
7. [Kernels & threads](#7-kernels--threads)
8. [Bridges: binary vs INT4/FP8 (decision tree)](#8-bridges-binary-vs-int4fp8-decision-tree)
9. [HF / local optimiser tutorial path](#9-hf--local-optimiser-tutorial-path)
10. [Interpreting metrics](#10-interpreting-metrics)
11. [Troubleshooting](#11-troubleshooting)
12. [Next steps → ROADMAP](#12-next-steps--roadmap)

Commands below use **Windows `.bat` / PowerShell** syntax. Unix notes are
inline (`/` paths, `export` instead of `set`).

---

## 1. Who this is for / thesis (1 screen)

**You want:** to **optimise** PyTorch (or HF) models for **CPU / edge** by
packing weights to **1–1.58 bits** and running **real** XNOR/popcount (or
ternary) kernels — then get an honest report (size, cosine, latency).

**You do not want:** `sign()` in `nn.Linear` pretending to be **32× on GPU**.
That is an anti-pattern (often *slower*). Commodity GPU quality → **INT4 /
FP8 / AWQ / vLLM**. Local BitNet-style LLMs → **bitnet.cpp**. This repo is
the **Binary Neural Network Optimiser lab/product** for packed CPU/edge.

| Locked claim | Meaning |
|--------------|---------|
| Compression **32×** | Exact for aligned uint64 pack of binary weights — **theory / size**, not e2e latency |
| Native GEMM **err = 0** | Packed output matches ±1 FP reference when the DLL is loaded |
| Dual metrics | Always report **theory** (word reduction) **and** **wall-clock** separately |
| Repro | Same *conclusions* via `golden_floors.json` + `results/*.json` — floats need not be bit-identical |

**Product verb:** `bnn optimise` (preferred). Legacy: `bnn wrap --ultra`.

---

## 2. Prerequisites

| Item | Requirement |
|------|-------------|
| Python | **3.11+** (3.12 recommended; avoid bleeding-edge unless you know torch wheels exist) |
| OS | Windows / Linux / macOS (x64 or arm64). Prefer native kernel via `compile_native`; NumPy packed GEMM is the correctness fallback if native is absent |
| Git | Clone the repo |
| Disk / RAM | Few hundred MB peak for large GEMM microbenches; `data/` downloads on first train |
| Windows native (optional but preferred) | **MSVC x64** — Visual Studio 2022 Build Tools + “Desktop development with C++”. Open an **x64 Native Tools** shell if `cl` is missing from PATH |
| MinGW | **Do not** build the native DLL with MinGW — WinError 193 / wrong arch |

Optional extras:

```bat
pip install -e ".[hf]" -c constraints.txt
```

Needed only for [tutorial 08](tutorials/08_HF_OPTIMISER.md) (Hugging Face).

---

## 3. Install & verify

### 3.1 Clone and install

**Windows:**

```bat
git clone https://github.com/KanakMalpani/Binary-Neural-Networks.git
cd Binary-Neural-Networks
python -m pip install -U pip
pip install -e ".[dev]" -c constraints.txt
```

**Unix:**

```bash
git clone https://github.com/KanakMalpani/Binary-Neural-Networks.git
cd Binary-Neural-Networks
python -m pip install -U pip
pip install -e ".[dev]" -c constraints.txt
```

**Expect:** install finishes without error; `bnn --version` prints something like `bnn 0.2.0`.

**Failure modes:** wrong Python on PATH; torch wheel missing for your version → use 3.12; corporate proxy → configure pip.

### 3.2 Compile native kernels (Windows)

```bat
python -m bnn.kernels.compile_native
```

Or: `bnn compile-native`.

**Expect:** a message that the DLL was built / already present under `bnn/kernels/` (gitignored).

**Failure modes:**

| Symptom | Fix |
|---------|-----|
| `cl` not found | Install VS Build Tools C++; use **x64 Native Tools** prompt |
| WinError 193 on load | Rebuild with **MSVC x64**, never MinGW 32-bit |
| Linux/macOS | Run `python -m bnn.kernels.compile_native` (preferred); if `.so` missing, NumPy path is used — still run `bnn repro` |

### 3.3 Verify — `bnn repro`

```bat
bnn repro
```

Equivalent: `python scripts/repro_all.py` or `python -m bnn repro`.

**Expect (tail of stdout):**

```text
REPRO: PASS
```

Exit code **0**. Typical wall time: **2–5 minutes** (no full retrain).

**What it checks (among others):**

- Compression **32×** (export-check)
- Native GEMM **err = 0** when DLL present (else NumPy path via pytest)
- Committed `results/*.json` vs `tests/golden_floors.json` floors

**Failure modes:** see [§11](#11-troubleshooting). Do **not** invent new benchmark shapes to “make it green.”

Optional longer smoke trains (does not overwrite published goldens by default):

```bat
bnn repro --mode full
```

---

## 4. 15-minute path: `bnn optimise`

This is the **primary product path**: toy FP model → policy (auto/hybrid) →
optional light QAT → versioned JSON report → optional `.bnnpack`.

### 4.1 One command

```bat
bnn optimise --policy auto --report results\optimise_report.json
```

**Unix:** `bnn optimise --policy auto --report results/optimise_report.json`

With light STE recovery + packfile:

```bat
bnn optimise --policy auto --qat-steps 40 --force --pack results\demo_opt.bnnpack --report results\optimise_report.json
```

**Expect:**

- Exit code 0
- `results/optimise_report.json` (or suite JSON) with schema **`bnn_optimise_report_v1`**
- Console lines mentioning policies, cosine / agreement, compression, and whether drop-in is allowed

Useful flags:

| Flag | Meaning |
|------|---------|
| `--policy auto` | Hardware-aware binary vs ternary vs skip |
| `--qat-steps 40` | Light STE recovery on FFN names (binary path) |
| `--force` | Allow drop-in *claim* below cosine threshold (report still marks `forced: true`) |
| `--pack PATH` | Also encode a toy MLP `.bnnpack` |
| `--report PATH` | Where to write the JSON report |

**Legacy (still works, prefer optimise):**

```bat
bnn wrap --ultra --policy auto --qat-steps 40 --force
```

Plain `bnn wrap` (no `--ultra`) is a **legacy** wide-MLP microbench and may emit
`DeprecationWarning` → use `bnn optimise`.

### 4.2 What the report means (read honestly)

Open `results/optimise_report.json` (or the primary object inside a suite).

| Field / idea | How to read it |
|--------------|----------------|
| `compression_replaced_weights` ≈ **32** | **Theory** — aligned binary pack size ratio |
| Cosine / top-1 agreement | Quality after wrap (+ optional QAT). Low cosine → not drop-in |
| `drop_in_ok` / `status: REFUSE_DROP_IN_CLAIM` | Marketing gate — do not ship as drop-in without metrics |
| `e2e_latency_ms_*` / `layer_microbench.speedup_gemm_only_vs_torch` | **Wall-clock** — machine-dependent; not “32×” |
| `policy_reason` / skipped layers | Why embed/attn/lm_head stayed FP |

**Thesis reminder:** never quote compression alone as end-to-end speedup.

### 4.3 Python API (same story)

```python
import torch
import torch.nn as nn
from bnn.optimise import OptimiseConfig, optimise_model

class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(64, 64)
        self.ffn_fc1 = nn.Linear(64, 256)
        self.ffn_fc2 = nn.Linear(256, 64)
        self.lm_head = nn.Linear(64, 10)

    def forward(self, x):
        h = torch.relu(self.embed(x))
        return self.lm_head(self.ffn_fc2(torch.relu(self.ffn_fc1(h))))

model = Tiny()
x = torch.randn(8, 64)
result = optimise_model(
    model,
    x,
    OptimiseConfig(policy="hybrid_ffn", mode="binary_xnor", min_in_features=32, force=True),
)
print(result.payload["compression_replaced_weights"], result.payload["status"])
```

Deeper walkthrough: [tutorial 07](tutorials/07_OPTIMISER_QUICKSTART.md).

---

## 5. Weight codec: encode / decode `.bnnpack`

Portable packed weights for BinaryLinear / packed XNOR layers.

```bat
bnn encode --source mlp --hidden 256 --out results\demo.bnnpack
bnn decode --pack results\demo.bnnpack
```

Or a random Linear:

```bat
bnn encode --source random --in-features 512 --out-features 512 --out results\demo_rand.bnnpack
bnn decode --pack results\demo_rand.bnnpack
```

**Expect encode:**

```text
Wrote results\demo.bnnpack layers=... compression=32.00x
```

**Expect decode:**

```text
DECODE: PASS
```

Each layer should show `fp_err=0.0` (or `0`) and ~**32×** compression.

**Failure modes:** corrupt / wrong-endian file; non-packed layers only; path typos.

Also covered in [tutorial 06](tutorials/06_encoder_decoder.md).

---

## 6. Train modalities

Verify committed goldens anytime with `bnn repro` (no retrain). Full trains
are optional and longer.

### 6.1 MNIST binary MLP — [tutorial 01](tutorials/01_mnist_binary.md)

```bat
bnn validate-native
bnn export-check
bnn train --epochs 3 --seed 42 --model binary_mlp
bnn bench --reps 5
```

**Expect:** compression **32×**; native err **0** when DLL present; `binary_mlp`
test accuracy within floors in `tests/golden_floors.json` (typically ≥ ~95% when
FP ≥ ~97%).

### 6.2 Wrap Linears (legacy microbench) — [tutorial 02](tutorials/02_wrap_linear.md)

Prefer §4 (`bnn optimise`). For the wide-MLP microbench only:

```bat
bnn wrap --mode binary_xnor --hidden 4096 --batch 32
```

### 6.3 CIFAR Bi-Real proxy — [tutorial 03](tutorials/03_cifar_bireal.md)

```bat
pip install datasets
bnn train-cifar --epochs 5 --subset 20000
```

### 6.4 Image lane (CIFAR) — [tutorial 04](tutorials/04_image_cifar.md)

```bat
bnn train-image --epochs 8 --subset 30000 --seed 0 --approx-sign
```

Committed golden: `results/image_cifar.json`. Smoke: `pytest tests\test_vision_smoke.py -q`.

**Honesty:** Conv pack today is **size** (~32×) + dequant FP forward — no native
binary-conv DLL. Do not claim 32× wall-clock for Conv.

### 6.5 Audio (synthetic) — [tutorial 05](tutorials/05_audio.md)

```bat
bnn train-audio --epochs 5 --n-train 800 --n-test 200 --seed 0
```

**Not** production ASR — use INT8 Whisper / ORT for real speech. Golden:
`results/audio_synth.json`.

### 6.6 Seq2seq encoder–decoder — [tutorial 06](tutorials/06_encoder_decoder.md)

```bat
bnn train-seq2seq --task seq2seq --steps 80 --out results\seq2seq_encoder_decoder.json
bnn wrap-transformer --qat-steps 40
```

Architecture: FP attention + LayerNorm; binary/ternary FFN via STE.

---

## 7. Kernels & threads

### 7.1 Profile a packed Linear

```bat
bnn profile --batch 64 --in-features 4096 --out-features 4096
```

Optional: `--out results\profile.json`. JSON includes native vs NumPy / torch
timings when available.

### 7.2 Thread control

**Windows:**

```bat
set BNN_NUM_THREADS=4
bnn bench --reps 5
```

**Unix:**

```bash
export BNN_NUM_THREADS=4
bnn bench --reps 5
```

`OMP_NUM_THREADS` is also respected. Prefer **4–8** threads; matching logical
CPU count can oversubscribe and plateau (memory bandwidth). Details:
[`34_COMPUTE_SPEEDUP.md`](34_COMPUTE_SPEEDUP.md).

### 7.3 Validate / export

```bat
bnn validate-native
bnn export-check
```

**Expect:** export-check reports **32.00×** compression; validate-native reports
**err = 0** when the DLL loads.

---

## 8. Bridges: binary vs INT4/FP8 (decision tree)

Use this repo’s packed BNN path only when it matches the goal. Otherwise
**bridge** — do not fake a BNN win.

```
GPU server max quality?     → FP8 / AWQ-INT4 + vLLM / torchao   (NOT classic BNN)
CPU local LLM chat?         → BitNet checkpoint? bitnet.cpp : GGUF Q4_K_M
Edge vision (can retrain)?  → Bi-Real + this repo / LCE / FINN ; else INT8
Phone NPU stock SDK?        → INT8/INT4  (no stock 1-bit)
Research XNOR kernels?      → this repo (`bnn optimise` / train / kernels)
Diffusion fidelity?         → INT8/FP8 PTQ ; avoid full BNN
```

CLI helper:

```bat
bnn recommend --goal edge-vision
```

| Goal example | Prefer |
|--------------|--------|
| `edge-vision` | This repo / Bi-Real patterns |
| GPU LLM serve | [`24_GPU_INT4_FP8_LANE.md`](24_GPU_INT4_FP8_LANE.md) + `scripts/bridges/` |
| BitNet ternary | [`23_BITNET_CPP_BRIDGE.md`](23_BITNET_CPP_BRIDGE.md) |
| HF → GGUF | [`22_HF_TO_GGUF_GUIDE.md`](22_HF_TO_GGUF_GUIDE.md) |

Full tree: [`18_DECISION_TREE_AND_COMPLETE_ROADMAP.md`](18_DECISION_TREE_AND_COMPLETE_ROADMAP.md) ·
one-pager [`25_ONEPAGER.md`](25_ONEPAGER.md).

---

## 9. HF / local optimiser tutorial path

| Step | Doc / command |
|------|----------------|
| Local toy optimise | §4 + [07_OPTIMISER_QUICKSTART.md](tutorials/07_OPTIMISER_QUICKSTART.md) |
| Hugging Face | [08_HF_OPTIMISER.md](tutorials/08_HF_OPTIMISER.md) |
| API ADR | [`adr/0001_public_optimiser_api.md`](adr/0001_public_optimiser_api.md) |

HF install + tiny demo:

```bat
pip install -e ".[hf]" -c constraints.txt
python scripts\hf_tiny_wrap_demo.py --out results\hf_tiny_wrap.json
```

**Warning:** PTQ wrap of Transformers without real calib + QAT is pedagogical.
Attention / embeds usually skipped (`hybrid_ffn`). Production GPU LLMs → INT4/FP8;
BitNet serve → bitnet.cpp.

Optional offline-hostile test (network):

```bat
pytest -q -m "hf" tests\test_hf_optimiser.py
```

Default `bnn repro` stays offline-friendly (HF tests marked `hf` / `slow`).

---

## 10. Interpreting metrics

| Metric | Kind | Honest use |
|--------|------|------------|
| Compression **32×** (binary aligned pack) | Exact / theory | Size / word reduction — **not** e2e latency |
| Ternary pack ~**16×** | Theory | Weight-only ternary pedagogy |
| Cosine / top-1 vs FP | Quality | Drop-in gate; refuse marketing if below threshold |
| Native GEMM **err = 0** | Exact | Correctness of packed kernels |
| `speedup_gemm_only_vs_torch` / bench × | Wall-clock | Machine-dependent; publish dual metrics |
| Soft speedup floors in `golden_floors.json` | Soft | Variance OK if compression + err gates pass |

**Dual speedup reporting (required culture):**

1. **Theoretical** word / bit reduction (e.g. 32× pack).
2. **Measured** wall-clock (and optional energy-proxy) on named shapes.

Never collapse (1) into (2) in README claims, papers, or PRs.

Compare new numbers only to:

- `tests/golden_floors.json`
- committed `results/*.json`

Do **not** invent alternate bench shapes as “the” golden.

---

## 11. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `AttributeError: numpy has no attribute bitwise_count` | Fixed via `bnn.kernels.popcount` LUT fallback (NumPy 1.24+). Upgrade to NumPy 2+ optional; reinstall with `-c constraints.txt` |
| `REPRO: FAIL` / non-zero exit | Read failing gate; fix env/DLL; do not change shapes or floors without explicit intent |
| WinError 193 loading DLL | Rebuild with **MSVC x64**; delete MinGW-built DLL; `python -m bnn.kernels.compile_native --force` |
| `native_kernel_available() == False` | Possible if compile/load failed; NumPy path must still pass pytest. Prefer `compile_native` on Linux/macOS/ARM too |
| `cl` / MSVC missing | VS 2022 Build Tools + C++ workload; **x64 Native Tools** shell |
| Cosine collapse / `REFUSE_DROP_IN_CLAIM` | Expected for aggressive binary PTQ — use `--policy auto` / ternary / QAT / `--force` with honesty |
| CIFAR/MNIST download fails | Network; retry; data under `data/` (gitignored — never commit) |
| Speedup below soft floor | Machine variance; keep compression=32 and err=0 as hard gates |
| `bnn wrap` DeprecationWarning | Switch to `bnn optimise` |
| HF import / download errors | `pip install -e ".[hf]"`; check network; use tiny test model |
| Python too new (no torch wheel) | Use **3.12** |

Expanded notes: [`REPRODUCIBILITY.md`](../REPRODUCIBILITY.md) troubleshooting table.

---

## 12. Next steps → ROADMAP

You now have: install → `bnn repro` → `bnn optimise` → encode/decode → optional
trains → metrics literacy → bridge decision tree.

**Product direction when lost:**

1. Read [`ROADMAP.md`](../ROADMAP.md) §0 → §7 → current phase → lowest unchecked TODO.
2. Keep tutorials green; prefer `bnn optimise` over legacy wrap.
3. Do not invent benches; do not claim GPU 32× from `sign()`.

**Tutorial map (ordered):**

| # | File | Topic |
|---|------|--------|
| 01 | [`01_mnist_binary.md`](tutorials/01_mnist_binary.md) | MNIST + bench |
| 02 | [`02_wrap_linear.md`](tutorials/02_wrap_linear.md) | Wrap Linears (prefer `optimise`) |
| 03 | [`03_cifar_bireal.md`](tutorials/03_cifar_bireal.md) | CIFAR proxy |
| 04 | [`04_image_cifar.md`](tutorials/04_image_cifar.md) | Image lane |
| 05 | [`05_audio.md`](tutorials/05_audio.md) | Synthetic audio |
| 06 | [`06_encoder_decoder.md`](tutorials/06_encoder_decoder.md) | Seq2seq + `.bnnpack` |
| 07 | [`07_OPTIMISER_QUICKSTART.md`](tutorials/07_OPTIMISER_QUICKSTART.md) | Optimiser quickstart |
| 08 | [`08_HF_OPTIMISER.md`](tutorials/08_HF_OPTIMISER.md) | Hugging Face path |

**Index:** [`docs/README.md`](README.md) · smoke confirmation:
[`39_GUIDE_E2E_COMPLETION.md`](39_GUIDE_E2E_COMPLETION.md).
