# Reproducibility (humans + other AIs)

Anyone cloning this repo should get the **same conclusions** under published
tolerance gates — not necessarily bit-identical floats.

**Thesis lock is unchanged:** packed binary/ternary CPU kernels for inference;
STE training is simulation, not a GPU speed claim.

## 5-command fast verify

```bat
git clone https://github.com/KanakMalpani/Binary-Neural-Networks.git
cd Binary-Neural-Networks
python -m pip install -U pip
pip install -e ".[dev]" -c constraints.txt
bnn repro
```

Equivalent: `python scripts/repro_all.py` (default `--mode verify`).

Expect **REPRO: PASS** and exit code `0` in a few minutes.

## AI coding agents

If you are an AI agent, follow **[`AGENTS.md`](AGENTS.md)** exactly:
run the listed commands in order; do **not** invent new benches; compare to
`tests/golden_floors.json` and committed `results/*.json`.

## Environment

| Item | Requirement |
|------|-------------|
| Python | **3.11+** (3.12 recommended) |
| OS | Windows x64 (native DLL) or Linux/macOS (NumPy fallback) |
| Native kernel | Windows **MSVC x64** via `python -m bnn.kernels.compile_native` |
| MinGW | **Do not use** for the DLL — WinError 193 / wrong arch |
| Deps | `pyproject.toml` + `constraints.txt` |
| Device for goldens | **CPU** (`bnn.determinism.set_repro_seed(..., force_cpu=True)`) |

### Hardware assumptions

- CPU x64 with enough RAM for 8192×8192 GEMM microbench (~few hundred MB peak).
- Native popcount DLL: Windows + Visual Studio Build Tools / VS 2022.
- Elsewhere: NumPy packed GEMM (correctness preserved; speedups lower).

## What is committed vs must-rerun

| Artifact | Role |
|----------|------|
| `results/*.json` | **Published goldens** (source of truth for gates) |
| `tests/golden_floors.json` | Tolerance floors + recorded numbers |
| `data/` | **Not committed** (gitignored). MNIST/CIFAR download on first train; synthetic audio generated |
| `checkpoints/*.pt` | Not required for verify |
| Native `*.dll` | Built locally; gitignored |

**Fast verify** does **not** retrain. It checks committed JSONs + live microchecks
(export-check, pytest, native validate when DLL present).

**Full regen** (optional):

```bat
bnn repro --mode full
```

Writes smoke trains to `results/_repro_smoke_*.json` by default.
To overwrite published goldens (only when intentionally refreshing):

```bat
bnn repro --mode full --overwrite-goldens
```

Full published MNIST / CIFAR regenerations (longer):

```bat
bnn train --epochs 3 --seed 42
bnn train-image --epochs 8 --subset 30000 --seed 0 --approx-sign
bnn train-audio --epochs 5 --seed 0
bnn wrap --mode binary_xnor
bnn bench --reps 5
bnn eval-suite --skip-pytest
```

## Guaranteed identical vs tolerance-gated

| Claim | Guarantee |
|-------|-----------|
| Weight pack compression | **Exact 32×** (uint64 bit pack) |
| Native GEMM vs ±1 FP (when DLL loaded) | **err = 0** |
| NumPy packed GEMM vs FP | **err = 0** (pytest) |
| MNIST / CIFAR / audio accuracies | **Tolerance gates** (±pp in `golden_floors.json`) |
| Kernel speedups | Soft floors (machine-dependent); conclusions: native beats NumPy FP on large shapes when DLL present |
| Float tensors across OS/BLAS | **Not** bit-identical |

Nondeterministic / variable: CUDA (avoided for goldens), some Conv paths,
wall-clock latency, first-run dataset download time.

## Expected wall times (approx, CPU)

| Step | Time |
|------|------|
| `pip install -e ".[dev]"` | 1–5 min (torch wheel) |
| `compile-native` | &lt; 1 min (Windows MSVC) |
| `bnn repro` verify | **2–5 min** |
| `bnn repro --mode full` smokes | +5–20 min (CIFAR download first time) |
| Full MNIST 3 epochs | ~5–15 min |
| Full image CIFAR 8 ep / 30k | ~30–60+ min |
| Full audio 5 ep | &lt; 1 min |

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| WinError 193 loading DLL | Rebuild with **MSVC x64**, not MinGW 32-bit |
| `native_kernel_available() == False` | Expected on Linux/macOS; pytest NumPy path still PASS |
| Missing MSVC | Install VS 2022 Build Tools + “Desktop development with C++”; open **x64 Native Tools** shell |
| CIFAR/MNIST download fails | Network; retry. Data lands under `data/` (ignored) |
| Speedup below soft floor | Machine variance OK if compression=32 and err=0; check `golden_floors` min 2.0× |
| pytest fails golden gates | Do not invent new benches — inspect `results/*.json` vs floors; ask before overwriting goldens |

## CI

GitHub Actions (`.github/workflows/ci.yml`):

- **Windows:** install → compile-native (best-effort) → pytest → export-check → `bnn repro --skip-compile` (or equivalent)
- **Linux:** pytest + export-check + golden compare (NumPy fallback; native skipped)

## Related docs

- [`AGENTS.md`](AGENTS.md) — agent entrypoint
- [`docs/30_REPRO_FOR_OTHER_AIS.md`](docs/30_REPRO_FOR_OTHER_AIS.md) — what shipped for third-party repro
- [`docs/29_FINAL_COMPLETION.md`](docs/29_FINAL_COMPLETION.md) — done criteria
