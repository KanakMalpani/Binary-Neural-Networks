# macOS notes (W14.T05)

| Field | Value |
|-------|-------|
| **Native kernel** | Not shipped; use **NumPy** correctness path |
| **NEON** | Spike deferred — [`spikes/ARM_NEON_SPIKE.md`](spikes/ARM_NEON_SPIKE.md) |
| **CI** | Not in default matrix (cost); Linux/Windows cover gates |

## Install

```bash
python -m pip install -U pip
pip install -e ".[dev]" -c constraints.txt
bnn repro
```

Optional: attempt `python -m bnn.kernels.compile_native` with Homebrew `gcc`
/`libomp` — best-effort; if `.so` fails to load, NumPy path remains correct.

## Accelerate

PyTorch may use Accelerate BLAS for FP baselines. Packed BNN path does not
claim Accelerate popcount wins until a Darwin native kernel lands.
