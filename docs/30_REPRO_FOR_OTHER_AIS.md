# 30 — Repro for other AIs (shipping note)

**Goal:** Third parties (humans or other coding agents) cloning
https://github.com/KanakMalpani/Binary-Neural-Networks get the same
**conclusions** under published gates.

## What was added

| Item | Purpose |
|------|---------|
| `bnn/determinism.py` | Seeds + `torch.use_deterministic_algorithms` + CPU policy |
| `scripts/repro_all.py` / `bnn repro` | One-command verify / optional full smoke |
| `tests/golden_floors.json` (v2) | Floors for MNIST, image, audio, wrap, compression, native |
| `tests/test_golden_gates.py` | Pytest asserts vs committed `results/*.json` |
| `constraints.txt` + pinned `pyproject.toml` | Portable dep band (Python ≥3.11) |
| `REPRODUCIBILITY.md` | Human + AI runbook |
| `AGENTS.md` | Strict agent command order |
| CI updates | Windows + Linux run reproducible smokes |

## Modes

- **Fast verify:** compile (best-effort) → pytest → export-check → validate-native
  (skip if no DLL) → golden compare → SUMMARY. No retrain.
- **Full:** + short deterministic smokes (default writes `_repro_smoke_*.json`).

## Guarantees

- Identical: compression 32×; native/NumPy GEMM err=0 (when path applies).
- Gated: accuracies within ±pp of published floors (not bit-identical floats).

Thesis lock unchanged. Datasets stay gitignored.
