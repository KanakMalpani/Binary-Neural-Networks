# Compatibility matrix

**Task:** W14.T01 · **Audit:** 2026-07-25

| OS | Arch | Python | Torch | Native kernel | Notes |
|----|------|--------|-------|---------------|-------|
| Windows 10/11 | x64 | 3.11–3.13 | pinned in `constraints.txt` | MSVC OpenMP DLL | Preferred native path |
| Linux | x64 | 3.11–3.13 | same | NumPy fallback (`.so` = W2.T02) | CI pytest + repro |
| macOS | x64 / arm64 | 3.11–3.13 | same | NumPy fallback (NEON spike = W2.T04) | Not in default CI yet |

**Requires-python:** `>=3.11` (`pyproject.toml`).

**Extras:** `[dev]`, `[hf]` (transformers), `[all]`.

**CI today:** Windows + Linux (see `.github/workflows/ci.yml`). Python matrix
3.11/3.12/3.13 = W8.T03 (tracked).

**Policy:** floats need not be bit-identical across machines; same **conclusions**
vs `tests/golden_floors.json`.
