# Compatibility matrix

**Task:** W14.T01 / W14.T02 · **Audit:** 2026-07-25 (v0.3.0)

| OS | Arch | Python | Torch | Native kernel | Notes |
|----|------|--------|-------|---------------|-------|
| Windows 10/11 | x64 | 3.11–3.13 | pinned in `constraints.txt` | MSVC OpenMP DLL | Preferred native path; CI soft compile |
| Linux | x64 | 3.11–3.13 | same | GCC `.so` **CI hard gate** | `linux-native` + `linux-py-matrix` jobs |
| macOS | x64 / arm64 | 3.11–3.13 | same | NumPy fallback | See [`MACOS_NOTES.md`](MACOS_NOTES.md); NEON = spike |

**Requires-python:** `>=3.11` (`pyproject.toml`).

**Extras:** `[dev]`, `[hf]` (transformers), `[all]`.

**CI:** Windows (3.12) + Linux native hard (3.12) + Linux Python matrix 3.11/3.12/3.13.

**Policy:** floats need not be bit-identical across machines; same **conclusions**
vs `tests/golden_floors.json`.
