# Compatibility matrix

**Task:** W14.T01 / W14.T02 · **Audit:** 2026-07-28 (portable SIMD + portability CI)

| OS | Arch | Python | Torch | Native kernel | Notes |
|----|------|--------|-------|---------------|-------|
| Windows 10/11 | x64 | 3.11–3.13 | pinned in `constraints.txt` | MSVC OpenMP DLL; runtime AVX-512→AVX2→scalar | Preferred lab path; CI soft compile |
| Linux | x64 | 3.11–3.13 | same | GCC `.so` **CI hard gate**; same runtime dispatch | `linux-native` + `linux-py-matrix` |
| Linux | arm64 | 3.12 | same | Clang/GCC `.so` + **NEON** | `portability` (`ubuntu-24.04-arm`); err=0 |
| macOS | arm64 | 3.11–3.13 | same | Clang `.so` + **NEON** | `portability` (`macos-latest`); see [`MACOS_NOTES.md`](MACOS_NOTES.md) |
| macOS | x86_64 | 3.11–3.13 | same | Clang `.so`; AVX2/scalar | `portability` (`macos-15-intel`) |
| Anything else | — | 3.11+ | same | **NumPy fallback** | Correctness first; no fake speedups |

**Requires-python:** `>=3.11` (`pyproject.toml`).

**Extras:** `[dev]`, `[hf]` (transformers), `[all]`.

**CI:** Windows (3.12) + Linux native hard (3.12) + Linux Python matrix 3.11/3.12/3.13 + **portability** (linux-arm64, macos-arm64, macos-x86_64).

**Policy:** floats need not be bit-identical across machines; same **conclusions**
vs `tests/golden_floors.json`. Runtime ISA details: [`41_PORTABLE_SIMD_KERNEL.md`](41_PORTABLE_SIMD_KERNEL.md).
