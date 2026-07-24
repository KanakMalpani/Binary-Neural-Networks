# Contributing / agent protocol

1. Read `docs/21_E2E_ROADMAP_COMPLETE_REPO.md` — pick first unchecked task with deps met.
2. Do not reopen science gaps in `docs/09` as blockers.
3. Honor ADR non-goals (`docs/08`): no CUDA-BNN 32× claims; OpenMP/ImageNet optional.
4. After a task: mark `[x]` in §10 of `docs/21`; add a `CHANGELOG.md` Unreleased line.
5. Run relevant tests: `pytest -q`, `bnn export-check`, `bnn validate-native`.
6. Keep PRs to one phase subsection when humans commit (agents: no commits unless asked).

Windows: compile kernels with **MSVC x64** (`python -m bnn.kernels.compile_native`). MinGW 32-bit → WinError 193.
