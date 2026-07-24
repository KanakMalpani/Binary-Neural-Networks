# Contributing

Thank you for helping keep this lab honest and reproducible.

## Before you change code

1. Read [`AGENTS.md`](AGENTS.md) (agents) or [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) (humans).
2. Honor the **thesis lock**: packed CPU/edge kernels; GPU → INT4/FP8; no fake 32× e2e.
3. Do not reopen closed science gaps in [`docs/09_GAP_REGISTER.md`](docs/09_GAP_REGISTER.md) as blockers.
4. Prefer improving gates/docs/DX over inventing new benchmark shapes.

## Dev setup

```bat
pip install -e ".[dev]" -c constraints.txt
python -m bnn.kernels.compile_native
pytest -q
bnn repro
```

Windows: compile with **MSVC x64** only. MinGW 32-bit → WinError 193.

## Checks before a PR / push

| Check | Command |
|-------|---------|
| Unit + golden gates | `pytest -q` |
| Compression microcheck | `bnn export-check` |
| Native (Windows) | `bnn validate-native` |
| Full fast repro | `bnn repro` |

Default CI is the **fast** path (`-m "not slow"` when slow tests exist).
Do not commit `data/`, `*.dll`, or checkpoints.

## Changelog

Add a bullet under `## Unreleased` in [`CHANGELOG.md`](CHANGELOG.md).

## Roadmap tasks

Pick an unchecked item from the canonical plan — [`ROADMAP.md`](ROADMAP.md)
(twin: [`docs/37_WORLD_CLASS_BNN_OPTIMISER_ROADMAP.md`](docs/37_WORLD_CLASS_BNN_OPTIMISER_ROADMAP.md)) —
with dependencies met; mark `[x]` when done (keep both files identical).

Historical lab COMPLETE checklist: [`docs/21_E2E_ROADMAP_COMPLETE_REPO.md`](docs/21_E2E_ROADMAP_COMPLETE_REPO.md).
