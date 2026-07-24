## Summary

<!-- 1–3 bullets: what & why (link ROADMAP task IDs when applicable, e.g. W1.T05) -->

-

## Thesis / honesty check

- [ ] No claim of GPU 32× (or e2e latency) from `sign()` / STE alone
- [ ] Dual metrics kept separate when discussing compression vs wall-clock
- [ ] No invented benchmark shapes; compared only to `tests/golden_floors.json` / committed `results/*.json`

## Repro checklist

- [ ] `pip install -e ".[dev]" -c constraints.txt` (already / in CI)
- [ ] `pytest -q` (or relevant subset) passes
- [ ] `bnn repro` → `REPRO: PASS` (required if kernels / wrap / codec / goldens touched)
- [ ] Windows: `python -m bnn.kernels.compile_native` if native path changed

## Docs / roadmap

- [ ] `CHANGELOG.md` Unreleased bullet (if user-visible)
- [ ] `ROADMAP.md` (+ twin `docs/37_…`) checkboxes updated if a task completed
- [ ] No `data/` datasets or secrets committed

## Test plan

- [ ]
