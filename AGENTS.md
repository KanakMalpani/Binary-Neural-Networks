# AGENTS.md — entrypoint for AI coding agents

You are reproducing or extending **Binary Neural Networks**
(https://github.com/KanakMalpani/Binary-Neural-Networks).

**Human-equivalent path:** follow [`docs/GUIDE_E2E.md`](docs/GUIDE_E2E.md)
(install → `bnn repro` → `bnn optimise` → encode/decode → modalities → metrics).
Prefer `bnn optimise` over legacy `bnn wrap --ultra` in new docs and demos.

## Do this in order (do not invent new benches)

1. `cd` to the repo root.
2. `python -m pip install -U pip`
3. `pip install -e ".[dev]" -c constraints.txt`
4. On Windows only (optional but preferred): `python -m bnn.kernels.compile_native`
5. `bnn repro`  
   — or — `python scripts/repro_all.py`
6. Confirm stdout ends with `REPRO: PASS` and exit code `0`.
7. Compare any new metrics only against:
   - `tests/golden_floors.json`
   - committed `results/*.json`
8. Read `REPRODUCIBILITY.md` for hardware notes and full regen.
9. **Thesis lock:** do not claim GPU 32× from `sign()`; packed CPU kernels + honest STE sim.
10. Do **not** commit `data/` datasets or force-push.
11. **When lost on product direction:** read root [`ROADMAP.md`](ROADMAP.md) (§0 → §7 → current phase → lowest unchecked TODO). Do not invent alternate benches.
12. **User / human path:** [`docs/GUIDE_E2E.md`](docs/GUIDE_E2E.md) — same conclusions as this file’s verify steps, written as a complete narrative.

## Modes

| Command | When |
|---------|------|
| `bnn repro` | Default fast verify (&lt; few minutes) |
| `bnn repro --mode full` | Short smoke trains → `results/_repro_smoke_*.json` |
| `bnn repro --mode full --overwrite-goldens` | Only if explicitly refreshing published goldens |

## Pass criteria (same conclusions)

- Compression **32×** exact (uint64 pack)
- Native GEMM **err = 0** when DLL present (else NumPy path err = 0 via pytest)
- MNIST / image / audio accuracies within floors in `golden_floors.json`
- Do not require bit-identical floats across machines

## Forbidden

- Inventing alternate benchmark shapes as “the” golden
- Skipping gates and declaring success from anecdotes
- Changing thesis / claiming fake GPU wins
- Committing huge datasets under `data/`
