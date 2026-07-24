# ADR 0001 — Public optimiser API

| Field | Value |
|-------|-------|
| **Status** | Accepted |
| **Date** | 2026-07-25 |
| **Task** | W1.T01 |
| **Deciders** | Lab maintainers |

## Context

The lab already has `wrap_model`, ultra-wrap demos, calibrate/QAT sketches, and
`.bnnpack` encode/decode. Outsiders still lack **one obvious entrypoint** with a
versioned report contract. World-class optimiser products (torchao, Optimum,
bitsandbytes) win on a stable surface — not on more MNIST tables.

## Decision

1. **Public Python API** lives under:
   - `bnn.optimise.optimise_model` — preferred product verb
   - `bnn.wrap` / `bnn.wrapper.wrap_model` — stable wrap primitives (re-exported)
2. **CLI:** `bnn optimise` runs the ultra hybrid/calib/QAT demo path and writes a
   versioned JSON report; optional `--pack` encodes Linear weights to `.bnnpack`.
3. **Report schema:** `bnn_optimise_report_v1` (see `bnn.wrap.schema`) with dual
   metrics: compression (theory) **separate** from latency (wall-clock).
4. **Semver:** breaking removals of symbols in `bnn.optimise.__all__` and
   `bnn.wrap.__all__` require a major bump + deprecation window
   (`docs/SEMVER_AND_DEPRECATION.md`).
5. **Thesis lock unchanged:** no GPU 32× from `sign()`; pack + popcount for speed.

## Consequences

- New tutorials teach `bnn optimise` / `optimise_model` first.
- Legacy `bnn wrap` (non-`--ultra`) remains; may emit `DeprecationWarning` pointing
  at `bnn optimise` when appropriate.
- Compatibility tests lock `__all__` exports (`tests/test_public_api.py`).

## Alternatives considered

| Option | Why not |
|--------|---------|
| Only scripts under `scripts/` | Not a product contract |
| Rename everything to `quantize_*` | Collides with INT8/INT4 mental model; we are binary/ternary pack |
| Freeze only CLI, no Python API | HF / library users need imports |

## Links

- `bnn/optimise.py`, `bnn/wrap/api.py`, `bnn/wrap/schema.py`
- `docs/SEMVER_AND_DEPRECATION.md`
- Tutorials: `docs/tutorials/07_OPTIMISER_QUICKSTART.md`, `08_HF_OPTIMISER.md`
