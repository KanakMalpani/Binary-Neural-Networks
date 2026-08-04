# Lane KG — integrity / enrichment

| Field | Value |
|-------|-------|
| **Branch** | `lane/kg-enrich` |
| **Base** | `main` @ `f40f65e` |
| **Status** | Integrity enrichment PR (Wave 1 lanes A–I remain open) |
| **Date** | 2026-08-04 |

## Owned paths

- `knowledge_graph/**` (JSON + GraphML + enrichment integrity overlay + VIEW/GAPS notes)
- `scripts/apply_kg_integrity.py`, `scripts/build_bnn_kg.py` (source honesty), `scripts/merge_kg_enrichment.py` (`same_as` map)
- `bnn/cli.py` (`bnn kg`), `bnn/kg/` (unchanged API)
- `.github/workflows/ci.yml` (kg validate gate)
- `docs/44_KNOWLEDGE_GRAPH.md`, `docs/lanes/kg.md`, `mkdocs.yml` nav pointer
- `tests/test_kg.py` (integrity extras)

## What this lane does

1. Fix broken `sources[]` (ternary path, WASM/BITNET pending-PR honesty, FBI non-path).
2. Add WC-O / recommend / eval-suite / sys_kg / GPTQ / Q-Sparse / BitDistiller / RAPL Result / moonshot non-goals.
3. Repair over-aliased `same_as` → proper relations.
4. CI gate: `kg_validate` + `tests/test_kg.py`.
5. Tiny CLI + doc cross-links so agents find recommend/eval/KG.

## ROADMAP proposals (integrator — do not twin-edit here)

| Item | Suggestion |
|------|------------|
| Wave 2 KG sync after lane merges | Flip OpenGap statuses from `open_pr` → `established`/`closed` when A–I land |
| PyPI Trusted Publisher | Remains human gate (`gap_pypi_trusted`) |
| WC-O closed claim | Only after Lane A merge + measured dual metrics |

## Residuals (deliberately not done)

- Do **not** merge Wave 1 PRs #16–#24 in this PR.
- Do **not** invent golden floors / ImageNet SOTA / GPU 32× from `sign()`.
- Do **not** land Lane H energy tree on main here — Result node cites PR #22 + main `energy_bound`.
- WASM_SIMD.md / BITNET_PIN.md stay branch-local until F/I merge; KG points at PRs + docs that exist on main.
