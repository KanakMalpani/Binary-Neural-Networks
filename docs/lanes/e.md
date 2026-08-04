# Lane E — Research / bridges

**Branch:** `lane/e-research`  
**Base:** `main` @ `5910978`  
**Date:** 2026-08-04

## Owned paths

- `scripts/bridges/**` (CLI wiring; Lane I may harden `llamacpp*`)
- `bnn/cli.py` — **bridge subcommands only** (+ minimal `--from-results` on pareto)
- `docs/23*`, `docs/24*`, `docs/32*`, `docs/PUBLICATION_PLAN.md`, `docs/02*` (W12.T04)
- `scripts/pareto_report.py`, `scripts/figure_from_results.py`
- Progress: this file

## Task status (integrator flips ROADMAP twin)

| ID | Task | Lane status |
|----|------|-------------|
| W12.T01 | Link local series in docs | `[x]` docs/32 vault table |
| W12.T02 | Publication plan + claims whitelist ↔ goldens | `[x]` PUBLICATION_PLAN.md polished |
| W12.T03 | Figure pipeline from `results/*.json` | `[x]` `figure_from_results.py` + `pareto --from-results` |
| W12.T04 | Related work table maintenance | `[x]` docs/02 last-reviewed + positioning |
| W12.T05 | Novel candidates triage ship/defer | `[x]` docs/32 triage table |
| — | First-class `bnn bridge …` CLI | `[x]` list / gpu / cpu-llm / figures |

## Acceptance

- `bnn bridge list` / `bnn bridge gpu` / `bnn bridge cpu-llm` exit 0
- `bnn bridge figures` exit 0 against committed goldens (no invented shapes)
- `bnn pareto --from-results` builds from committed `results/*.json`
- Focused tests: `tests/test_bridge_cli.py`

## Residuals

- Venue LaTeX / arXiv submit still optional (plan only).
- Lane I may further harden `scripts/bridges/llamacpp*` + pins; E only wires CLI + docs/23.
- PNG plots need optional `matplotlib`; JSON manifests always work.
- Integrator: apply checkbox flips in ROADMAP twin from this file.
