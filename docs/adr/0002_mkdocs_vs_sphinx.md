# ADR 0002 — MkDocs vs Sphinx for API docs

| Field | Value |
|-------|-------|
| **Status** | Accepted |
| **Date** | 2026-07-25 |
| **Task** | W9.T05 |

## Context

World-class DX needs browsable API docs beyond `docs/api/README.md`. Historical
lab docs (`docs/21`) preferred plain Markdown; we revisit for the optimiser product.

## Decision

1. Use **MkDocs Material** as the site generator (simple, MD-native, matches existing
   `docs/*.md` corpus).
2. Keep research series as Markdown pages; add `mkdocs.yml` stub + `docs/api/` pages.
3. Autodoc (W9.T06) via `mkdocstrings` in a follow-up — not blocking Phase A/B.
4. Sphinx remains an option if type-heavy API growth demands it; revisit at v1.0.

## Consequences

- Stub config lands in repo root: `mkdocs.yml` (nav: Home, Tutorials, API, Roadmap).
- GitHub Pages deploys from `.github/workflows/pages.yml` (`mkdocs build --strict`, same `mkdocs.yml`). First live URL needs Settings → Pages → Source = GitHub Actions.
- No requirement that agents build the site for `bnn repro`.

## Alternatives

| Option | Why not now |
|--------|-------------|
| Sphinx + autodoc only | Heavier; existing docs are MD-first |
| No site, README only | Fails WC-D2 |
