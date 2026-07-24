# Semver & deprecation policy

**Task:** W1.T02 · **Related ADR:** [`docs/adr/0001_public_optimiser_api.md`](adr/0001_public_optimiser_api.md)

## Version source of truth

- Package version: `bnn/_version.py` and `pyproject.toml` (keep in sync).
- CLI: `bnn version` / `bnn --version` must match.
- Git tags: `vMAJOR.MINOR.PATCH` for releases (see ROADMAP release checklist).

## What is “public API”

**Stable (semver-protected):**

- `bnn.optimise`: `optimise_model`, `OptimiseResult`, `OptimiseConfig`
- `bnn.wrap` / `bnn.wrapper`: symbols listed in `__all__`
- Report schema id: `bnn_optimise_report_v1` field names documented in
  `bnn.wrap.schema`
- CLI verbs documented in README: `repro`, `optimise`, `encode`, `decode`, …

**Internal / best-effort:**

- `scripts/*` demos (may change without major bump)
- Private helpers (`_foo`), undocumented kwargs
- Native DLL ABI (rebuild via `compile_native`; not a stable C API yet)

## Bump rules

| Change | Bump |
|--------|------|
| Bugfix, docs, new optional kwargs with defaults | PATCH |
| New public symbol, new CLI verb, new schema *fields* (backward compatible) | MINOR |
| Remove/rename public symbol; change meaning of required report fields; drop Python version | MAJOR |

## Deprecation process

1. Mark in docs + `CHANGELOG` **Unreleased** / next MINOR.
2. Emit `DeprecationWarning` for ≥ **one MINOR** release (or 60 days, whichever longer).
3. Remove only in a **MAJOR** bump.
4. Prefer aliases (`optimise` → ultra wrap) over silent behavior changes.

## Dual-metric honesty (non-negotiable)

Changing how we *label* compression vs latency in user-facing text is allowed in
PATCH/docs; claiming e2e 32× from theory alone is **forbidden** at any version
(thesis lock — not a semver concern, a project invariant).
