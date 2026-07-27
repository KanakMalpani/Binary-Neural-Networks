# Roadmap E2E session report — v0.3.0 (2026-07-25)

**Verdict:** Phase C substantially done; Phase D release hygiene done (`v0.3.0`);
non-moonshot in-repo TODOs flipped. Still **lab / beta optimiser** for full WC-§1
**v1.0** — see remaining below.

## Phases

| Phase | Status | Notes |
|-------|--------|-------|
| A API freeze | Done (prior) | `bnn optimise`, schema v1, ADR |
| B HF UX | Done (prior) | Tutorials 07–08, GUIDE_E2E |
| C Multi-arch / eval | **Done this session** | Linux native hard CI, py matrix, Pareto, fair protocol, sensitivity, NEON/AVX notes |
| D Launch hygiene | **Done this session** | Tag/Release `v0.3.0`, SBOM, PyPI prep, launch checklist |
| E Research | Partial | Publication plan draft; figures via Pareto plot |
| F Ecosystem | Partial | ONNX explicit defer; v2 design sketch; leaderboard template |

## World-class gates (§1)

| Gate | Status |
|------|--------|
| WC-A1–A3 | **Met** |
| WC-K1–K2 | **Met** |
| WC-K3 | **Met** (Win+Linux+macOS/ARM native via portable SIMD; NumPy fallback remains) |
| WC-K4 | **Met** (Pareto + fair protocol) |
| WC-O1–O4 | Partial (sensitivity yes; full search/QAT recipe polish open) |
| WC-R1 | **Met** |
| WC-R2–R4 | Partial (matrix+SBOM+LICENSE; PyPI upload / hard audit open) |
| WC-D1 | **Met** |
| WC-D2–D5 | Partial (API stub; Discussions/v1.0 tag open) |
| WC-P1–P2 | Partial |

## Deliverables (this session)

- `.github/workflows/ci.yml` — `linux-native` hard, `linux-py-matrix`, soft supply-chain
- `bnn/eval/pareto.py`, `bnn pareto`, `scripts/pareto_report.py`
- `bnn/wrap/sensitivity.py` + `OptimiseConfig.sensitivity`
- Docs: FAIR_EVAL, SBOM, PYPI_PUBLISH, LAUNCH_CHECKLIST, FLAMEGRAPH, MACOS,
  MOONSHOT_DEFERRALS, spikes/*, PUBLICATION_PLAN, BNNPACK_V2_DESIGN, RECIPES_INDEX,
  LEADERBOARD_TEMPLATE
- Version bump **0.3.0**; ROADMAP twin synced

## Remaining for true v1.0

> **Supersession (2026-07-28):** portable SIMD + portability CI closed W2.T04/T05
> and macOS/ARM native — see [`41_PORTABLE_SIMD_KERNEL.md`](41_PORTABLE_SIMD_KERNEL.md).
> Item 1 below is historical; do not treat as open.

1. ~~macOS CI and/or ARM NEON native~~ **DONE** (`docs/41` + `portability` CI)
2. PyPI Trusted Publishing + real upload
3. Harden pip-audit / attestations
4. Autodoc MkDocs site (W9.T06)
5. Full layer search W3.T06 + stronger QAT demo (WC-O)
6. ~~Enable Discussions (human)~~ **done**
7. Optional: safetensors, WASM, RAPL, ImageNet protocol runner — moonshots

## Verify

```bat
pytest -q -m "not slow and not hf"
bnn repro
```

Expect `REPRO: PASS`.
