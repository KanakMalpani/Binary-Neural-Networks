# Roadmap execution log — 2026-07-25

Session goal: complete **Phase A (API freeze / OSS hygiene)** and push into
**Phase B (HF optimiser UX)** per root `ROADMAP.md`.

## Tasks completed

### Phase A — API freeze

| ID | Result |
|----|--------|
| W1.T01 | ADR `docs/adr/0001_public_optimiser_api.md` |
| W1.T02 | `docs/SEMVER_AND_DEPRECATION.md` |
| W1.T03 | Frozen `__all__` on `bnn.wrap` + `docs/api` |
| W1.T04 | `tests/test_public_api.py` |
| W1.T05 | CLI `bnn optimise` |
| W1.T06 | Schema `bnn_optimise_report_v1` in `bnn/wrap/schema.py` |
| W1.T07 | DeprecationWarning on legacy `bnn wrap` (non-ultra) |

### Phase A / D — OSS hygiene (pulled forward)

| ID | Result |
|----|--------|
| W11.T01 | Root `LICENSE` (MIT) |
| W11.T02–T05 | Issue templates, PR template, CODEOWNERS, COC |
| W10.T02 | `SECURITY.md` |
| W10.T01 / T04 | `MODEL_CARD.md` (+ ethics notes) |
| W11.T10 | `CITATION.cff` |
| W11.T09 | CONTRIBUTING already pointed at ROADMAP |

### Phase B — HF / UX

| ID | Result |
|----|--------|
| W9.T01 | `docs/tutorials/07_OPTIMISER_QUICKSTART.md` |
| W5.T03–T04 | Tutorial 08 + `tests/test_hf_optimiser.py` (`hf`/`slow`) |
| W9.T03 | README + `docs/README` synced |
| W4.T08 | `bnn/zoo_registry.json` |
| W6.T04 | `docs/DATASET_CARDS.md` |

### Extra (Phase C / DX started)

| ID | Result |
|----|--------|
| W2.T02 | Linux GCC `.so` path already in `compile_native` — marked done |
| W2.T03 | Linux CI compile + validate (soft `continue-on-error`) |
| W7.T02 | `docs/BENCH_SHAPES.md` |
| W9.T05 / T07 | MkDocs stub + ADR index |
| W14.T01 | `docs/COMPATIBILITY_MATRIX.md` |

## Still open (world-class 1.0)

- **Phase C:** W2.T04 ARM NEON; harden Linux native CI (remove soft fail); W8.T03 py matrix; W7.T03–T05 Pareto + fair protocol; W13.T02 flamegraph howto
- **Phase D:** W8.T05–T08 Releases / SBOM / PyPI; W11.T06 Discussions (manual); W11.T07 launch checklist; W10.T05 pip-audit
- **Phase E:** W12 publication plan / figures; paper alignment
- **Phase F / moonshots:** `.bnnpack` v2, safetensors, ONNX, AVX-512, WASM, RAPL, ImageNet protocol runner, bitnet.cpp submodule
- **W3.T05–T06** layer-wise sensitivity search (can slip)
- **WC-\*** gates in ROADMAP §1 not all green yet — still **lab / beta optimiser**

## Blockers / deferred with reason

| Item | Reason |
|------|--------|
| W11.T06 Enable Discussions | Manual GitHub settings |
| W8.T05 Tagged release | Needs human tag decision (`v0.3.0` checklist mostly green except CHANGELOG+tag) |
| W2.T04 / T05 ARM / AVX-512 | Needs target hardware |
| W5.T05–T07 codec v2 / safetensors / ONNX | Design spike after API freeze |
| Full ImageNet / RAPL / bitnet submodule | Explicit moonshots / non-goals for gate |

## Gates

Run after this session:

```bat
pytest -q -m "not slow and not hf"
bnn repro
```

Expect `REPRO: PASS`.

## Files of note

- `bnn/optimise.py`, `bnn/wrap/schema.py`, `bnn/cli.py`
- `LICENSE`, `SECURITY.md`, `CODE_OF_CONDUCT.md`, `MODEL_CARD.md`, `CITATION.cff`
- `.github/ISSUE_TEMPLATE/*`, `.github/PULL_REQUEST_TEMPLATE.md`, `.github/CODEOWNERS`
- `docs/38_ROADMAP_EXECUTION_LOG.md` (this file)

---

# Session 2 — Phase C+D → v0.3.0 (2026-07-25)

## Tasks completed

| ID | Result |
|----|--------|
| W2.T03 / W8.T10 | Linux native CI **hard** gate |
| W8.T03 / W14.T02 | Python 3.11–3.13 CI matrix |
| W7.T03–T05 / T07 | Pareto JSON + fair protocol + leaderboard template + plot |
| W3.T05 | Layer-wise sensitivity API + optional optimise flag |
| W2.T04 / T05 | ARM NEON / AVX-512 spike & moonshot notes |
| W13.T02 | Flamegraph howto |
| W14.T05 | macOS notes |
| W6.T06 | Recipes index |
| W8.T05–T06 | v0.3.0 release path + SBOM script |
| W8.T08 | PyPI prep docs (no upload) |
| W10.T05 | Soft pip-audit CI job |
| W11.T07 | Launch checklist |
| W11.T06 | Documented manual Discussions step |
| W5.T07 | ONNX explicit defer |
| W5.T05 | `.bnnpack` v2 design sketch |
| W12.T02 | Publication plan draft |

## Gates

See `docs/40_ROADMAP_E2E_SESSION.md`. Still **not** claiming WC v1.0.

## Verify

```bat
pytest -q -m "not slow and not hf"
bnn repro
```
