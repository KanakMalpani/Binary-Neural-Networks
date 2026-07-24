# Changelog

## 0.3.0 — 2026-07-25

### Phase C — multi-arch & eval

- **Linux native CI hard gate:** GCC `.so` compile + `validate-native` must pass
  (no soft `continue-on-error` on Linux).
- **Python 3.11–3.13** CI matrix job (`linux-py-matrix`).
- **Pareto JSON** schema `bnn_pareto_report_v1` + `bnn pareto` /
  `scripts/pareto_report.py` (+ optional matplotlib plot).
- **Fair eval protocol** `docs/FAIR_EVAL_PROTOCOL.md`; leaderboard template.
- **W3.T05** layer-wise sensitivity (`bnn.wrap.sensitivity`) + optional
  `OptimiseConfig.sensitivity`.
- **ARM NEON / AVX-512** documented spike/moonshot notes under `docs/spikes/`.
- Flamegraph howto `docs/FLAMEGRAPH_HOWTO.md`; macOS notes; recipes index.

### Phase D — launch hygiene

- Version **0.3.0**; annotated tag + GitHub Release.
- SBOM script `scripts/generate_sbom.py` + `docs/SBOM.md`.
- PyPI prep checklist `docs/PYPI_PUBLISH.md` (dry-run; no auto-upload).
- Launch checklist `docs/LAUNCH_CHECKLIST.md` (Discussions = manual).
- Moonshot deferrals `docs/MOONSHOT_DEFERRALS.md` (ONNX / ImageNet / RAPL / …).

### Docs

- Session report `docs/40_ROADMAP_E2E_SESSION.md`; execution log append.
- Publication plan + `.bnnpack` v2 design sketch (not implemented).

## 0.2.0

### Docs — master E2E User Guide

- **`docs/GUIDE_E2E.md`** primary user guide (zero → optimiser results); linked from
  README, `docs/README.md`, `AGENTS.md`.
- Tutorials 01–08 cross-linked; prefer `bnn optimise` over `bnn wrap --ultra`.
- Completion note: `docs/39_GUIDE_E2E_COMPLETION.md`; ROADMAP W9.T08/T09 done.

### Fix — NumPy 1.x popcount

- **`bnn.kernels.popcount.bitwise_count`:** LUT fallback when `np.bitwise_count`
  is missing (NumPy &lt; 2.0). Restores `bnn repro` on NumPy 1.26.

### Phase A/B — optimiser product + OSS hygiene

- **LICENSE** (MIT) at repo root; `SECURITY.md`, `CODE_OF_CONDUCT.md`,
  `.github` issue/PR templates, `CODEOWNERS`, `CITATION.cff`, `MODEL_CARD.md`.
- **Public optimiser API:** `bnn.optimise.optimise_model` + ADR
  `docs/adr/0001_public_optimiser_api.md`; semver policy
  `docs/SEMVER_AND_DEPRECATION.md`; report schema `bnn_optimise_report_v1`.
- **CLI:** `bnn optimise` (preferred over `bnn wrap --ultra`); legacy wrap warns.
- **Tutorials:** `07_OPTIMISER_QUICKSTART`, `08_HF_OPTIMISER`; optional
  `tests/test_hf_optimiser.py` (`hf`/`slow`).
- **DX:** MkDocs stub (`mkdocs.yml` + ADR 0002), zoo registry
  `bnn/zoo_registry.json`, dataset cards, compatibility matrix.
- **Tests:** `tests/test_public_api.py` locks public exports.

### World-class optimiser roadmap

- **Canonical plan:** root `ROADMAP.md` + twin `docs/37_WORLD_CLASS_BNN_OPTIMISER_ROADMAP.md`
  (audit scorecard, W1–W14 workstreams, phases A–F, agent protocol).
- Pointers updated: `docs/21` (historical COMPLETE), `docs/10`, README, `docs/README`,
  `CONTRIBUTING`, `AGENTS.md`.

### Encoder / Decoder + codec (next lane)

- **Seq models:** `bnn.seq` — Binary Transformer Encoder/Decoder, Seq2Seq reverse
  task, BinaryAutoEncoder; CLI `bnn train-seq2seq`.
- **Weight codec:** `bnn.codec` + `.bnnpack`; CLI `bnn encode` / `bnn decode`
  (32× pack, GEMM round-trip err=0).
- **Wrap lane:** `bnn wrap-transformer` → `results/tiny_transformer_wrap.json`.
- **Profile:** `bnn profile` pack/act/gemm/overhead vs FP32.
- **Bridges:** `scripts/bridges/torchao_int4_recipe.py`,
  `scripts/bridges/llamacpp_bitnet_recipe.py`.
- **Docs:** `docs/36_ENCODER_DECODER_AND_NEXT.md`, tutorial 06.

### Quality upgrade (0.2.0)

- **Repro:** `bnn repro` / `scripts/repro_all.py`, `REPRODUCIBILITY.md`, `AGENTS.md`,
  golden floors v2 + `tests/test_golden_gates.py`, `bnn.determinism.set_repro_seed`.
- **Packaging:** version `0.2.0`, keywords/classifiers/urls, extras, `constraints.txt`,
  `python -m bnn`, CLI `--version` / `bnn version`, polished help epilog.
- **DX / safety:** `bnn.paths` traversal guards, safer `torch.load`, CIFAR batch
  validation, fail-loud `validate_native` (exit 2 if no DLL), packed shape/dtype checks.
- **Tests/CI:** CLI/paths/determinism smokes; pip cache; Windows+Linux repro gates;
  pytest markers `slow` / `native`.
- **Docs:** README rewrite, `docs/README.md` index, accurate API reference, SUMMARY
  honesty table (theory vs wall-clock), `docs/30` + `docs/31`.

## 0.1.0

- Packaging: `pyproject.toml`, console script `bnn`, editable install.
- Tests: pytest suite (STE, pack, native GEMM, wrapper, golden floors).
- CI: GitHub Actions Windows + Linux.
- CLI: compile/validate/bench/train/wrap/eval-suite/recommend.
- Export + eval_report SUMMARY regenerator.
- Docs: tutorials, HF/GGUF/BitNet/GPU guides, one-pager.
- Wrapper: `wrap_model` policies (`hybrid_ffn`, `all_large_linear`).
- Image + audio modality lanes with measured results.
