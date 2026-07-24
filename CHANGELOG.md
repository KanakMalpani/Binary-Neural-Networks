# Changelog

## Unreleased

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
