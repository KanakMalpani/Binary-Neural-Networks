# Changelog

## Unreleased

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
