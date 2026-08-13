# Changelog

## Unreleased

- Docs: first-principles 90-day transformation plan (`docs/TRANSFORMATION_PLAN.md`) — proposal only; does not flip ROADMAP WC checkboxes or invent goldens.

## [1.0.0] — 2026-08-05

### Wave 2 integrator + Wave 3 v1.0

- Merged lanes A–I + KG enrichment: WC-O (calibrate/distill/BN fuse), codec v2 +
  safetensors, docs/eval/safety, research bridges + `bnn bridge`, PyPI packaging
  dry-run/attestations, WASM pedagogy, ResNet-BiReal + ImageNet protocol runner,
  RAPL/energy-proxy, bitnet.cpp recipe pins (no giant submodule).
- `OptimiseConfig.fuse_bn` / `distill_steps` wired into `bnn/optimise.py` with
  report fields `fuse` / `distill`.
- ROADMAP twin flipped from lane notes; moonshot deferrals refreshed honestly.
- **Human residual:** PyPI Trusted Publisher registration for `bnn-lab` (see
  `docs/PYPI_PUBLISH.md`). No API-token publish path.

### Packaging — PyPI distribution name `bnn-lab` (W8.T08)

- The short name `bnn` is **already taken** on PyPI (unrelated package). Distribution
  name is now **`bnn-lab`**; import path and console script remain `bnn`.
- GitHub Environment `pypi` created for Trusted Publishing; `docs/PYPI_PUBLISH.md`
  rewritten with the one-time pypi.org publisher steps. Live upload still needs
  that registration, then Actions → wheels → publish=true.
- Wheels dry-run + attestations verified; first OIDC upload pending human publisher.

### Memory footprint report (W13.T05) + arena decision (W2.T07)

- **`bnn memory` / `bnn.memory.memory_report`** — per-layer footprint that keeps
  **resident** (measured from real buffers) separate from **theoretical** (the
  encoding's pack ratio). `TernaryWeightOnlyLinear` stores int8 while encoding
  2 bits; reporting only the theoretical figure would claim a saving no machine
  observes. Also reports a whole-model ratio that includes the FP embeddings /
  attention / norms that were deliberately never wrapped.
  Example (512→2048→512, `binary_xnor`): resident **29.68x**, theoretical 32.00x —
  the gap is per-channel `alpha` and FP32 `bias`. Whole-model with a real
  embedding table: **2.07x** vs 3.33x tracked.
- `forward_transient_bytes` sizes the per-forward buffers. Activation packing is
  32x smaller than FP32 activations and nearly free; the **FP32 output**
  dominates transient memory, which is the buffer to plan edge deployments
  around.
- **W2.T07 memory arena: measured, then declined.** Output allocation is
  **1.4–1.8%** of kernel time (9.74 µs vs 694.93 µs at 64x4096x4096). Against
  that, `torch.from_numpy(np.ascontiguousarray(y))` *aliases* the NumPy buffer,
  so recycling it would silently corrupt tensors the caller still holds — a
  data-corruption bug for ~1.5%. Recorded in `docs/43_MEMORY_FOOTPRINT.md` with
  the threshold for revisiting, so nobody has to re-derive it.

### Ternary cross-ISA parity (W2.T09)

- The ternary kernel has its own per-ISA `popcount_and`, but only the *binary*
  path was tested across scalar / AVX2 / AVX-512 / NEON — a SIMD-only ternary
  bug could have shipped silently. Added parity tests over six shapes covering
  every vector-width remainder: all paths are **bit-identical to each other** and
  to the NumPy path, and match the dequantised FP reference within the float32
  scale rounding. No bug found; the gap was in coverage.
- Also covers negative scale, all-zero ternary rows, and precomputed vs
  on-the-fly `pop_p`/`pop_n`.

### Performance — encoder / decoder / STE

- **Fused attention (SDPA).** `MultiHeadAttention` and `CrossAttention` now use
  `F.scaled_dot_product_attention` instead of a manual `q @ k.T` → softmax →
  `@ v`. The `(B, H, T, T)` score matrix is no longer materialised, and the
  causal path needs no mask, so nothing is allocated per forward (the old code
  rebuilt `torch.triu(torch.ones(T, T))` on every call). Verified equal to the
  previous computation to ~5e-7 (float32 rounding) on both causal and
  non-causal paths.
- **Sign is 4x cheaper to allocate.** `torch.where(x > 0, ones_like(x),
  -ones_like(x))` allocated four full tensors; `x.gt(0).to(x.dtype).mul_(2).sub_(1)`
  allocates one. **Bit-identical** — including `0.0`, `-0.0`, `±inf`, and
  float16/32/64 — and 1.4–1.65x faster on its own. Applied to all five call
  sites (`SignSTE`, `ApproxSignSTE`, `TanhSoftSTE`, `sign_pm1`, packed conv).
  This is the hottest op in every binary layer: it was ~22% of encoder time.

Combined, measured on this machine (min-of-5, same process):

| case | before | after | speedup |
|---|---|---|---|
| encoder T=64 / 128 / 256 | 5.86 / 8.18 / 16.54 ms | 4.71 / 6.67 / 11.72 ms | 1.25 / 1.23 / **1.41x** |
| decoder (causal) T=64 / 128 / 256 | 6.35 / 9.35 / 19.26 ms | 4.61 / 6.68 / 12.76 ms | 1.38 / 1.40 / **1.51x** |
| seq2seq (cross-attn) T=64 / 128 | 14.97 / 21.89 ms | 10.80 / 16.55 ms | 1.39 / 1.32x |
| codec encode / roundtrip | 4.98 / 9.19 ms | 3.98 / 7.34 ms | 1.25 / 1.25x |

`train_seq2seq` still reaches `eval_token_acc = 1.0`, and all repro gates pass.

### Optimiser — per-layer mode search (W3.T06) + QAT recipe (W3.T07)

- **`bnn.wrap.search_layer_modes`** picks binary / ternary / skip **per layer**
  to maximise theoretical compression subject to a measured cosine floor. It
  starts maximally aggressive and greedily relaxes the layer costing the most
  quality, re-measuring the *whole* model each step — per-layer scoring cannot
  see layer interactions. `O(L)` probes per relaxation, not `3**L`.
- Measured trade-off is monotonic (enforced by test — a search that reported
  more compression at higher quality would be lying): floor 0.00 → 32.0x at
  cosine 0.27; floor 0.90 → 1.71x at 0.950; floor 0.999 → 1.00x at 1.000.
- **`docs/42_QAT_AND_LAYER_SEARCH.md`** — runnable QAT recipe with the knobs
  that actually matter (steps, teacher, lr, target layers, real calibration
  data) and the correct order of operations: search *before* QAT, so training
  is not spent on layers the search would skip.

### Docs — autodoc site (W9.T06)

- `mkdocstrings` wired into `mkdocs.yml`; seven generated API pages covering
  kernels (incl. runtime dispatch), optimiser, wrapping, layers/STE,
  encoder/decoder, codec, and reporting. **`mkdocs build --strict` passes.**
- `docs = [...]` extra added; new `docs` CI job builds the site and fails on a
  renamed symbol rather than silently emptying a page.
- `tests/test_docs_links.py` enforces repo-wide link integrity, mkdocs nav
  targets, that every `:::` autodoc reference resolves, and that everything in
  `bnn.__all__` appears in the API docs. This is what lets MkDocs skip
  `not_found` for the root-level docs (README / ROADMAP / REPRODUCIBILITY /
  AGENTS) that live outside `docs_dir` by design.
- Fixed a broken `docs/GUIDE_E2E.md` link in the roadmap (the file already
  lives in `docs/`, so the prefix doubled up).

### Supply chain (W8.T07 / W10.T05)

- **Build provenance attestations** on every wheel and the sdist via
  `actions/attest-build-provenance`, so a consumer can run
  `gh attestation verify bnn-*.whl --repo ...`. Skipped on forks, which cannot
  mint an OIDC token.
- **pip-audit is now a hard gate** — but scoped to `requirements.txt` (the
  shipped dependency set) rather than the whole environment, because failing the
  build on pip-audit's own transitive deps is noise, not security. Any advisory
  not on the explicit triaged list fails CI; the two current ignores each carry
  the reason and the condition for removal. A report-only full-environment sweep
  still runs so new findings stay visible.

### CI reliability

- **Fast tests can no longer touch the network.** `tests/conftest.py` blocks
  socket creation for any non-`slow`/`hf` test, with an `allow_network` opt-out
  and a meta-test proving the guard fires. A truncated CIFAR download on the
  macOS runner is what last turned CI red; this makes the fix structural rather
  than conventional.
- **CIFAR download hardened**: 60s timeout (`urlretrieve` accepts none, so a
  stalled socket hung the runner), atomic `.part` + `os.replace` (a partial file
  could otherwise be cached as a valid archive), exponential backoff, and exact
  **MD5 verification** — a size floor cannot tell truncated from corrupt.
- `tarfile.extractall` now uses `filter="data"` with a manual member-validation
  fallback for Python 3.11 patch levels that lack it (CVE-2007-4559).
- Real-CIFAR vision smoke restored as a `slow` test that **skips** on an
  unreachable CDN instead of failing.

### CLI — in-process script dispatch + handler coverage

- ``bnn.cli`` runs ``scripts/*.py`` in-process via small helpers
  (``_script_path`` / ``_load_script_module`` / ``_call_script_main``): basename
  jail, failed-import cache cleanup, ``sys.argv`` rewrite for ``parse_args()``
  mains, exceptions → exit 1. ``_ultra_wrap_extra`` forwards ``--mode`` as
  chosen (policy=auto no longer clobbers an explicit mode).
- ``bnn optimise --pack`` uses in-process ``cmd_encode``.
- ``tests/test_cli_handlers.py``: exact argv tables + ``sys.argv`` bridge tests;
  ``bnn/cli.py`` coverage ~92%.

### OSS hygiene (GitHub settings)

- Repo About description + topics; Discussions enabled; ``main`` branch protection
  (required checks: quality / windows / linux-native; no force-push/deletes).

### Kernel — portable runtime SIMD dispatch (closes W2.T04 + W2.T05)

- **Runtime ISA dispatch** in `bnn/kernels/binary_gemm.c`: AVX-512 VPOPCNTDQ →
  AVX2 (`vpshufb` nibble LUT) → ARM64 NEON (`vcntq_u8`) → scalar. Detection uses
  `cpuid` **and** `xgetbv` (OS must have enabled YMM/ZMM state). No
  `-march=native` — one build stays valid on any CPU of the same architecture.
  AVX-512 is used when present, never required.
- **Single OpenMP region + 4-row batch blocking.** The old kernel forked a
  parallel team per batch row and re-streamed all of `W` `B` times. Aggregate
  **5.1×** faster over 12 shapes (up to 19.9× on small-N/large-batch).
- **Fused `alpha`/`bias` epilogue** (`binary_gemm_u64_scaled`): the wrapper's
  `y *= alpha; y += bias` NumPy passes cost as much as the vectorised GEMM.
- New Python API: `kernel_name()`, `cpu_features()`, `available_kernels()`,
  `set_kernel()`; `BNN_KERNEL=scalar|avx2|avx512|neon` env override.
- ARM NEON and AVX-512 spike notes moved from *deferred* to **delivered**.
- Full write-up: `docs/41_PORTABLE_SIMD_KERNEL.md`.

### Fixed

- **`fp32_gemm` timed its own memcpy.** It called `.astype(np.float32)` on
  already-float32 inputs; `astype` copies unconditionally, so the FP32 baseline
  included ~64 MB of copying per call and **every published "vs FP32" speedup was
  ~2× overstated**. Now uses `asarray`. `results/benchmark.json` was measured
  against the old baseline and needs regeneration before its ratios are quoted.
- **`_pack_activations_fast` was the slowest step in the forward pass.** It
  expanded the batch to a `(B, words, 64)` uint64 temporary; now delegates to
  `pack_binary_pm1` (`np.packbits`) — bit-identical, ~6.5× faster.
- `profile_packed_linear` added `bias` unconditionally in its warmup loop while
  the timed loop guarded for `None` — latent `TypeError` for bias-free layers.
- `scripts/validate_native.py` benchmark closures captured loop variables
  (`B023`) instead of binding them.
- `tests/test_codec.py` had an assertion neutralised by `or True`, and used
  `assert False` for exception flow (removed under `python -O`).

### Packaging — prebuilt wheels (no compiler required)

- `setup.py` builds the kernel into the wheel, with an OpenMP → single-threaded
  → no-native fallback ladder. **An install never fails because of the C
  kernel**; worst case you get the correct NumPy path.
- `[tool.cibuildwheel]` + `.github/workflows/wheels.yml` (`workflow_dispatch`,
  and automatic on `v*` tags) build Linux/macOS/Windows × x86-64/arm64 wheels,
  plus an sdist that is verified to compile from source.
- `scripts/check_wheel_kernel.py` — post-install smoke test that loads the
  *shipped* binary and asserts err=0 on every ISA path. NumPy-only, so wheel
  testing does not drag in torch.
- PyPI publishing is a separate job that only runs on an explicit manual
  dispatch with `publish=true`, via trusted publishing (no stored token).
- The loader now also accepts ABI-tagged filenames
  (`_binary_gemm_native.cpython-312-x86_64-linux-gnu.so`) as shipped in wheels.

### Quality

- **Ruff** configured and enforced (hard CI gate); repo is clean.
- **mypy** is now a **hard CI gate** — 51 errors → 0. Real fixes, not silencing:
  PyTorch buffers declared on the module classes, `_try_load_native()` returns a
  plain `CDLL | None` instead of a `CDLL | bool | None` sentinel that leaked
  into every caller, `nn.Linear`/`TernaryLinear` union attributes declared, and
  a `str`/`list[str]` collision in the MSVC builder untangled.
  `bnn/py.typed` added so downstream users get the types.
- **Coverage 66% → 80.2%, now a hard CI gate** (`--cov-fail-under=80`), up from
  143 to **380 tests**. New suites cover the previously untested surface:
  `eval_report` (0% → covered), `wrap/qat` (8%), `wrap/calibrate`, `wrap/schema`,
  `wrap/guardrails`, `export`, `data` (IDX parsing via synthetic files, no
  network), `kernels/compile_native` (toolchain decisions, monkeypatched),
  `paths`, `logutil`, `determinism`, `audio/features`, the packed ternary /
  dequant / conv modules, and the full CLI subcommand surface.
- Tests assert behaviour rather than lines: kernel input contracts fail fast on
  dtype/shape mismatch, `set_num_threads` cannot change results, every ISA path
  stays bit-identical, and the wrap report's dual-metric honesty rule is
  exercised in both directions.
- **Dependabot** (actions + pip, with torch/numpy pinned out since they gate the
  numeric goldens), **CodeQL** (python + c-cpp), **OpenSSF Scorecard**,
  `.editorconfig`, and README status badges.
- **Fixed two malformed issue templates.** `bug_report.yml` began a scalar with
  a backtick and `config.yml` contained an unquoted `REPRO: PASS` — both are
  YAML errors, so GitHub was silently rejecting those forms.
- **All 28 action references pinned to commit SHAs** (with the release tag as a
  trailing comment), resolved via the GitHub API rather than hand-copied. Tags
  are mutable; this is the single largest OpenSSF Scorecard penalty. Dependabot
  is configured to bump the pins.
- **Workflows are now linted with `actionlint`**, which caught a build-breaking
  error: `macos-13` has been **retired** by GitHub, so both Intel-macOS jobs
  would have failed with "no runner matching labels". Moved to `macos-15-intel`.
- `cibuildwheel` was pinned to `v2.21.3` while upstream is **4.1.1** — two
  majors stale, and 3.x+ validates `[tool.cibuildwheel]` strictly. Bumped and
  the config verified locally with `--print-build-identifiers` on all three
  platforms. Target selection moved into `pyproject.toml` as the single source
  of truth so it cannot drift from a `CIBW_BUILD` env var.
- **`pip install --no-binary :all:` in the sdist job** would have forced source
  builds of NumPy *and* torch. Scoped to `--no-binary bnn --no-deps`; verified
  locally that the sdist compiles the kernel and passes err=0 without torch.
- CodeQL was missing `actions: read` / `contents: read`.

### Fixed (build flags)

- **`/O2` was passed twice on MSVC.** setuptools already supplies
  `/O2 /W3 /GL /DNDEBUG /MD`; `setup.py` appended another `/O2`. Now it adds
  only `/openmp`, and the single-threaded fallback adds nothing on MSVC. The
  deliberate `-O3` on Unix stays (Python's own `CFLAGS` carry `-O2` and the last
  optimisation flag wins).

### Fixed (state leak)

- **`eval_report.render_summary()` rebound the module-level `RESULTS` global**
  and never restored it, so passing a custom directory silently redirected every
  later call in the process — a subsequent no-arg call read the caller's
  (often already-deleted) temp directory instead of `results/`. The directory is
  now threaded through as a parameter; regression test included.

### Fixed (privacy / portability)

- **Six committed files embedded the author's absolute home directory**
  (`C:\Users\<user>\...`) — three docs as copy-paste `cd` commands that could
  never work for a reader, and three result JSONs. Added `bnn.paths.repo_relative`
  and switched the generators (`train.py`, `tiny_transformer_wrap_demo.py`,
  `energy_bound_measured.py`) to emit repo-relative POSIX paths, then sanitised
  the committed files. No tracked file references a local home directory now.

### Docs

- **Ten documents were unreachable from the docs index**, including the SOTA
  survey, failure analysis and deep-research report — the substantive research
  backing the thesis. Added *Research background*, *Planning / design sketches*
  and *Kernel internals* sections to `docs/README.md`; every doc is now linked.
- Fixed two broken relative links: `MODEL_CARD.md` pointed at `../CITATION.cff`
  (outside the repo — it sits at the root), and the roadmap linked
  `docs/GUIDE_E2E.md` from inside `docs/`.

### Removed

- `results/wrap_demo_ternary.{json,md}` and `results/wrap_demo_dequant.{json,md}`
  — no generator exists for them anywhere in the tree, nothing references them,
  and their timings predate the kernel rewrite.

### Goldens regenerated

`results/benchmark.json`, `results/benchmark.md` and `results/profile.json` were
measured against the inflated FP32 baseline and have been regenerated. Floors in
`tests/golden_floors.json` are unchanged and still pass.

| | before | after |
|---|---|---|
| `speedup_compute_vs_numpy_fp32` @ 64×4096×4096 | 3.61× (inflated baseline) | **23.86×** |
| `profile.speedup_vs_fp32` @ 32×1024×1024 | 0.60× | **2.54×** |
| `profile.e2e_forward_ms` @ 32×1024×1024 | 1.396 | **0.236** |
| `ultra_wrap.binary_gemm_only_speedup_wide` | 2.12× | **9.39×** |

Also regenerated from the new kernel: `wrap_demo.json`, `ultra_wrap.json`,
`tiny_transformer_wrap.json`, `hybrid_ffn_wrap.json`, `energy_bound.json`
(derived from `wrap_demo` latency). Training-derived goldens
(`train_results`, `image_cifar`, `audio_synth`, `cifar10_proxy`) were **not**
regenerated: their accuracy figures are the goldens and are unaffected by kernel
changes, so retraining would burn hours to move only incidental timing fields.
- CI: `concurrency` cancels superseded runs, least-privilege `permissions`.
- **New `portability` CI matrix** — `ubuntu-24.04-arm`, `macos-latest` (Apple
  Silicon), `macos-13` (Intel) — each builds native, validates cross-ISA err=0,
  and re-runs the kernel suite with `BNN_KERNEL=scalar`.

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
