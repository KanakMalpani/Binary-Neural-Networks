# Lane D — Docs / eval / safety / demos

**Branch:** `lane/d-docs`  
**Base:** `main` @ `5910978`  
**Owns:** `docs/**` (except `41`/`42` + other lane notes), `mkdocs*`,
`tests/test_docs*`, `scripts/run_eval_suite.py`, `bnn/profile` budgets,
path-safety helpers/tests. Progress lives here — integrator flips ROADMAP twin.

## Task status (for integrator)

| ID | Status | Evidence |
|----|--------|----------|
| W9.T06 | `[x]` expand | `docs/api/paths.md`, profile budget symbols on reporting page, mkdocs nav |
| W7.T08 | `[x]` | `scripts/run_eval_suite.py` always runs codec + seq pytest smokes |
| W13.T03 | `[x]` | `bnn.profile.SOFT_BUDGETS_MS` + `check_soft_budgets`; eval-suite warn / `--strict-budgets` |
| W13.T04 | `[x]` | `docs/34` synced to committed `results/benchmark.json`; test asserts thread curves |
| W13.T06 | `[x]` | `profile_packed_linear(..., compare_baselines=True)` → FP32 + INT8-WO timings |
| W10.T03 | `[x]` | `tests/test_paths_security.py` checkpoint + refuse-pickle tests |
| W10.T06 | `[x]` | `warn_untrusted_pack` + `load_bnnpack` soft-warn outside lab roots |
| W9.T10 | `[x]` | `docs/demos/optimise_quickstart.cast` + README |
| W14.T03 | `[x]` | `docs/TORCH_PIN_POLICY.md` linked from compat matrix |
| W14.T06 | `[x]` | `scripts/smoke_optional_extras.py` + `tests/test_optional_extras_matrix.py` |

## Residuals

- Full multi-version HF/torchao CI matrix remains `workflow_dispatch` / optional
  (smoke is enough for v1 gate).
- Real asciinema re-record on a clean machine before marketing screenshots
  (committed cast is pedagogy-fixed timings).
- Soft budgets are **warn-only** in eval-suite unless `--strict-budgets`.

## Focused tests

```bash
pytest -q tests/test_paths_security.py tests/test_profile.py \
  tests/test_optional_extras_matrix.py tests/test_docs_links.py \
  tests/test_codec.py::test_roundtrip_gemm_err_zero \
  tests/test_seq_encoder_decoder.py::test_encoder_forward_shape
```
