# Lane A — Optimiser / WC-O progress

Branch: `lane/a-wco`  
Base: `main` @ `5910978`  
Worktree: `C:\Users\mrkan\CRAZZY\bnn-lane-a-wco` (isolated from parallel lanes)

Owned paths only: `bnn/wrap/**`, `scripts/*wrap*`, `scripts/distill*`, `tests/test_wrap*`, `tests/test_qat*`, `docs/42*`, this note.

Thesis lock respected: no GPU 32× from `sign()`; dual-metric honesty; no invented goldens.

## ROADMAP checkbox proposals (for Wave 2 integrator)

| ID | Was | Propose | Evidence |
|----|-----|---------|----------|
| W3.T01 | `[~]` | `[x]` | `bnn.wrap.calibrate` unifies Tensor / Module; `calibrate_model` + `CalibReport` |
| W3.T02 | `[~]` | `[x]` | `wrap_model` always emits `effectiveness` (stub or measured) |
| W3.T03 | `[~]` | `[x]` | `policy_reason` always non-empty on `WrapReport` |
| W3.T04 | `[~]` | `[x]` | `tests/test_wrap_wco.py` drop-in / force / refuse cases |
| W3.T05 | `[x]` | `[x]` | unchanged |
| W3.T06 | `[x]` | `[x]` | unchanged (`search_layer_modes`) |
| W3.T07 | `[x]` | `[x]` | docs/42 recipe refreshed |
| W3.T08 | `[ ]` | `[x]` | `bnn/wrap/distill.py` + `scripts/distill_wrap_demo.py` + `tests/test_qat_distill.py` |
| W3.T09 | `[~]` | `[x]` | `bnn/wrap/fuse.py` + `wrap_model(..., fuse_bn=True)` |
| WC-O1–O4 | `[~]` | `[x]` / note residual | calib+auto+drop-in+QAT/distill demo; see residuals |

## Delivered

- Unified calibrate entry (`calibrate` / `calibrate_model`)
- Effectiveness always present; drop-in honesty helpers
- Real distill API with before/after cosine + demo script
- BN fuse (Linear+BN1d + BiReal) on wrap path
- Focused tests: `test_wrap_wco.py`, `test_wrap_fuse.py`, `test_qat_distill.py`
- Docs: `docs/42_QAT_AND_LAYER_SEARCH.md`

## Residuals (honest)

1. **`bnn/optimise.py` wiring** — Lane A does not own this file. Integrator should add `OptimiseConfig.fuse_bn` and optional `distill_steps` calling `fuse_bn_for_wrap_` / `distill_binary_student` before wrap.
2. **HW detect richness** — auto policy still uses native-kernel + CUDA heuristic.
3. **BitDistill-scale** — explicitly out of scope; demo is toy STE KD with measured cosine.
4. **QAT uplift is demo-scale** — tests assert machinery + finite metrics; production uplift needs real data.

## Focused test commands

```bash
pytest tests/test_wrap_wco.py tests/test_wrap_fuse.py tests/test_qat_distill.py tests/test_qat.py tests/test_wrapper.py -q
python scripts/distill_wrap_demo.py --steps 40
```
