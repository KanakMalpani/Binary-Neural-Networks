# Fair evaluation protocol (W7.T05)

**Rule:** same shapes as [`BENCH_SHAPES.md`](BENCH_SHAPES.md); dual-metric reporting;
no invented goldens.

## Machine disclosure (required in Pareto / latency submissions)

| Field | Example |
|-------|---------|
| CPU model | `platform.processor()` / `lscpu` |
| OS | Windows 11 / Ubuntu 22.04 / … |
| Python / torch | from `constraints.txt` pin |
| Native kernel | Win/Linux/macOS/ARM native (runtime ISA) / NumPy fallback |
| Threads | OpenMP / `BNN_NUM_THREADS` / `torch.set_num_threads` |
| Warmup | ≥3 iters discarded before timed reps |
| Power | RAPL / board Joules optional (moonshot); else energy-proxy |

## Procedure

1. `pip install -e ".[dev]" -c constraints.txt`
2. Compile native when available: `python -m bnn.kernels.compile_native`
3. Confirm `bnn validate-native` (exit 0) **or** document NumPy-only path
4. Warmup ≥3; time ≥8 reps (kernel microbench) or documented CLI defaults
5. Record **both**:
   - Theoretical compression (pack ratio)
   - Wall-clock latency / samples_per_s
6. Emit Pareto JSON: `python scripts/pareto_report.py …` (`bnn_pareto_report_v1`)

## Honesty

- Never label theoretical 32× as e2e latency.
- Never claim GPU 32× from `sign()` / STE.
- Floats need not be bit-identical across machines; **conclusions** vs
  `tests/golden_floors.json` must agree.

## Leaderboard (manual, W7.T07)

Submit a PR adding `results/community/<handle>_<date>.json` that validates with
`bnn.eval.pareto.validate_pareto_report` and links this protocol + CPU model.
