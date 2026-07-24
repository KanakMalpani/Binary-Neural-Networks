# Leaderboard template (W7.T07)

Manual community submissions only — fair protocol required.

## Submission

1. Run on your machine per [`FAIR_EVAL_PROTOCOL.md`](FAIR_EVAL_PROTOCOL.md).
2. Produce JSON via `python scripts/pareto_report.py …` (`bnn_pareto_report_v1`).
3. Open a PR adding:

```
results/community/<github_handle>_<YYYYMMDD>.json
```

4. PR description must include: CPU model, OS, native yes/no, thread count, warmup.

## Review rules

- Validates with `bnn.eval.pareto.validate_pareto_report`
- Uses only shapes from [`BENCH_SHAPES.md`](BENCH_SHAPES.md) (or clearly marked non-golden probes)
- Dual-metric language; no GPU-32×-from-sign claims
- Maintainers may request re-run; floats need not match

## Non-goals

- Automated ranking that invents shapes
- Cross-machine bit-identical floats as pass/fail
