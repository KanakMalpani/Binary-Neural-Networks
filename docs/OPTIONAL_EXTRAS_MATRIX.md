# Optional extras version matrix (W14.T06)

Core lab (`bnn-lab`) does **not** require `transformers` or `torchao`. Those
extras are bridges / pedagogy only — never part of the 32× pack thesis gate.

| Extra | Pin (pyproject) | Probe | CI default |
|-------|-----------------|-------|------------|
| `[hf]` → `transformers` | `>=4.40,<5` | `python scripts/smoke_optional_extras.py` | skipped unless `[hf]` installed |
| `torchao` | **unpinned** (GPU host recipe) | same script `--probe torchao` | never required on CPU CI |
| `datasets` / `huggingface_hub` | with `[hf]` | import smoke | optional |

## How to run

```bash
pip install -e ".[hf]" -c constraints.txt   # optional
python scripts/smoke_optional_extras.py
python scripts/smoke_optional_extras.py --require hf   # fail if missing
```

Pytest marker:

```bash
pytest -q -m optional_extras tests/test_optional_extras_matrix.py
```

## Honesty

- **torchao INT4/FP8** is the commodity **GPU** path — not classic BNN XNOR.
  See [`24_GPU_INT4_FP8_LANE.md`](24_GPU_INT4_FP8_LANE.md) and
  `scripts/bridges/torchao_int4_recipe.py`.
- A green optional probe does **not** refresh `golden_floors.json`.
- Version skew between local HF and CI is expected; pins keep the band honest.

## Status

Smoke script + skip-if-missing pytest shipped. Full multi-version matrix in CI
remains optional (`workflow_dispatch`) — CPU runners should not pull CUDA
torchao wheels by default.
