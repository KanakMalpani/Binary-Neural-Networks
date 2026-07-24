# 39 — GUIDE_E2E completion (smoke confirmation)

| Field | Value |
|-------|-------|
| **Date** | 2026-07-25 |
| **Artifact** | [`GUIDE_E2E.md`](GUIDE_E2E.md) |
| **Thesis** | Packed CPU/edge; no fake GPU 32× from `sign()` |

## What shipped

- Master narrative guide: `docs/GUIDE_E2E.md` (install → repro → `bnn optimise` → codec → modalities → threads → bridges → HF → metrics → troubleshooting → ROADMAP).
- Tutorials 01–08 cross-linked; prefer `bnn optimise` over `bnn wrap --ultra`.
- README + `docs/README.md` + `AGENTS.md` point at the User Guide.
- ROADMAP / `docs/37`: W9.T08, W9.T09, WC-D1 tutorial path marked done.
- Code fix smoked by the guide: NumPy 1.26 lacks `np.bitwise_count` → `bnn.kernels.popcount` LUT fallback (constraints allow NumPy 1.24+).

## Smoke results (this machine)

| Step | Result |
|------|--------|
| `python -m bnn.kernels.compile_native` | PASS (DLL present) |
| `bnn encode` / `bnn decode` | `compression=32.00x`, `DECODE: PASS`, `fp_err=0.0` |
| `bnn optimise --policy auto --qat-steps 20 --force --pack …` | Exit 0; schema `bnn_optimise_report_v1` / suite `ultra_wrap_suite_v1` |
| `bnn profile` (small shape) | Exit 0 |
| `bnn repro` | **`REPRO: PASS`** (pytest, export-check, validate-native err=0, golden-compare) |

Python: 3.12.10 · NumPy: 1.26.4 · native kernel: available.

## Follow-up (not blocking)

- W9.T06 autodoc API reference
- W9.T10 optional GIF/asciinema demos
