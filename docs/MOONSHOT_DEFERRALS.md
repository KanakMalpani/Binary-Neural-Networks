# Moonshot / Phase F deferrals (post v1.0.0)

Honest leftovers after Wave 2 integration. Delivered moonshots are listed as
**closed**; remaining rows are non-gates unless marked human-blocking.

> **Closed earlier:** W2.T04 ARM NEON and W2.T05 AVX-512/AVX2 runtime dispatch —
> see [`41_PORTABLE_SIMD_KERNEL.md`](41_PORTABLE_SIMD_KERNEL.md).

## Closed in Wave 1–2 (not deferrals anymore)

| ID | Item | Evidence |
|----|------|----------|
| W2.T06 / M1 | WASM SIMD pedagogy | `wasm/`, `docs/spikes/WASM_SIMD.md`, `tests/test_wasm_parity.py` |
| W5.T05 | `.bnnpack` v2 | ADR 0003 + hash verify |
| W5.T06 | safetensors export | `bnn/codec/safetensors_export.py` |
| W5.T07 / M3 | ONNX full custom op | **CLOSED-BY-POLICY** bridge-only — `docs/spikes/ONNX_BRIDGE_ONLY.md` |
| W3.T06 | Layer mode search | `search_layer_modes` + docs/42 |
| W3.T08 | Distill integration | `bnn/wrap/distill.py` + `OptimiseConfig.distill_steps` |
| W4.T05 | ResNet-BiReal CIFAR ref | `ResNetBiRealCIFAR` |
| M5 | RAPL / energy path | `bnn/energy/**`, Windows CLOSED-BY-PROXY |
| M6 / W6.T07 | ImageNet protocol runner | `scripts/imagenet_protocol.py` (smoke/proxy; no SOTA) |
| — | bitnet.cpp submodule | **CLOSED-BY-POLICY** — recipe + SHA pin (`third_party/BITNET_PIN.md`) |
| W9.T06 | Autodoc API site | MkDocs mkdocstrings (CI `--strict`) |
| W12.T02–T05 | Publication / figures / triage | docs/32 + `bnn bridge figures` |
| W8.T08 | PyPI Trusted Publishing upload | [`bnn-lab` 1.0.0](https://pypi.org/project/bnn-lab/1.0.0/) OIDC (run 31825286443); no API-token path |

## Still open / residual

| ID | Item | Blocker | Acceptance leftover |
|----|------|---------|---------------------|
| M5+ | Privileged wrap-workload RAPL | OS / powercap permissions | Spike loop pedagogy only; energy-proxy remains default |
| M3+ | ORT custom op revisit | Consumer demand + dual metrics | Keep bridge recommendation |
| — | BitDistill-scale KD | Recipe time / data | Toy STE KD demo is enough for WC-O4 |
| — | Venue paper submit | Author time | Plan exists; not a repro gate |
| M4 | Community leaderboard submissions | External contributors | Template + fair protocol shipped |

Post-v1 product leftovers (not moonshots): live HF Space and B1 submit — Hub `.bnnpack` canaries **shipped** (PR #42; not SOTA; wrap pack is PTQ bytes, not the QAT checkpoint). See [`TRANSFORMATION_PLAN.md`](TRANSFORMATION_PLAN.md). In-repo 2026-08-15: `wrap_demo` AND-gate, NumPy BLAS fallback, `demo/space/` (Space not live).

## PyPI (shipped)

Pending Trusted Publisher for `bnn-lab` / `wheels.yml` / env `pypi` was used for
the first OIDC upload on **`main`**. After that upload the publisher should show
as **active** on pypi.org Publishing. Recurring releases stay OIDC-only — do
**not** invent long-lived API tokens. See [`PYPI_PUBLISH.md`](PYPI_PUBLISH.md).

## Thesis lock (unchanged)

Packed CPU/edge XNOR–popcount; never GPU 32× from `sign()`; no invented golden
shapes; dual-metric honesty.
