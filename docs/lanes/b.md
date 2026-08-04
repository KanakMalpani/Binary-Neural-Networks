# Lane B — Codec v2 + safetensors + Conv2d

| Field | Value |
|-------|-------|
| **Branch** | `lane/b-codec` |
| **Base** | `main` @ `5910978` |
| **Date** | 2026-08-04 |
| **Status** | Ready for integrator review |

## Owned work completed

| Task | Status | Notes |
|------|--------|-------|
| W5.T05 `.bnnpack` v2 | `[x]` | Ternary + Conv2d kinds, `content_sha256`, container `hashes`; default write v2; v1 still loads |
| W5.T06 safetensors | `[x]` | `bnn.codec.export_bnnpack_safetensors` + JSON meta sidecar (soft-dep `safetensors`) |
| W5.T09 Conv2d polish | `[x]` | `weight_packed_i64` buffer, state_dict sync, non-mutating forward, groups/dilation=1 guards |
| W5.T07 / M3 ONNX | `[x]` | Bridge-only confirmed; spike refreshed at `docs/spikes/ONNX_BRIDGE_ONLY.md` |

## Files touched (ownership)

- `bnn/codec/packfile.py`, `bnn/codec/__init__.py`, `bnn/codec/safetensors_export.py`
- `bnn/wrap/packed_linear.py` — **only** `PackedBinaryConv2d` polish
- `tests/test_codec.py`
- `docs/adr/0003_bnnpack_v2.md`, `docs/adr/README.md` (index row)
- `docs/spikes/ONNX_BRIDGE_ONLY.md`
- `docs/lanes/b.md` (this file)

## ROADMAP checkbox flips for integrator

Apply to twin when merging (do **not** edit ROADMAP from this lane):

- [x] W5.T05 `.bnnpack` v2 (was design sketch)
- [x] W5.T06 safetensors
- [x] W5.T09 Conv2d packed polish
- W5.T07 remains executed defer; spike path now `docs/spikes/ONNX_BRIDGE_ONLY.md`

## Residuals

- `pyproject.toml` does not list `safetensors` (Lane C packaging). Soft-import + install hint; often pulled transitively by HF. Integrator/Lane C may add to `hf` extra.
- No full ORT custom op (intentional).
- Did not edit `docs/MOONSHOT_DEFERRALS.md` / ROADMAP twin (integrator).

## Acceptance

Focused: `pytest tests/test_codec.py -q` — expect PASS (compression 32×, GEMM err=0, v2 hashes, ternary/conv, safetensors).
