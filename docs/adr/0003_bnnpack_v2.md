# ADR 0003 — `.bnnpack` v2 schema

| Field | Value |
|-------|-------|
| **Status** | Accepted |
| **Date** | 2026-08-04 |
| **Task** | W5.T05 (+ W5.T06 safetensors companion) |
| **Deciders** | Lab maintainers (Lane B) |

## Context

v1 (`BNNPACK1` / `version: 1`) ships binary XNOR Linear blobs via
`torch.save` + `weights_only` load. Product needs:

1. Ternary weight-only layers in the same container
2. Packed Conv2d (size path) blobs
3. Per-layer content hashes for untrusted-file warnings
4. A side-car safetensors export of packed tensors (W5.T06)

ONNX Runtime custom ops remain **out of scope** (W5.T07 / M3 bridge-only —
see `docs/spikes/ONNX_BRIDGE_ONLY.md`).

## Decision

1. **Magic stays** `BNNPACK1` (string gate). **Version field** is the schema
   discriminator: writers default to `version: 2`; readers accept `1` and `2`.
2. **Layer `kind` values:**
   - `binary_xnor` — existing Linear packed uint64 words (v1 compatible)
   - `ternary_weight_only` — 2-bit packed `{-1,0,+1}` + scale (+ optional bias)
   - `binary_conv_packed` — Conv2d flattened row-pack + per-out-channel alpha
3. **Hashes:** each v2 layer blob includes `content_sha256` (hex SHA-256 of the
   canonical packed weight bytes). Container may also expose `hashes` map
   `layer_name → sha256`.
4. **Meta:** free-form dict; recommended keys: `policy`, `created_by`,
   `schema_notes`. No required meta for round-trip.
5. **Safetensors:** optional export maps packed tensors to a `.safetensors`
   file with JSON sidecar meta (kind / shapes / hashes). Soft-depends on the
   `safetensors` package (often present via HF extras).
6. **Thesis lock:** compression claims remain exact 32× for aligned binary packs;
   Conv2d forward stays **dequant + `F.conv2d`** (size win, not a fake XNOR
   GPU 32× claim). Ternary reports theoretical 2-bit size honestly.

## Consequences

- `load_bnnpack` rejects unknown versions loudly (unchanged policy).
- v1 files keep loading; encode defaults write v2.
- `decode_file` dispatches by `kind`; GEMM err=0 checks apply to
  `binary_xnor` only.
- Integrator may flip ROADMAP W5.T05 / W5.T06 from lane note `docs/lanes/b.md`.

## Alternatives considered

| Option | Why not |
|--------|---------|
| New magic `BNNPACK2` | Breaks simple gate; version field already exists |
| Force safetensors as only container | Breaks v1 CLI / torch.save story; dual-format is fine |
| Full ORT custom op in-tree | XL moonshot; bridge-only is the executed decision |

## Links

- `bnn/codec/packfile.py`, `bnn/codec/safetensors_export.py`
- `docs/BNNPACK_V2_DESIGN.md` (pre-implementation sketch)
- `docs/spikes/ONNX_BRIDGE_ONLY.md`
- `tests/test_codec.py`
