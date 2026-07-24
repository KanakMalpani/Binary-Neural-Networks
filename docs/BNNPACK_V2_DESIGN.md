# `.bnnpack` v2 design sketch (W5.T05) — not implemented

| Field | Value |
|-------|-------|
| **Status** | Design only (v1 remains shipping format) |
| **Date** | 2026-07-25 |

## Goals

- Ternary bitplanes + binary in one container
- Richer meta (policy, sensitivity summary, HW probe)
- Content hashes per tensor for untrusted-file warnings
- Forward-compatible schema version field (already in v1 header)

## Non-goals for v2.0 draft

- Full safetensors interop (W5.T06 follows)
- ONNX (deferred — `docs/MOONSHOT_DEFERRALS.md`)

## Migration

v1 readers must reject unknown major versions loudly. v2 writers may emit
v1-compatible subset when ternary unused.

## Next step

ADR under `docs/adr/` when implementing; keep round-trip tests green.
