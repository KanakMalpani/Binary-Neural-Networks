# Lane F — Moonshot WASM (W2.T06)

| Field | Value |
|-------|-------|
| **Branch** | `lane/f-wasm` |
| **Base** | `main` @ `5910978` |
| **Worktree** | `C:\Users\mrkan\CRAZZY\Binary-Neural-Network-lane-f` |
| **Owns** | `wasm/**`, `bnn/kernels/wasm/**`, WASM docs/spike, `docs/lanes/f.md`, `tests/test_wasm_parity.py` |
| **May read** | `bnn/kernels/binary_gemm.c` (read-only) |
| **Status** | **Delivered (pedagogy)** — 2026-08-04 |

## Tasks

| ID | Item | Lane status |
|----|------|-------------|
| W2.T06 | WASM SIMD prototype (optional) | `[x]` pedagogy delivered |
| M1 | WASM SIMD popcount demo | `[x]` Node + HTML demos |

## Acceptance (this lane)

- [x] Pedagogy C source with scalar + WASM SIMD128 popcount
- [x] Optional Rust `wasm32-unknown-unknown` build (`wasm/build.py --rust`)
- [x] Browser / Node demo (`wasm/js/demo.html`, `demo_node.mjs`)
- [x] NumPy / Python parity tests (`tests/test_wasm_parity.py`, err=0)
- [x] Honest non-goals vs native CPU kernels documented (`docs/spikes/WASM_SIMD.md`)
- [x] Spike marked **DELIVERED** (pedagogy)

## Integrator checklist (do not edit ROADMAP in this lane)

When merging, flip in twin ROADMAP / `docs/37_…`:

- Capability row **WASM** → delivered (pedagogy)
- Moonshot **M1** → `[x]`
- Workstream **W2.T06** → `[x]`
- Phase F exit bullet / launch checklist WASM line → point at `docs/spikes/WASM_SIMD.md`
- Optionally refresh `docs/MOONSHOT_DEFERRALS.md` to remove W2.T06 or mark closed

## Residuals

- Compiled `wasm/dist/binary_gemm_wasm.wasm` is **optional**; CI green without emcc/clang.
- Python `simd128` kernel name is a **label only** (math remains scalar popcount).
- No wall-clock benches vs native — by design.
- Full Emscripten browser glue (glue JS, memory grow helpers beyond demo) left as follow-up if a product surface appears.

## Thesis lock

Packed CPU/edge XNOR–popcount remains the product claim. WASM does not invent
GPU 32× from `sign()`, does not change golden floors, and does not replace
`docs/41` native kernels.
