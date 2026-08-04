# Lane I — Moonshot bitnet.cpp bridge

| Field | Value |
|-------|-------|
| Branch | `lane/i-bitnet` |
| Base | `main` @ `5910978` |
| Status | **Delivered** (recipe + pin; no giant submodule) |
| Date | 2026-08-04 |

## Owned paths touched

- `scripts/bridges/llamacpp_bitnet_recipe.py` — hardened: pins, `--check`, `--probe`
- `scripts/bridges/llamacpp_bitnet_pins.json` — thin SHA/tag pin (preferred over submodule)
- `docs/23_BITNET_CPP_BRIDGE.md` — full pinned recipe + policy
- `tests/test_llamacpp_bitnet_bridge.py` — schema / recipe / probe smoke
- `results/bridge_cpu_llamacpp_bitnet.json` — regenerated from recipe
- `third_party/BITNET_PIN.md` — explicit non-vendor note

## ROADMAP checkbox proposals (integrator only)

Do **not** edit twin ROADMAP from this lane. When merging, apply:

| ID | Current | Propose | Evidence |
|----|---------|---------|----------|
| W4.T06 | `[~]` docs + bridges | `[x]` or keep `[~]` with “pinned bridge shipped” note | docs/23 + pins + tests |
| WC-P2 | bridges first-class | still `[~]` until Lane E `bnn bridge` CLI | recipe is first-class; CLI is E |
| Scorecard “Bridges GPU/BitNet” | `[~]` | keep `[~]` pending CLI; handoff clarity improved | docs/23 |
| Moonshot “bitnet.cpp submodule” | deferred | **CLOSED-BY-POLICY** — recipe+pin replaces submodule | pins `vendor_submodule: false` |

## Acceptance

- [x] Recipe prints pinned clone/build/quantize steps
- [x] `--check` validates pins without network
- [x] No microsoft/BitNet git submodule added
- [x] Docs/23 documents pins, non-goals, dual metrics
- [x] Pytest covers pins + recipe emission

## Residuals

1. **Lane E:** first-class `bnn bridge …` CLI over this script (ownership: E).
2. **Full upstream build** not run in this lane (toolchain + multi-GB models); operators clone at pin locally.
3. **Re-pin cadence:** when microsoft/BitNet moves, bump `llamacpp_bitnet_pins.json` + docs/23 snapshot table in one PR.
4. Integrator updates `docs/MOONSHOT_DEFERRALS.md` row for bitnet submodule → CLOSED-BY-POLICY / recipe+pin.
