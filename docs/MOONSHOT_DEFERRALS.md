# Moonshot / Phase F deferrals (explicit blockers)

Honest leftovers toward **v1.0 world-class**. None of these are pretend-done.

| ID | Item | Blocker | Acceptance leftover |
|----|------|---------|---------------------|
| W2.T04 | ARM NEON | No aarch64 CI agent | See [`spikes/ARM_NEON_SPIKE.md`](spikes/ARM_NEON_SPIKE.md) |
| W2.T05 | AVX-512 VPOPCNTDQ | Hardware + dispatch | [`spikes/AVX512_MOONSHOT.md`](spikes/AVX512_MOONSHOT.md) |
| W2.T06 | WASM SIMD | Browser pedagogy only | Optional demo after C/D |
| W5.T05 | `.bnnpack` v2 | Design after v0.3 usage | Schema ADR + hashes + ternary meta |
| W5.T06 | safetensors export | Depends W5.T05 | Packed tensor export path |
| W5.T07 | ONNX full | Heavy ORT custom op | **Defer:** document bridge-only; spike note OK |
| M3 | ONNX Runtime custom op | Same | Keep as bridge recommendation |
| M5 | RAPL / board Joules | OS privileges / HW | Energy-proxy remains default |
| M6 | Full ImageNet protocol runner | Dataset + time | Folder protocol stub only (`W6.T07`) |
| — | bitnet.cpp submodule | Vendor pin / size | Keep `scripts/bridges/llamacpp_bitnet_recipe.py` |
| W8.T07 | Artifact attestations | Org policy | Optional on later tags |
| W8.T08 | PyPI Trusted Publishing | Manual PyPI↔GH link | [`PYPI_PUBLISH.md`](PYPI_PUBLISH.md) |
| W9.T06 | Autodoc API site | MkDocs deploy | Stub exists; expand later |
| W11.T06 | GitHub Discussions | Manual settings toggle | Listed on launch checklist |
| W12.T02 | Publication plan | Venue + claims whitelist | After WC gates closer |
| W3.T06 | Search binary/ternary/skip | Builds on W3.T05 | Sensitivity scores exist; full search later |
| W3.T08 | Distill integration | Recipe time | `distill_sketch.py` remains sketch |
| W2.T07 | Memory arena | Perf eng | OpenMP thread API enough for 0.3 |
| W4.T05 | ResNet-BiReal CIFAR ref | Optional zoo polish | Bi-Real CNN exists |

## ONNX decision (W5.T07 executed as defer)

**Decision:** do **not** ship a full ONNX custom-op runtime in-tree for v0.3/v0.4.
Prefer:

1. `.bnnpack` encode/decode for packed CPU path
2. Documented bridges to torchao / GGUF / bitnet.cpp for production serving
3. Revisit ONNX only if a consumer demands ORT custom ops with measured dual metrics

## ImageNet / RAPL / bitnet submodule

Remain **non-gates** for optimiser preview. Thesis lock unchanged.
