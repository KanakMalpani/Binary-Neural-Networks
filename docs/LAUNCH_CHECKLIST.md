# Public launch checklist (W11.T07)

Optimiser **preview** launch = **v0.3.0** (not yet world-class v1.0).

## Done in-repo

- [x] MIT `LICENSE`
- [x] `SECURITY.md`, `CODE_OF_CONDUCT.md`, `CONTRIBUTING.md`, `CODEOWNERS`
- [x] Issue / PR templates with repro checklist
- [x] `MODEL_CARD.md` + limitations
- [x] `CITATION.cff`
- [x] `bnn optimise` + tutorials 07–08 + [`GUIDE_E2E.md`](GUIDE_E2E.md)
- [x] `bnn repro` culture + CI Win/Linux (+ Linux native hard gate)
- [x] Portability CI (linux-arm64 NEON, macos-arm64 NEON, macos-x86_64) + portable SIMD (`docs/41`)
- [x] SBOM script [`SBOM.md`](SBOM.md) / `scripts/generate_sbom.py`
- [x] PyPI prep doc [`PYPI_PUBLISH.md`](PYPI_PUBLISH.md)
- [x] CHANGELOG entry for 0.3.0
- [x] Annotated tag `v0.3.0` + GitHub Release

## Manual (human / GitHub settings)

- [x] Enable **Discussions** on the repo (Settings → Features) — W11.T06
- [x] Repo **About** description + topics (binary-neural-networks, xnor, quantization, …)
- [x] Branch protection on `main`: required checks `quality` / `windows` / `linux-native`, no force-push/deletes, resolve conversations
- [x] Label good first issues (`good first issue`) — W11.T08 (#1, #2)
- [ ] Configure PyPI Trusted Publishing when ready — W8.T08
- [ ] Optional: README shields for CI / release once Actions green on tag

## Not required for v0.3 preview

- Full WC-§1 green (that is **v1.0**)
- ImageNet SOTA, RAPL Joules, bitnet.cpp submodule, ONNX full — moonshots
- WASM SIMD (optional moonshot) — see [`MOONSHOT_DEFERRALS.md`](MOONSHOT_DEFERRALS.md)
- ~~AVX-512 / ARM NEON native~~ — **delivered** (`docs/41`; AVX-512 used when present, never required)

## Post-tag verify

```bat
git checkout v0.3.0
pip install -e ".[dev]" -c constraints.txt
bnn repro
```

Expect `REPRO: PASS`.
