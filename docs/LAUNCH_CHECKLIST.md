# Public launch checklist (W11.T07)

GitHub **v1.0.0** is tagged (2026-08-04). This is **not** a v0.3 preview
checklist. **`bnn-lab` is still not on PyPI** — Trusted Publisher remains a
human gate (`docs/PYPI_PUBLISH.md`). Do not treat the tag as `pip install
bnn-lab`.

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
- [x] CHANGELOG entries for 0.3.0 and 1.0.0
- [x] Annotated tags `v0.3.0` and `v1.0.0` + GitHub Releases

## Manual (human / GitHub settings)

- [x] Enable **Discussions** on the repo (Settings → Features) — W11.T06
- [x] Repo **About** description + topics (binary-neural-networks, xnor, quantization, …)
- [x] Branch protection on `main`: required checks `quality` / `windows` / `linux-native`, no force-push/deletes, resolve conversations
- [x] Label good first issues (`good first issue`) — W11.T08 (#1, #2)
- [~] PyPI Trusted Publishing — W8.T08
  - [x] Distribution name **`bnn-lab`** (`bnn` taken on PyPI); GitHub env **`pypi`**; `wheels.yml` publish job (OIDC only)
  - [ ] Register pending Trusted Publisher on pypi.org (`bnn-lab` / `wheels.yml` / env `pypi`)
  - [ ] Then Actions → wheels → **publish=true** on **`main`** (not frozen tag `v1.0.0`; see `PYPI_PUBLISH.md`)
- [x] Repository is **public** — free Code Scanning + Scorecard public badge API work
- [x] README shields for CI / CodeQL / Scorecard / wheels (PyPI `bnn-lab` badge stays 404 until first upload)

## Not required to claim the v1.0.0 GitHub tag

- Live `pip install bnn-lab` from PyPI (human Trusted Publisher)
- Venue paper submit
- ImageNet SOTA, privileged RAPL Joules — see [`MOONSHOT_DEFERRALS.md`](MOONSHOT_DEFERRALS.md)
- ~~WASM SIMD / Bi-Real CIFAR ref / bitnet recipe~~ — **delivered** (moonshot residuals documented)
- ~~AVX-512 / ARM NEON native~~ — **delivered** (`docs/41`; AVX-512 used when present, never required)

## Post-tag verify (from Git, not PyPI)

```bat
git checkout v1.0.0
pip install -e ".[dev]" -c constraints.txt
bnn repro
```

Expect `REPRO: PASS`. After Trusted Publisher upload, the clean-venv check is
`pip install bnn-lab==1.0.0` + `bnn repro`.
