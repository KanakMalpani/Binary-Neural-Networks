# Public launch checklist (W11.T07)

GitHub **v1.0.0** is tagged (2026-08-04). **`bnn-lab` 1.0.0 is on PyPI**
(OIDC Trusted Publisher — [`PYPI_PUBLISH.md`](PYPI_PUBLISH.md)). Import/CLI
remain `bnn`. A pip wheel is library-only; `bnn repro` still needs a clone +
`[dev]`.

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
- [x] PyPI Trusted Publishing — W8.T08
  - [x] Distribution name **`bnn-lab`** (`bnn` taken on PyPI); GitHub env **`pypi`**; `wheels.yml` publish job (OIDC only)
  - [x] Trusted Publisher on pypi.org (`bnn-lab` / `wheels.yml` / env `pypi`) — pending converted to **active** after first upload
  - [x] Actions → wheels → **publish=true** on **`main`** (run [31825286443](https://github.com/KanakMalpani/Binary-Neural-Networks/actions/runs/31825286443); not frozen tag `v1.0.0`)
- [x] Repository is **public** — free Code Scanning + Scorecard public badge API work
- [x] README shields for CI / CodeQL / Scorecard / wheels / Pages; live [`pypi/v/bnn-lab`](https://pypi.org/project/bnn-lab/) badge

## Not required to claim the v1.0.0 GitHub tag

- Venue paper submit
- ImageNet SOTA, privileged RAPL Joules — see [`MOONSHOT_DEFERRALS.md`](MOONSHOT_DEFERRALS.md)
- ~~Live `pip install bnn-lab` from PyPI~~ — **shipped** 2026-08-14 (`bnn-lab` 1.0.0)
- ~~WASM SIMD / Bi-Real CIFAR ref / bitnet recipe~~ — **delivered** (moonshot residuals documented)
- ~~AVX-512 / ARM NEON native~~ — **delivered** (`docs/41`; AVX-512 used when present, never required)

## Post-tag verify

From Git (CLI / `bnn repro`):

```bat
git checkout v1.0.0
pip install -e ".[dev]" -c constraints.txt
bnn repro
```

From PyPI (library import only — no `bnn repro` on the next line):

```bat
pip install bnn-lab==1.0.0
python -c "import bnn; print(bnn.__version__)"
```
