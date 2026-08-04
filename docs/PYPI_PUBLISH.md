# PyPI publish (W8.T08)

**Status (2026-08-04, Lane C):** packaging + `wheels.yml` (build / attest /
dry-run summary / OIDC publish) are ready. **`bnn-lab` is not on PyPI yet**
(`https://pypi.org/pypi/bnn-lab/json` → **404**). Live upload is blocked on a
**human Trusted Publisher** registration — see [Human blocker](#human-blocker)
below. This repo does **not** use long-lived PyPI API tokens.

## Why the distribution name is `bnn-lab`

The short name [`bnn`](https://pypi.org/project/bnn/) is **already taken** on
PyPI (Adrian Bulat’s unrelated binary-networks package, 0.1.2). Publishing
under that name would fail or collide.

| Surface | Name |
|---------|------|
| **PyPI / pip** | `bnn-lab` |
| **Wheel / sdist filename** | `bnn_lab-…` (PEP 503 / PEP 440 normalisation) |
| **Import** | `import bnn` (unchanged) |
| **CLI** | `bnn` console script (unchanged) |

```bat
pip install bnn-lab
bnn repro
```

## Checklist

- [x] `pyproject.toml` distribution name `bnn-lab`, version synced with `bnn/_version.py`
- [x] `readme`, `license`, classifiers (3.11–3.13), `project.urls` (incl. PyPI + publish runbook)
- [x] Console script `bnn` (import package remains `bnn`)
- [x] `constraints.txt` for reproducible installs
- [x] `wheels.yml` build matrix + W8.T07 attestations on wheels and sdist
- [x] `wheels.yml` `package-check` dry-run job (artifacts present; no upload)
- [x] `wheels.yml` publish job gated on `workflow_dispatch` + `publish=true` + `environment: pypi`
- [x] GitHub Environment `pypi` created on the repo (Settings → Environments)
- [ ] **PyPI.org pending Trusted Publisher** for `bnn-lab` / `wheels.yml` / env `pypi`
- [ ] First production publish via Actions (`publish=true`) after publisher is linked
- [ ] Post-upload: `pip install bnn-lab` + `bnn repro` on a clean venv

## Human blocker

Trusted Publishing is **not** configured for this project yet:

1. GitHub env **`pypi`** exists on `KanakMalpani/Binary-Neural-Networks`.
2. PyPI project **`bnn-lab` does not exist** (JSON API 404 as of Lane C).
3. Therefore OIDC publish would fail if `publish=true` were dispatched now.
4. **Do not** add a PyPI API token secret or invent a token-based publish path.
5. Lane C stops before any failing upload: only `publish=false` dry-runs are
   dispatched until a maintainer completes the steps below.

### Trusted Publishing (maintainer, once)

1. Create a free account on https://pypi.org (or log in).
2. Confirm GitHub Environment named **`pypi`** on
   `KanakMalpani/Binary-Neural-Networks` (already created). Optional:
   add required reviewers on that environment for public-repo safety.
3. On PyPI: **Publishing → Add a new pending publisher**:
   - PyPI project name: `bnn-lab`
   - Owner: `KanakMalpani`
   - Repository: `Binary-Neural-Networks`
   - Workflow: `wheels.yml`
   - Environment: `pypi`
4. In GitHub Actions → **wheels** → *Run workflow* → set **publish = true**
   (only after the pending publisher is saved).
5. First successful OIDC upload creates the PyPI project and unblocks the
   README PyPI badge.

The publish job uses OIDC (`id-token: write`); no long-lived API token in the repo.

## Wheel matrix notes (Lane C dry-run)

cibuildwheel on **macOS/Windows** sets `BNN_NO_OPENMP=1` so wheels ship
portable SIMD kernels without Homebrew `libomp` (delocate/macOS 26 bottle
mismatch) or MSVC `vcomp*.dll` ctypes load failures. Linux still links
`libgomp` when available. Thesis win remains packed XNOR–popcount, not thread
scaling. Local OpenMP: omit the env var / use `BNN_FORCE_OPENMP` on macOS via
`compile_native`.

**Skipped target:** `cp313-macosx_x86_64` — PyTorch has no wheel there, so a
`bnn-lab` install (hard-deps `torch`) cannot succeed; we do not publish that
tag. Use cp311/cp312 on Intel Mac, or cp313 on Apple Silicon / Win / Linux.

`hf` optional extra includes `safetensors` for Lane B packed export.

## Dry-run (Actions, preferred)

```bat
gh workflow run wheels.yml --ref <branch-or-main> -f publish=false
gh run watch
```

Expect: matrix wheels + sdist + attestations + `package-check` **success**;
`publish` job **skipped**.

## Local dry-run (no upload)

```bat
python -m pip install -U build twine
python -m build
twine check dist/*
```

Expect: `bnn_lab-0.3.0` sdist + wheel, **PASSED**.

### Historical note

`twine check` for the old distribution name `bnn-0.3.0` passed on 2026-07-25;
the rename is required solely because of the PyPI name collision.

## Interim install (Git / Release)

```bat
pip install "bnn-lab @ git+https://github.com/KanakMalpani/Binary-Neural-Networks.git@v0.3.0"
```

Or editable:

```bat
pip install -e ".[dev]" -c constraints.txt
```

## Attestations (W8.T07)

After a wheels run on this repo:

```bat
gh attestation verify path\to\bnn_lab-*.whl --repo KanakMalpani/Binary-Neural-Networks
gh attestation verify path\to\bnn_lab-*.tar.gz --repo KanakMalpani/Binary-Neural-Networks
```

See also [`SBOM.md`](SBOM.md).

## Public-repo note

This repository is **public**. OpenSSF Scorecard’s public badge API and free
Code Scanning apply. Trusted Publishing for **`bnn-lab`** remains the only
maintainer gate for a live `pip install bnn-lab`.
