# PyPI publish (W8.T08)

**Status (2026-08-13, Lane C residual):** packaging + `wheels.yml` (build /
attest / dry-run summary / OIDC publish) are ready. GitHub tag **`v1.0.0`**
exists. **`bnn-lab` is not on PyPI yet**
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

Those commands work **after** the first Trusted Publisher upload. Until then
use the [interim Git / Release install](#interim-install-git--release) below.

## Checklist

- [x] `pyproject.toml` distribution name `bnn-lab`, version synced with `bnn/_version.py` (**1.0.0**)
- [x] `readme`, `license`, classifiers (3.11–3.13), `project.urls` (incl. PyPI + publish runbook)
- [x] Console script `bnn` (import package remains `bnn`)
- [x] `constraints.txt` for reproducible installs
- [x] `wheels.yml` build matrix + W8.T07 attestations on wheels and sdist
- [x] `wheels.yml` `package-check` dry-run job (artifacts present; no upload)
- [x] `wheels.yml` publish job gated on `workflow_dispatch` + `publish=true` + `environment: pypi`
- [x] GitHub Environment `pypi` created on the repo (Settings → Environments)
- [ ] **PyPI.org pending Trusted Publisher** for `bnn-lab` / `wheels.yml` / env `pypi`
- [ ] First production publish via Actions (`publish=true`) after publisher is linked
- [ ] Post-upload: `pip install bnn-lab==1.0.0` + `bnn repro` on a clean venv

## Human blocker

Trusted Publishing is **not** configured for this project yet:

1. GitHub env **`pypi`** exists on `KanakMalpani/Binary-Neural-Networks`.
2. PyPI project **`bnn-lab` does not exist** (JSON API **404** as of 2026-08-13).
3. OIDC `publish=true` on tag `v1.0.0` failed with `invalid-publisher`
   ([run 31031733046](https://github.com/KanakMalpani/Binary-Neural-Networks/actions/runs/31031733046),
   2026-08-05): valid GitHub OIDC token, **no matching pending publisher**.
4. Retry on 2026-08-13:
   - [31698000321](https://github.com/KanakMalpani/Binary-Neural-Networks/actions/runs/31698000321)
     (`v1.0.0`): never reached publish — `windows-amd64` `cp313` wheel test
     crashed (`check_wheel_kernel.py` exit **3221225477** / `0xC0000005`).
   - [31700631120](https://github.com/KanakMalpani/Binary-Neural-Networks/actions/runs/31700631120)
     (`lane/c-pypi-honesty`, `cp313-win_amd64` skipped): matrix + `package-check`
     **success**; publish **`invalid-publisher`** again. Win+3.13 is skipped
     (see [Wheel matrix notes](#wheel-matrix-notes-lane-c-dry-run)).
5. **Do not** add a PyPI API token secret or invent a token-based publish path.
   `wheels.yml` is OIDC-only (`id-token: write`, `pypa/gh-action-pypi-publish`).

### Trusted Publishing (maintainer, once)

1. Create a free account on https://pypi.org (or log in).
2. Confirm GitHub Environment named **`pypi`** on
   `KanakMalpani/Binary-Neural-Networks` (already created). Optional:
   add required reviewers on that environment for public-repo safety.
3. On PyPI: **Publishing → Add a new pending publisher**:
   - PyPI project name: `bnn-lab`
   - Owner: `KanakMalpani`
   - Repository: `Binary-Neural-Networks`
   - Workflow filename: `wheels.yml` (not the workflow display name)
   - Environment: `pypi`
4. After the pending publisher is **saved**, GitHub Actions → **wheels** →
   *Run workflow*:
   - Use ref **`main`** (after this skip lands) or this docs branch — **not**
     tag `v1.0.0`. The frozen tag still builds crashing `cp313-win_amd64`.
   - Set **publish = true**.
5. First successful OIDC upload creates the PyPI project and unblocks the
   README PyPI badge. Then: `pip install bnn-lab==1.0.0` + `bnn repro`.

The publish job uses OIDC (`id-token: write`); no long-lived API token in the repo.

## Wheel matrix notes (Lane C dry-run)

cibuildwheel on **macOS/Windows** sets `BNN_NO_OPENMP=1` so wheels ship
portable SIMD kernels without Homebrew `libomp` (delocate/macOS 26 bottle
mismatch) or MSVC `vcomp*.dll` ctypes load failures. Linux still links
`libgomp` when available. Thesis win remains packed XNOR–popcount, not thread
scaling. `BNN_NO_OPENMP` is the wheel-build kill switch in `setup.py`;
`BNN_FORCE_OPENMP=1` (or `compile_native --openmp`) is the local macOS opt-in
when you knowingly want OpenMP despite the default-off policy.

**Skipped targets:**

| Pattern | Why |
|---------|-----|
| `*musllinux*` | PyTorch has no musl wheels — `bnn-lab` (hard-deps `torch`) cannot install |
| `cp313-macosx_x86_64` | No torch wheel — do not publish unloadable artifacts |
| `cp313-win_amd64` | numpy 1.26.x (`numpy>=1.24,<2`) + CPython 3.13.14: kernel smoke access-violates; do not ship an untested Win+3.13 wheel until numpy 2 is allowed |

Use manylinux / Win **cp311–312** / macOS arm64 (and cp311–312 on Intel Mac).
`hf` optional extra includes `safetensors` for Lane B packed export.

## Dry-run (Actions, preferred)

```bat
gh workflow run wheels.yml --ref main -f publish=false
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

Expect: `bnn_lab-1.0.0` sdist + wheel, **PASSED**.

### Historical note

`twine check` for the old distribution name `bnn-0.3.0` passed on 2026-07-25;
the rename is required solely because of the PyPI name collision.

## Interim install (Git / Release)

```bat
pip install "bnn-lab @ git+https://github.com/KanakMalpani/Binary-Neural-Networks.git@v1.0.0"
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
