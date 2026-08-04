# PyPI publish (W8.T08)

**Status (2026-08-04):** packaging + wheels workflow ready; **live upload blocked
until Trusted Publishing is linked** (and a free PyPI project name is used).

## Why the distribution name is `bnn-lab`

The short name [`bnn`](https://pypi.org/project/bnn/) is **already taken** on
PyPI (Adrian Bulat’s unrelated binary-networks package, 0.1.2). Publishing
under that name would fail or collide.

| Surface | Name |
|---------|------|
| **PyPI / pip** | `bnn-lab` |
| **Import** | `import bnn` (unchanged) |
| **CLI** | `bnn` console script (unchanged) |

```bat
pip install bnn-lab
bnn repro
```

## Checklist

- [x] `pyproject.toml` distribution name `bnn-lab`, version synced with `bnn/_version.py`
- [x] `readme`, `license`, classifiers (3.11–3.13), project.urls
- [x] Console script `bnn` (import package remains `bnn`)
- [x] `constraints.txt` for reproducible installs
- [x] `wheels.yml` publish job gated on `workflow_dispatch` + `publish=true` + `environment: pypi`
- [ ] GitHub Environment `pypi` created on the repo (Settings → Environments)
- [ ] PyPI.org project + **Trusted Publisher** pointing at this repo / `wheels.yml` / `pypi` env
- [ ] TestPyPI dry-run (optional) then production publish via Actions

## Trusted Publishing (maintainer, once)

1. Create a free account on https://pypi.org (or log in).
2. Create GitHub Environment named **`pypi`** on
   `KanakMalpani/Binary-Neural-Networks` (Settings → Environments → New).
   Reviewers optional (useful on a public repo; free private repos may not support them).
3. On PyPI: **Publishing → Add a new pending publisher**:
   - PyPI project name: `bnn-lab`
   - Owner: `KanakMalpani`
   - Repository: `Binary-Neural-Networks`
   - Workflow: `wheels.yml`
   - Environment: `pypi`
4. In GitHub Actions → **wheels** → *Run workflow* → set **publish = true**.

The publish job uses OIDC (`id-token: write`); no long-lived API token in the repo.

## Local dry-run (no upload)

```bat
python -m pip install -U build twine
python -m build
twine check dist/*
```

Expect: `bnn_lab-0.3.0` (or `bnn-lab-0.3.0`) sdist + wheel, **PASSED**.

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

## Public-repo note

This repository is **public**. OpenSSF Scorecard’s public badge API and free
Code Scanning apply; supply-chain badges no longer need a private-repo caveat.
Trusted Publishing for **`bnn-lab`** remains a separate maintainer step above.
