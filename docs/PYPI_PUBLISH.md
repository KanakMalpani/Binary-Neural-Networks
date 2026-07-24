# PyPI publish preparation (W8.T08)

**Status (2026-07-25):** packaging ready for **dry-run**; live upload only when
Trusted Publishing / API token is configured on the GitHub repo.

## Checklist

- [x] `pyproject.toml` name `bnn`, version synced with `bnn/_version.py`
- [x] `readme`, `license`, classifiers (3.11–3.13), project.urls
- [x] Console script `bnn`
- [x] `constraints.txt` for reproducible installs
- [ ] GitHub Environment `pypi` + Trusted Publisher (manual)
- [ ] TestPyPI dry-run from a clean checkout
- [ ] Production `twine upload` / `pypa/gh-action-pypi-publish`

### Twine dry-run (2026-07-25)

```
python -m build
twine check dist/*
```

Result: **PASSED** for `bnn-0.3.0` sdist + wheel.

## Why not auto-publish yet

Trusted Publishing is a one-time GitHub ↔ PyPI link owned by the maintainer.
Until that exists, release artifacts ship on **GitHub Releases** (+ SBOM).

## Install from Git (interim)

```bat
pip install "bnn @ git+https://github.com/KanakMalpani/Binary-Neural-Networks.git@v0.3.0"
```
