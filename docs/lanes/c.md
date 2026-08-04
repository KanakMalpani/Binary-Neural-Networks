# Lane C — PyPI / attestations / badges

**Branch:** `lane/c-release`  
**Base:** `main` @ `f40f65e` (rebased)  
**Tip:** e28050a
**Owned paths:** `.github/workflows/wheels.yml`, `docs/PYPI*`, `docs/SBOM*`,
README packaging badge row, careful `pyproject.toml` packaging metadata,
`docs/lanes/c.md`.

## Thesis lock

Packed CPU / edge XNOR–popcount; never claim GPU 32× from `sign()`; no invented
golden shapes; dual-metric honesty.

## Status (2026-08-04)

| Item | State |
|------|--------|
| W8.T06 SBOM docs | Updated — links attestations + PyPI runbook |
| W8.T07 attestations in `wheels.yml` | Present on wheels + sdist; verify cmd `bnn_lab-*` |
| W8.T08 workflow gates | `publish=false` dry-run + `package-check`; OIDC only |
| README PyPI badge | Added (`pypi/v/bnn-lab`) — not-found until first upload |
| `project.urls` | Added `PyPI` + `Publish runbook` |
| `hf` extra | Soft dep `safetensors>=0.4,<1` (Lane B export) |
| cibuildwheel skips | `*musllinux*`, `cp313-macosx_x86_64` (no torch → unloadable) |
| macOS/Windows OpenMP | `BNN_NO_OPENMP=1` in cibuildwheel (SIMD still ships) |
| GitHub env `pypi` | **Exists** (0 protection rules) |
| PyPI project `bnn-lab` | **Missing** (JSON API 404) |
| Trusted Publisher | **Human blocker** — pending publisher not registered |
| Wheels dry-run | **PASS** [30926016148](https://github.com/KanakMalpani/Binary-Neural-Networks/actions/runs/30926016148) — all arches + `package-check`; `publish` skipped |
| `publish=true` | **Not dispatched** (would fail without publisher) |
| API-token publish | **Forbidden / not invented** |
| Stash `c-release-wip-not-lane-b` | Inspected — Lane C already on branch; left `bnn/wrap/metrics.py` for Lane A |

## ROADMAP checkbox hints (for integrator)

- W8.T07 → keep `[x]` (attestations ran on successful dry-run artifacts).
- W8.T08 → remain `[~]` until human Trusted Publisher + first OIDC upload.
- WC-R / §7 “PyPI or documented why not yet” → `docs/PYPI_PUBLISH.md` § Human blocker.

## Human steps left

1. On https://pypi.org → **Publishing → Add a new pending publisher**:
   - Project: `bnn-lab`
   - Owner: `KanakMalpani`
   - Repo: `Binary-Neural-Networks`
   - Workflow: `wheels.yml`
   - Environment: `pypi`
2. Optional: add required reviewers on GitHub env `pypi`.
3. Actions → **wheels** → Run workflow → **publish = true** (only after step 1).
4. Verify: `pip install bnn-lab` && `bnn repro` → `REPRO: PASS`.
5. Confirm README PyPI badge resolves to `0.3.0` (or next tag).

## Dry-run dispatch

```bat
gh workflow run wheels.yml --ref lane/c-release -f publish=false
```

| Run | Result |
|-----|--------|
| [30922180976](https://github.com/KanakMalpani/Binary-Neural-Networks/actions/runs/30922180976) | Cancelled after macOS libomp + Win vcomp failures |
| [30922905099](https://github.com/KanakMalpani/Binary-Neural-Networks/actions/runs/30922905099) / [30924842591](https://github.com/KanakMalpani/Binary-Neural-Networks/actions/runs/30924842591) | Partial — exposed torch gaps (cp313 mac x86 / musllinux) |
| [30926016148](https://github.com/KanakMalpani/Binary-Neural-Networks/actions/runs/30926016148) | **PASS** — sdist + 5 arch wheels + package-check; publish skipped |

**PR:** https://github.com/KanakMalpani/Binary-Neural-Networks/pull/16  

**Still blocked for upload:** PyPI Trusted Publisher for `bnn-lab` (project 404). Do not invent an API-token path.
