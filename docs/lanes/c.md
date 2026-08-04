# Lane C — PyPI / attestations / badges

**Branch:** `lane/c-release`  
**Base:** `main` @ `5910978`  
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
| W8.T07 attestations in `wheels.yml` | Present on wheels + sdist; verify command fixed to `bnn_lab-*` |
| W8.T08 workflow gates | `publish=false` dry-run + `package-check`; OIDC publish only |
| README PyPI badge | Added (`pypi/v/bnn-lab`) — shows not-found until first upload |
| `project.urls` | Added `PyPI` + `Publish runbook` |
| GitHub env `pypi` | **Exists** (0 protection rules) |
| PyPI project `bnn-lab` | **Missing** (JSON API 404) |
| Trusted Publisher | **Human blocker** — pending publisher not registered |
| `publish=true` | **Not dispatched** (would fail without publisher) |
| API-token publish | **Forbidden / not invented** |

## ROADMAP checkbox hints (for integrator)

- W8.T07 → keep `[x]` (hooks verified in workflow; e2e attestations confirmed on dry-run run once green).
- W8.T08 → remain `[~]` until human Trusted Publisher + first OIDC upload.
- WC-R / §7 “PyPI or documented why not yet” → documented in `docs/PYPI_PUBLISH.md` § Human blocker.

## Human steps left

1. On https://pypi.org → **Publishing → Add a new pending publisher**:
   - Project: `bnn-lab`
   - Owner: `KanakMalpani`
   - Repo: `Binary-Neural-Networks`
   - Workflow: `wheels.yml`
   - Environment: `pypi`
2. Optional: add required reviewers on GitHub env `pypi`.
3. Actions → **wheels** → Run workflow → **publish = true**.
4. Verify: `pip install bnn-lab` && `bnn repro` → `REPRO: PASS`.
5. Confirm README PyPI badge resolves to `0.3.0` (or next tag).

## Dry-run dispatch

```bat
gh workflow run wheels.yml --ref lane/c-release -f publish=false
```

- First dry-run: https://github.com/KanakMalpani/Binary-Neural-Networks/actions/runs/30922180976
  - Exposed macOS `delocate`/libomp bottle mismatch + Windows OpenMP `vcomp` ctypes load failure
  - Fixed via `BNN_NO_OPENMP=1` on macOS/Windows cibuildwheel (SIMD still ships)
- Re-dispatch after fix: _(filled after push)_
- Branch tip: _(filled after push)_
- PR: https://github.com/KanakMalpani/Binary-Neural-Networks/pull/16
- Conclusion: _(await re-run; publish must stay skipped)_

**Still blocked for upload:** PyPI Trusted Publisher for `bnn-lab` (project 404).
