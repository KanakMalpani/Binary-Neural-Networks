# Lane C — PyPI / attestations / badges

**Branch:** `lane/c-pypi-honesty` (residual after PR #26 / tag `v1.0.0`)
**Base:** `origin/main` @ `bc4aa7e`
**Owned paths:** `.github/workflows/wheels.yml`, `docs/PYPI*`, `docs/SBOM*`,
README packaging badge row, careful `pyproject.toml` packaging metadata,
`docs/lanes/c.md`.

## Thesis lock

Packed CPU / edge XNOR–popcount; never claim GPU 32× from `sign()`; no invented
golden shapes; dual-metric honesty.

## Status (2026-08-13)

| Item | State |
|------|--------|
| W8.T06 SBOM docs | Examples use `bnn_lab-1.0.0-*` |
| W8.T07 attestations in `wheels.yml` | Present on wheels + sdist |
| W8.T08 workflow gates | `publish=false` dry-run + `package-check`; OIDC only |
| README PyPI badge | Present (`pypi/v/bnn-lab`) — **404 until first upload** |
| GitHub env `pypi` | **Exists** |
| PyPI project `bnn-lab` | **Missing** (JSON API 404, 2026-08-13) |
| Trusted Publisher | **Human blocker** — pending publisher not registered |
| `publish=true` @ `v1.0.0` | [31031733046](https://github.com/KanakMalpani/Binary-Neural-Networks/actions/runs/31031733046) — **`invalid-publisher`** |
| Retry `publish=true` @ `v1.0.0` | [31698000321](https://github.com/KanakMalpani/Binary-Neural-Networks/actions/runs/31698000321) — **Windows `cp313` AV**; publish skipped |
| cibuildwheel skips | `*musllinux*`, `cp313-macosx_x86_64`, **`cp313-win_amd64`** (untested; numpy&lt;2 + py3.13 AV) |
| API-token publish | **Forbidden / not invented** |

## ROADMAP checkbox hints (do not flip until live)

- W8.T07 → keep `[x]`.
- W8.T08 → remain `[~]` until human Trusted Publisher + first OIDC upload.
- Do **not** claim `pip install bnn-lab` from PyPI.

## Human steps left

1. On https://pypi.org → **Publishing → Add a new pending publisher**:
   - Project: `bnn-lab`
   - Owner: `KanakMalpani`
   - Repo: `Binary-Neural-Networks`
   - Workflow: `wheels.yml`
   - Environment: `pypi`
2. Optional: add required reviewers on GitHub env `pypi`.
3. After this skip is on `main`: Actions → **wheels** → Run workflow →
   ref **`main`** (not tag `v1.0.0`) → **publish = true**.
4. Verify: `pip install bnn-lab==1.0.0` && `bnn repro` → `REPRO: PASS`.
5. Confirm README PyPI badge resolves to `1.0.0`.

## Dispatch log

| Run | Result |
|-----|--------|
| [30926016148](https://github.com/KanakMalpani/Binary-Neural-Networks/actions/runs/30926016148) | Dry-run **PASS** (`publish=false`) |
| [30946319438](https://github.com/KanakMalpani/Binary-Neural-Networks/actions/runs/30946319438) | Tag-push wheels **PASS** (no publish job) |
| [31031733046](https://github.com/KanakMalpani/Binary-Neural-Networks/actions/runs/31031733046) | `publish=true` @ `v1.0.0` → **`invalid-publisher`** |
| [31698000321](https://github.com/KanakMalpani/Binary-Neural-Networks/actions/runs/31698000321) | `publish=true` @ `v1.0.0` → Windows `cp313` crash; publish skipped |

**Still blocked for upload:** PyPI Trusted Publisher for `bnn-lab` (project 404). Do not invent an API-token path.
