# Lane H — Moonshot energy / RAPL (M5)

| Field | Value |
|-------|-------|
| **Branch** | `lane/h-energy` |
| **Base** | `main` @ `5910978` |
| **Status** | Delivered (spike + proxy honesty) |
| **Date** | 2026-08-04 |

## Owned paths touched

- `bnn/energy/**` (new)
- `scripts/energy_bound_measured.py`
- `scripts/energy_estimate.py`
- `scripts/energy_rapl_spike.py` (new)
- `docs/spikes/RAPL_ENERGY_SPIKE.md` (new)
- `docs/lanes/h.md` (this file)
- `results/energy_bound.json` / `.md` (regenerated fields)
- `results/energy_rapl_spike.json` (new)
- `tests/test_energy.py` (new)

## ROADMAP checkbox (for integrator)

| ID | Suggested state | Notes |
|----|-----------------|-------|
| **M5** RAPL / board Joules | `[~]` or `[x]` spike | Linux RAPL path + Windows CLOSED-BY-PROXY; wrap Joules still E=P×t unless spike timed |
| Profile / bench residual “RAPL Joules moonshot” | Update residual text → spike delivered | See `docs/spikes/RAPL_ENERGY_SPIKE.md` |
| `docs/MOONSHOT_DEFERRALS.md` M5 row | Soften blocker | Energy-proxy default remains; RAPL when OS allows |

## Acceptance evidence

1. `python -c "from bnn.energy import detect_rapl, build_energy_bound"`
2. `python scripts/energy_bound_measured.py` → `energy_proxy` + `rapl` keys
3. `python scripts/energy_rapl_spike.py` → exit 0; Windows: `CLOSED-BY-PROXY`
4. `pytest tests/test_energy.py -q`

## Residuals

- No portable Windows RAPL (by design) — proxy only.
- Timed RAPL is a busy-loop pedagogy spike, not a golden floor / wrap_demo identity.
- Lane H does **not** edit `ROADMAP.md` / twin (integrator merges checkbox flips).
