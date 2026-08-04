<<<<<<< HEAD
# RAPL energy spike (pointer)

| Field | Value |
|-------|-------|
| **Status** | Open PR / moonshot — not a golden floor |
| **Lane** | H (`lane/h-energy`, [PR #22](https://github.com/KanakMalpani/Binary-Neural-Networks/pull/22)) |
| **Main-tip anchors** | [`docs/14_HARDWARE_AND_ENERGY.md`](../14_HARDWARE_AND_ENERGY.md), [`results/energy_bound.json`](../../results/energy_bound.json) |

Board Joules remain **CLOSED-BY-PROXY** on Windows (`E = P · t` from wrap latency).
A full RAPL probe + `results/energy_rapl_spike.json` lands with Lane H — do **not** invent
measured Joules or treat this stub as a repro golden.

Until PR #22 merges, prefer the energy-bound JSON + hardware doc above for citations.
This file exists so KG / docs links stay portable without claiming the spike is on `main`.
=======
# RAPL / board Joules spike (M5)

| Field | Value |
|-------|-------|
| **Status** | **SPIKE DELIVERED** — Linux RAPL path + Windows CLOSED-BY-PROXY honesty |
| **Date** | 2026-08-04 |
| **Task** | Moonshot **M5** (Lane H) |
| **Code** | `bnn/energy/**`, `scripts/energy_bound_measured.py`, `scripts/energy_rapl_spike.py` |
| **Results** | `results/energy_bound.json`, `results/energy_rapl_spike.json` |

## Intent

Replace pure assumed-power folklore with:

1. **Linux:** read `/sys/class/powercap/{intel,amd}-rapl:*/energy_uj` when readable;
   optional timed busy-loop Joules via `scripts/energy_rapl_spike.py`.
2. **Windows / no powercap:** keep **`CLOSED-BY-PROXY`** — `E = P_assumed × t_measured`
   from wrap latencies — never pretend RAPL exists in the Python stdlib.

## Dual-metric honesty

| Quantity | Meaning |
|----------|---------|
| Pack **32×** | Theoretical uint64 compression — **not** energy |
| `energy_proxy` | Relative to FP (`fp=1.0`, `binary=E_bin/E_fp`) for Pareto |
| `MEASURED_RAPL` | Timed package Joules on a spike loop (not a golden floor) |
| Literature 55–82% / 71× | Vendor / paper anchors — **not** this board |

## How to run

```bash
# Always: bind wrap latencies → energy_bound + Pareto energy_proxy fields
bnn energy-bound
# or
python scripts/energy_bound_measured.py

# Moonshot probe (0 exit on Windows with CLOSED-BY-PROXY)
python scripts/energy_rapl_spike.py --from-bound results/wrap_demo.json

# Optional one-shot estimate with RAPL domain list
python scripts/energy_estimate.py --latency-s 0.01 --power-w 25 --probe-rapl
```

## Pareto wiring

`results/energy_bound.json` now includes:

```json
"energy_proxy": {
  "fp": 1.0,
  "binary": 0.148,
  "unit": "relative_to_fp",
  "note": "…"
}
```

Consumers (`scripts/pareto_report.py`, `bnn.eval.pareto`) can set point
`energy_proxy` from `energy_bound["energy_proxy"]["binary"]` without inventing
new bench shapes. Wrap e2e Joules remain latency×P unless a RAPL spike timed
the **same** workload.

## Acceptance (Lane H)

| Check | Result |
|-------|--------|
| `bnn/energy` RAPL detect + measure API | Yes |
| Windows CLOSED-BY-PROXY status string | Yes |
| `energy_bound` emits `energy_proxy` | Yes |
| Spike JSON committed / regenerable | Yes |
| No invented golden shapes | Yes |
| Thesis lock (no GPU 32× from `sign()`) | Yes |

## Residual (integrator)

- Flip ROADMAP moonshot **M5** / twin when merging — Lane H does not edit ROADMAP.
- On CI Linux runners without powercap permissions, status stays proxy / unavailable — still green.
- Full wrap-workload RAPL (same shapes as `wrap_demo`) needs a privileged Linux box; spike loop is pedagogy only.
>>>>>>> origin/lane/h-energy
