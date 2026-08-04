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
