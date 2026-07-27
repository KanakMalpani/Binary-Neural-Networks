# Roadmap (pointer)

**Canonical living plan (world-class BNN optimiser):**

- Root: [`../ROADMAP.md`](../ROADMAP.md)
- Docs twin: [`37_WORLD_CLASS_BNN_OPTIMISER_ROADMAP.md`](37_WORLD_CLASS_BNN_OPTIMISER_ROADMAP.md)

**When lost:** read root `ROADMAP.md` §0 → §7 → current phase → lowest unchecked TODO. Run `bnn repro`. Do not invent benches.

**Historical lab COMPLETE plan:** [`21_E2E_ROADMAP_COMPLETE_REPO.md`](21_E2E_ROADMAP_COMPLETE_REPO.md) (D1–D12 evidence in [`22_COMPLETION_REPORT.md`](22_COMPLETION_REPORT.md)).

## Historical MVP checklist (superseded detail → see `21`, then `ROADMAP.md`)

- [x] First-principles + SOTA + failure docs
- [x] Perfected concept (kill naive 32× pitch)
- [x] Requirements + ADR + gap register
- [x] STE Binary/Ternary layers + Bi-Real CNN
- [x] Native CPU XNOR-popcount GEMM (MSVC)
- [x] Measured kernel speedups + compression
- [x] Train/bench/export scripts
- [x] Linear wrap + hybrid FFN sketch + CIFAR proxy + energy/FGSM proxies

**Next:** follow root [`ROADMAP.md`](../ROADMAP.md) §10 / §11.2 (lowest unchecked). W2.T04/T05 portable SIMD **done** (`docs/41`).
