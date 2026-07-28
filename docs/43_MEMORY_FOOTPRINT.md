# Memory footprint + the arena that wasn't worth building (W13.T05 / W2.T07)

Memory is the *easy* win in low-bit inference — packing is exact and permanent,
unlike latency which depends on kernels, threads and shapes. That makes it the
easiest place to overclaim, so this report separates two numbers that are
routinely conflated:

| Number | Meaning |
|--------|---------|
| **resident** | Bytes actually held by the module's buffers, right now |
| **theoretical** | Bytes the *encoding* needs — the pack ratio |

`TernaryWeightOnlyLinear` is the clearest case: it stores `int8` but encodes
2 bits per weight. Reporting only the theoretical number would claim a 16×
saving that no machine ever observes.

## Usage

```bash
bnn memory --dim 1024 --ff 4096 --mode binary_xnor
```

```python
from bnn.memory import memory_report, forward_transient_bytes

report = memory_report(model)
d = report.to_dict()
d["tracked_resident_compression"]        # measured, weights only
d["tracked_theoretical_compression"]     # encoding ceiling
d["whole_model_resident_compression"]    # includes FP embeddings / attention / norms
```

## What the numbers look like

A 2-layer MLP (512 → 2048 → 512) wrapped `binary_xnor`:

| Metric | Value |
|---|---|
| FP32 resident | 8,398,848 B |
| wrapped resident | 282,624 B |
| **resident compression** | **29.68×** |
| theoretical compression | 32.00× |

The 29.68 vs 32.00 gap is the per-channel `alpha` and the FP32 `bias`, which stay
float. That gap is not a defect — it is the number you should quote.

### Whole-model is lower still

Wrap the FFN of a model with a real embedding table and the end-to-end figure
drops sharply, because embeddings, attention and norms are deliberately left FP:

- tracked (wrapped layers only): **3.33×**
- whole model: **2.07×**

Quote the whole-model number when talking about deployment size. A test enforces
that `whole_model_resident_compression <= tracked_resident_compression`.

### Transient buffers

The weight saving is permanent; a forward still allocates:

| Stage | 64 × 4096 → 4096 |
|---|---|
| packed activations | 32,768 B (32× smaller than FP32 activations) |
| FP32 output | 1,048,576 B |
| **total transient** | **1,081,344 B** |

Activation packing is nearly free; the FP32 **output** dominates. For edge sizing
that is the buffer to plan around, not the packed input.

---

## W2.T07 — memory arena: measured, then declined

The roadmap called for an arena/pool for packed buffers. Before building it, the
cost it would remove was measured:

| Shape | output alloc | GEMM | alloc share |
|---|---|---|---|
| 64 × 4096 × 4096 | 9.74 µs | 694.93 µs | **1.40 %** |
| 32 × 1024 × 1024 | 0.80 µs | 44.71 µs | **1.78 %** |
| 1 × 4096 × 4096 | 0.32 µs | 20.03 µs | **1.59 %** |

Allocation is **1.4–1.8 %** of kernel time. Against that, an arena carries a real
correctness hazard:

```python
out = torch.from_numpy(np.ascontiguousarray(y))   # shares memory with y
```

`ascontiguousarray` returns the *same* array when it is already contiguous, so
the returned tensor aliases the numpy buffer. Recycling that buffer on the next
forward would silently corrupt a tensor the caller still holds — a data-corruption
bug in exchange for ~1.5 %.

**Decision: not built.** Revisit only if profiling ever shows allocation above
~10 % of kernel time, and then with copy-on-return or an explicitly documented
aliasing contract.

This is recorded rather than silently skipped: "we measured it and it wasn't
worth it" is a result, and the next person should not have to re-derive it.
