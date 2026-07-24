# First Principles: Why Binary Ops Can Be Exponentially Faster

## The cost of a neural network step

A dense layer computes \( y = Wx + b \) for \( W \in \mathbb{R}^{M \times N} \), \( x \in \mathbb{R}^{N} \).
Each output element needs \( N \) multiply-accumulates (MACs). Total: \( MN \) MACs.

On modern hardware, wall-clock time is dominated by the **worse** of:

1. **Arithmetic throughput** (FLOPs/s or INT ops/s)
2. **Memory bandwidth** (bytes/s moving weights + activations)
3. **Latency / cache misses / kernel launch overhead**

For large models, inference is often **memory-bandwidth bound**: you spend more time
fetching weights from DRAM than doing math on them (especially decode / batch-1 LLM
inference).

## What “binary” changes

### 1. Arithmetic: MAC → XNOR + popcount

If weights and activations are in \(\{+1,-1\}\) with lab encoding bit0↦\(+1\), bit1↦\(-1\):

\[
\langle w, x \rangle = N - 2 \cdot \mathrm{popcount}(w_{\mathrm{bit}} \oplus x_{\mathrm{bit}})
\]

**Derivation (one line):** agreements contribute \(+1\), disagreements \(-1\), so
\((N-d)-d = N-2d\).  Pad bits must encode \(+1\) so they do not inflate \(d\).
Machine-checked in `bnn.math.xnor_dot_identity` / `docs/35_BINARY_MATH_EFFECTIVENESS.md`.

Equivalently: pack 64 bits into a word, XOR (or XNOR), then `popcount`.

| Regime | Op per element | Notes |
|--------|----------------|-------|
| FP32 MAC | 1 mul + 1 add | Needs FPU / Tensor Core |
| Binary packed | ~1/32 XNOR+popcount | 32 binary MACs per 32-bit word |
| Ternary \(\{-1,0,+1\}\) | add / skip / sub | BitNet b1.58; multiplies largely eliminated |

**Theoretical compute reduction** for fully binary layers: up to **~32×** fewer word ops
vs scalar FP32 (packing factor), plus cheaper ops.

### 2. Memory: 32× denser weights

| Precision | Bytes / weight | Relative size |
|-----------|----------------|---------------|
| FP32 | 4 | 1× |
| FP16 / BF16 | 2 | 2× smaller |
| INT8 | 1 | 4× |
| Ternary (~1.58-bit packed) | ~0.2 | ~16–20× |
| Binary (1-bit) | 0.125 | **32×** |

If you are bandwidth-bound, shrinking weights by 32× can yield **order-of-magnitude**
latency wins even before counting cheaper arithmetic — *if* the runtime actually stores
and streams packed bits (not “fake binary” floats that still look like FP32 tensors).

### 3. Energy (ASIC / process-node view)

Published energy models (e.g. Horowitz-style tables used in BitNet analyses) show INT
additions costing far less energy than FP16 mul+add. BitNet b1.58 reports ~**71×** lower
arithmetic energy for matmul on a 7nm model vs FP16 — *arithmetic only*, not end-to-end SoC.

### 4. Cache residency

A 32× smaller weight tensor fits in L2/L3 where FP32 spills to DRAM. That converts a
bandwidth-bound kernel into a compute-bound (or cache-hit) one — the real “exponential”
feeling on CPU / edge NPUs.

## Where the exponential claim is true vs false

### True (with evidence)

| Setting | Mechanism | Typical measured gain |
|---------|-----------|----------------------|
| CPU / ARM edge with packed kernels (Larq Compute Engine, bitnet.cpp, Litespark) | XNOR/popcount or ternary add kernels + bandwidth | **~2–18×** end-to-end; microbenchmarks higher |
| Custom ASIC / FPGA / CIM | Native binary/ternary datapath | Can approach theoretical packing factors |
| LLM decode (batch 1), memory-bound | Weight footprint collapse | BitNet: memory ↓ ~3–4×, latency ↓ ~1.2–4×, throughput ↑ up to ~9× at 70B |

### False / overstated (common failure)

| Claim | Reality |
|-------|---------|
| “Binary = 32× faster on any GPU” | **Wrong.** NVIDIA GPUs are optimized for FP16/TF32/INT8 Tensor Cores. Naive PyTorch `sign()` BNNs still run as FP32 and are often *slower*. |
| “Training is 32× faster” | **Wrong.** Training keeps latent FP weights + STE; often *slower* than FP. |
| “Drop-in replace Linear → BinaryLinear in PyTorch = speedup” | **Wrong** without a packed kernel / compiler path. Simulation ≠ acceleration. |

## Bandwidth math (back-of-envelope)

For batch-1 matmul \( y = Wx \):

- Bytes moved ≈ \( 4MN \) (FP32 weights) ignoring activations
- Binary packed ≈ \( MN/8 \) bytes

Speedup upper bound ≈ \( \min(\text{arith factor},\; \text{BW factor},\; \text{Amdahl}) \).

If only 70% of runtime is matmul and you speed matmul 10×:

\[
\text{end-to-end} \approx \frac{1}{0.3 + 0.7/10} \approx 2.7\times
\]

This is why BitNet reports **1.2–4×** latency (not 32×) even with large memory wins:
embeddings, attention, norms, and kernel overhead remain.

## First-principles takeaway

1. **Binary/ternary attacks the real bottleneck** (DRAM traffic + expensive MACs).
2. **Exponential theoretical op reduction** is packing × cheaper ops — realizable on
   CPU/NPU/ASIC, not automatically on CUDA Tensor Cores.
3. **Correct solution** = (architecture that stays accurate when binary) + (training recipe)
   + (**packed inference kernels**). Missing any one of these voids the speedup.
