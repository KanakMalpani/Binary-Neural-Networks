# Calculated Speedup Model (formulas + numbers)

Machine context for local measurements: **PyTorch 2.12 CPU-only**, native MSVC
`binary_gemm_u64` with hardware `__popcnt64`, no CUDA.

---

## 1. Bit packing density

One weight in FP32 = 4 bytes = 32 bits.

Binary (±1) packed = **1 bit/weight**.

\[
\text{compression} = \frac{32}{1} = 32\times
\text{ (bytes: } 4 \rightarrow 0.125\text{)}
\]

Ternary \(\{-1,0,1\}\) ≈ \(\log_2 3 \approx 1.585\) bits if entropy-coded;
practical packing often uses **2 bits/weight** (I2_S) → **16×** vs FP32.

**Measured in this repo** (`scripts/export_check.py`):
`512×1024` binary weights → FP32 2,097,152 B → packed uint64 65,536 B → **32.00×**.

---

## 2. Arithmetic model: MAC → XNOR+popcount

For equal-length ±1 vectors of length \(N\):

\[
\langle w, x \rangle = N - 2\cdot \mathrm{popcount}(w_{\mathrm{bits}} \oplus x_{\mathrm{bits}})
\]

Pack \(N\) bits into \(W=\lceil N/64\rceil\) uint64 words.
Per output element cost ≈ \(W\) XOR + \(W\) popcount + 1 scale (vs \(N\) FP MACs).

**Programmatic calculators** (same formulas): `bnn.math.effective_ops_per_mac`,
`bnn.math.bytes_per_mac`, `bnn.math.amdahl_speedup` — see `docs/35_BINARY_MATH_EFFECTIVENESS.md`.

**Theoretical word-op reduction:**

\[
R_{\mathrm{arith}} = \frac{N}{\lceil N/64\rceil} \approx 64
\]

(for \(N\) divisible by 64). This is **not** wall-clock speedup.

| \(N\) | FP32 MACs / out | Word ops / out | \(R_{\mathrm{arith}}\) |
|------:|----------------:|---------------:|----------------------:|
| 2048 | 2048 | 32 | 64× |
| 4096 | 4096 | 64 | 64× |
| 8192 | 8192 | 128 | 64× |

---

## 3. Memory bandwidth bound (usually the real limit)

For batch-1 inference of \(y = Wx\), \(W\in\mathbb{R}^{M\times N}\):

\[
T_{\mathrm{mem}} \approx \frac{\mathrm{bytes}(W)}{B_{\mathrm{DRAM}}}
\]

\[
\frac{T_{\mathrm{mem,FP32}}}{T_{\mathrm{mem,bin}}} \approx 32
\]

if DRAM-bound and kernels stream packed weights.

If peak DRAM bandwidth \(B=50\) GB/s and \(M=N=8192\):

- FP32 weight bytes = \(8192^2 \times 4 = 268.4\) MB → \(T \approx 5.37\) ms
- Binary weight bytes = \(8192^2 / 8 = 8.39\) MB → \(T \approx 0.168\) ms
- Bandwidth-only speedup ≈ **32×** (upper bound before compute/Amdahl)

---

## 4. Amdahl end-to-end model (mandatory)

Let fraction \(f\) of runtime be replaceable matmuls, sped up by \(S_{\mathrm{kernel}}\):

\[
S_{\mathrm{e2e}} = \frac{1}{(1-f) + f/S_{\mathrm{kernel}}}
\]

| \(f\) | \(S_{\mathrm{kernel}}=4\) | \(S=8\) | \(S=32\) |
|------:|-------------------------:|--------:|---------:|
| 0.50 | 1.60× | 1.78× | 1.94× |
| 0.70 | 2.11× | 2.59× | 3.05× |
| 0.90 | 3.08× | 4.71× | 7.80× |
| 0.95 | 3.48× | 5.93× | 12.3× |

**Implication:** even a perfect 32× kernel cannot deliver 32× e2e unless matmuls
dominate. BitNet paper e2e latency gains (~1.2–4.1×) match this math.

---

## 5. Measured kernel results (this machine)

Fair protocol (`scripts/benchmark.py`): **weights pre-packed once** (deploy-time);
optional activation packing counted separately as e2e.

| Shape \(B\times N\times M\) | NumPy FP32 ms | Native compute ms | **\(S_{\mathrm{compute}}\)** | E2E+actpack ms | \(S_{\mathrm{e2e}}\) | Torch FP32 ms |
|----------------------------|--------------:|------------------:|------------------------------:|---------------:|---------------------:|--------------:|
| 128×2048×2048 | 18.01 | 4.13 | **4.36×** | 5.14 | **3.50×** | 3.63 |
| 64×4096×4096 | 22.02 | 6.10 | **3.61×** | 9.39 | **2.34×** | 11.25 |
| 32×8192×8192 | 113.87 | 12.26 | **9.29×** | 15.55 | **7.32×** | 42.15 |

Vs highly optimized **torch FP32** at 8192: \(42.15/12.26 \approx\) **3.44×** still.

**Observations:**
1. Speedup **grows with \(N\)** (packing + bandwidth).
2. Measured ≪ theoretical 64× word reduction — expected (single-thread C vs MT BLAS).
3. Broken protocol (re-packing \(W\) every call) falsely reported ~0.2× — fixed.
4. Pure NumPy popcount path remains much slower than native — **kernels matter**.

### Fake-binary trap

`torch.sign` + FP `linear` was **1.3–1.9× slower** than FP32 linear — simulation tax.

---

## 6. Energy sketch (literature, not measured here)

Using Horowitz-style tables as cited by BitNet b1.58 (arXiv:2402.17764):
INT add ≪ FP16 mul+add. Paper estimates **~71×** lower *arithmetic* energy for
matmul on 7nm for BitNet vs FP16 — **not** full SoC energy.

---

## 7. Literature wall-clock anchors (external)

| System | Hardware | Reported gain | Source |
|--------|----------|---------------|--------|
| Larq Compute Engine | Pixel phone ARM | **8.5–18.5×** vs FP | MLSys LCE paper |
| bitnet.cpp | ARM / x86 CPU | **1.37–5.07×** / **2.37–6.17×** | Microsoft BitNet infra |
| BitNet b1.58 3B | GPU (FasterTransformer+2bit) | latency **2.71×**, mem **3.55×** | arXiv:2402.17764 |
| BitNet b1.58 70B | 2×A100 | throughput **8.9×** | same |
| PyTorch BNN CUDA kernel (2019) | GPU | ~3× vs unoptimized control; **loses to cuDNN** | arXiv:1911.04477 |

---

## 8. Budget for a Bi-Real CNN on CPU (worked example)

Assume MNIST CNN: stem+head+BN = 25% time (FP), binary blocks = 75%.
Kernel \(S=4\):

\[
S_{\mathrm{e2e}} = 1/(0.25 + 0.75/4) = 2.29\times
\]

If only weights binary but activations FP (BinaryConnect): packing helps memory
but compute still needs mul — expect \(S_{\mathrm{kernel}}\sim 1.5\text{–}3\times\) bandwidth-only.

---

## 9. Decision rule from the math

1. If target is **CUDA datacenter LLM serving** → optimize \(f\) with **FP8/INT4**, not BNN.
2. If target is **CPU/edge inference** and matmuls dominate → **1-bit or 1.58-bit + packed kernels**.
3. Never advertise \(R_{\mathrm{arith}}\) or 32× compression as wall-clock without measuring \(S_{\mathrm{e2e}}\).
