# Binary Math Effectiveness — why / how / when

**Audience:** humans and other AIs reproducing this lab.  
**Thesis lock:** 32× is exact **weight compression** (uint64 pack) and a bandwidth upper bound — **not** a GPU wall-clock claim from `sign()`.

Companion code: `bnn/math/`, tests in `tests/test_math_identities.py`, STE math in `bnn/ste.py`.

---

## 1. The core identity (prove it)

Encoding (this repo, `bnn.kernels.packed` and `bnn.math.packing`):

| Bit | Value |
|-----|-------|
| 0 | \(+1\) |
| 1 | \(-1\) |

For \(x,w \in \{+1,-1\}^n\), let \(d = \mathrm{popcount}(x_{\mathrm{bit}} \oplus w_{\mathrm{bit}})\) (Hamming distance = disagreements). Then:

\[
\boxed{\langle x, w \rangle = n - 2d}
\]

**Derivation.** Each agreeing coordinate contributes \(+1\); each disagreement contributes \(-1\):

\[
\langle x,w\rangle = (n-d)(+1) + d(-1) = n - 2d.
\]

**XNOR form** (agreements \(a = n - d\)):

\[
\langle x,w\rangle = 2a - n = 2\cdot\mathrm{popcount}(\mathrm{XNOR}) - n.
\]

Hardware often prefers **XOR + popcount** (one fewer NOT than XNOR); see XOR-Net. This lab uses the XOR form.

### Edge cases

| Case | Rule |
|------|------|
| Padding to 64-bit words | Pad bits encode **+1** (bit 0). Identity uses logical \(n\), not \(64\cdot\mathrm{words}\). |
| \(n=0\) | Dot \(= 0\). |
| Channel scale \(\alpha=0\) | Scaled output is 0; unscaled identity still holds. |
| Non-±1 inputs | Lab packing projects `<=0 → -1`, `>0 → +1` before the identity. |

**Runnable proof:**

```bash
python scripts/math_identity_check.py
pytest tests/test_math_identities.py -q
```

---

## 2. Why binary math is “more effective”

Effectiveness = useful work per **joule**, per **cycle**, and per **DRAM byte** — not marketing FLOP counts.

### 2.1 Arithmetic

| Regime | Cost proxy per MAC-equivalent |
|--------|-------------------------------|
| FP32 | 1 FMA |
| Packed binary (64-bit words) | \(\lceil K/64\rceil\) XOR + popcount |

\[
R_{\mathrm{arith}} \approx \frac{K}{\lceil K/64\rceil} \approx 64
\quad (K \bmod 64 = 0)
\]

This is a **word-op reduction**, not wall-clock speedup. Horowitz-style tables (used in BitNet analyses) also show INT add ≪ FP mul energy on ASIC.

Code: `bnn.math.effective_ops_per_mac(k=...)`.

### 2.2 Memory / cache (usually the real lever)

| Precision | Bytes / weight | Compression vs FP32 |
|-----------|----------------|---------------------|
| FP32 | 4 | 1× |
| Binary | 0.125 | **32×** |

Cache-line density rises ~32× → more weights per L1/L2 line → fewer DRAM trips. For bandwidth-bound batch-1 matmul:

\[
\frac{T_{\mathrm{mem,FP32}}}{T_{\mathrm{mem,bin}}} \approx 32
\]

Code: `bnn.math.bytes_per_mac(...)`.

### 2.3 Information per DRAM byte

Streaming 32× more binary weights per byte means **more linear-algebra work per byte** when the kernel is packed. Fake-binary (`torch.sign` + FP GEMM) gets **none** of this — it still streams FP32.

---

## 3. When binary math is *less* effective

Use `bnn.math.when_binary_less_effective(...)`.

| Regime | Why binary loses |
|--------|------------------|
| Small \(K\) (≲256 on CPU) | Packing / call overhead dominates |
| Softmax attention | \(O(T^2)\) FP; Amdahl caps FFN-only wins |
| LayerNorm / RoPE / embeddings | Remain FP |
| NVIDIA Tensor Cores | FP16/BF16/INT8 already dense; naive bit kernels rarely win |
| Training | Latent FP + STE; often *slower* than FP train |

Amdahl (mandatory):

\[
S_{\mathrm{e2e}} = \frac{1}{(1-f) + f/S_{\mathrm{kernel}}}
\]

Even \(S_{\mathrm{kernel}}=32\) with \(f=0.7\) → \(S_{\mathrm{e2e}}\approx 3.05\times\).

---

## 4. Better binary math for *learning* (STE)

Forward stays hard `sign` (packing-friendly). Backward approximates \(\partial\,\mathrm{sign}\):

| Estimator | Backward | Cite / use |
|-----------|----------|------------|
| Clipped STE | \(1_{\lvert x\rvert\le 1}\) | BNN baseline |
| **ApproxSign** | Tent \(2-2\lvert x\rvert\) on \([-1,1]\) | Bi-Real Net (arXiv:1808.00278); `--approx-sign` on image/audio |
| **TanhSoft / EDE** | \(k t(1-\tanh^2(t x))\) | IR-Net (arXiv:1909.10788); `set_sign_mode("tanh_soft")` |
| ReAct / RSign | Learnable thresholds (angle sharpening) | Documented future; Bi-Real residual already in `BiRealBlock` |

**Gradient mismatch:** STE is discontinuous w.r.t. the true Dirac-of-sign; ApproxSign / EDE reduce mismatch near 0 (where most mass sits after BN). Small lab experiment:

```bash
python scripts/math_ste_compare.py
# → results/math_ste_compare.json
```

---

## 5. Ternary \(\{-1,0,1\}\) math

Absmean scale \(\gamma=\mathrm{mean}|w|\) (BitNet b1.58) is the practical closed-form scale used in `TernarySTE`. Ternary beats binary on **accuracy per bit** when:

\[
\frac{\mathrm{acc}_{\mathrm{tern}}}{b_{\mathrm{tern}}} > \frac{\mathrm{acc}_{\mathrm{bin}}}{1},
\quad b_{\mathrm{tern}}\in\{1.585\ \mathrm{(entropy)},\, 2\ \mathrm{(I2\_S pack)}\}.
\]

Code: `bnn.math.ternary_accuracy_per_bit(...)`.

---

## 6. Structured binary transforms (future note)

Walsh–Hadamard / binary codes can replace dense ±1 mats with fast butterfly transforms (FWHT) when the weight structure is constrained. **Not implemented** here — only document: optional future win when \(K\) is large and structure is acceptable; dense packed XNOR remains the default product path.

---

## 7. Before / after (this work)

| Before | After |
|--------|-------|
| Identity stated in docs / packed kernels | Proven in `bnn.math` + pytest across padded \(K\) |
| STE + ApproxSign only | + IR-Net-style tanh-soft / EDE schedule helpers |
| Effectiveness narrative scattered | Calculators + `docs/35` thresholds |
| STE comparison anecdotal | `results/math_ste_compare.json` |

---

## 8. How to run

```bash
pip install -e ".[dev]" -c constraints.txt
pytest tests/test_math_identities.py tests/test_ste.py -q
python scripts/math_identity_check.py
python scripts/math_ste_compare.py
```
