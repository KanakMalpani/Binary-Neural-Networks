# Hardware, SIMD, Energy, Compilers

## 13. CPU SIMD XNOR–popcount realities

Modern CPUs expose `POPCNT` (x86) / `CNT` (ARM). Peak binary GOPS estimates (FINN talks):

| Core | Rough peak binary GOPS | vs INT8 (order) |
|------|------------------------|-----------------|
| ARM Cortex-A57 + NEON | ~460 / core | ≫ INT8 |
| x86 with scalar popcnt | hundreds / core | ≫ INT8 |

**This repo (MSVC `__popcnt64`, single-thread):** measured GEMM speedups **~3.6–9.3×** vs NumPy
FP32 at N=4096–8192; wrap demo layer **~2.6×** vs torch Linear. Multi-thread OpenMP/AVX2
is an **ACCEPTED-NON-GOAL** polish (G11) — single-thread already proves the thesis.

**Trap:** Python/NumPy SWAR popcount ≪ hardware popcnt (we measured ~0.04× vs BLAS).

## 14. GPU: Tensor Cores beat naive XNOR

NVIDIA datacenter GPUs optimize **TF32/FP16/BF16/FP8/INT8/INT4** Tensor Core paths.
Custom binary popcount kernels:

- Can beat *unoptimized* FP (BinaryNet GPU kernel era ~7× vs naive)
- Often **lose to cuDNN/cuBLAS** (arXiv:1911.04477)
- Win mainly via **memory** (BitNet 70B throughput 8.9× from batch capacity)

**Production GPU path:** torchao FP8 / INT4 AWQ + vLLM/TensorRT — not classic BNN.

## 15. NPU / DSP / mobile

| Platform | Typical path | Binary role |
|----------|--------------|-------------|
| Qualcomm Hexagon | HTP INT4/8/16/FP16 (stock QNN) | **No** native 1-bit; BitNet needs custom Hexagon |
| Apple ANE / CoreML | Weight 4/8-bit compress | **No** 1-bit XNOR |
| Arm Ethos-U | INT8 (Vela/TFLite) | **No** native BNN |
| Larq Compute Engine | ARM CPU BGEMM | **Real BNN** 8.5–18.5× on Pixel-class |
| MediaTek / Samsung NPU | Vendor INT8/FP16 | Use vendor PTQ |

**Rule:** mobile NPU wins are **INT8-first**. Classic W+A binary → CPU LCE / this repo / FPGA FINN.
Full vendor table + decision entry: **`docs/20_NPU_VENDOR_CLOSURE.md`** (G16 CLOSED-BY-PROXY).

## 16. FPGA / ASIC

**FINN** (Xilinx): Brevitas QAT → FINN IR → HLS dataflow; XNOR+popcount MVTU;
BN+sign → threshold folding; maxpool → OR on binary.

Reported (FPGA'17 FINN prototypes, Zynq):

- MNIST SFC-max: **12.3M FPS**, ~0.31 µs latency, wall <22 W → enormous FPS/W
- CIFAR CNV-max: ~22k FPS class results (paper suite)

ASIC studies (e.g. Aria10 FPGA vs 14nm ASIC BNN accelerators): FPGA ≫ CPU/GPU efficiency
for bit-level ops; ASIC still ahead of FPGA but locks design.

**Practicality 2026:** FINN/Brevitas viable for industrial vision edge; LLM-scale BNNs on FPGA
still research (on-chip memory limits).

## 17. Memory hierarchy / Amdahl (extends docs/06)

\[
S_{\mathrm{e2e}}=\frac{1}{(1-f)+f/S_k},\quad
T_{\mathrm{mem}}\approx\frac{\mathrm{bytes}(W)}{B_{\mathrm{DRAM}}}
\]

Binary weights: **32×** fewer bytes → often moves kernel from DRAM-bound → L2/L3-bound.
Ternary packed ~**16×** (2-bit) or ~**20×** entropy-coded.

Include: KV cache (LLMs), activations, non-binary layers, Python/runtime overhead.

## 18. Energy (Joules), not just latency

\[
E \approx P_{\mathrm{avg}}\times t_{\mathrm{infer}}
\]

Literature anchors:

- BitNet b1.58: ~**71×** lower *arithmetic* energy for matmul (7nm model tables) — not full SoC
- bitnet.cpp: energy **−55–82%** on CPU vs FP baselines (vendor reports)
- FINN: maximize FPS/W; slow clock + massive parallelism can beat fast clock for energy

**This machine:** board Joules **CLOSED-BY-PROXY** — `scripts/energy_bound_measured.py` binds
\(E=P\cdot t\) to **measured** wrap_demo latencies + assumed P brackets + lit anchors
(`results/energy_bound.*`). RAPL not required for decision thesis.

## Compiler / runtime matrix (preview → full in docs/16)

| Stack | Best for | Low-bit |
|-------|----------|---------|
| TensorRT / vLLM | NVIDIA serve | FP8/INT4 |
| OpenVINO | Intel CPU/iGPU | INT8 |
| ORT | Cross-platform | INT8/QDQ |
| ExecuTorch | Mobile | torchao INT4/8 |
| TFLite / LCE | Android BNN | 1-bit via LCE |
| CoreML | Apple | FP16/INT8 |
| TVM / IREE / MLIR | Custom lowerings | Research |
| FINN | FPGA QNN/BNN | 1–few bit |

## Residual

None material. OpenMP polish and full RAPL meters are ACCEPTED-NON-GOAL / proxy-closed (see `09`, `19`).
