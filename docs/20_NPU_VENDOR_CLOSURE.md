# NPU / DSP 1-bit support — closure evidence

## Verdict (decision-ready)

**Vendor NPUs are INT8/INT4/FP16-first. Native 1-bit / 1.58-bit XNOR is not a
drop-in path.** Custom kernels required (rare). Product decision tree: **INT8 on NPU**;
**1-bit on CPU (LCE / this repo) or FPGA (FINN)**; **ternary LLM via bitnet.cpp or custom Hexagon**.

| Vendor | Documented precisions | Native 1-bit BNN? | Source |
|--------|----------------------|-------------------|--------|
| **Qualcomm HTP** | INT4, INT8, INT16, FP16 | **No** in QNN stock | [Qualcomm AI hardware docs](https://docs.qualcomm.com/doc/80-63195-1/topic/AI-hardware-cores-accelerators.html) — HTP needs quantization to those types |
| **Qualcomm BitNet** | Ternary needs **custom** Hexagon kernels | Not stock | ENERZAi: QNN has no ternary matmul; custom 1.58 kernels on QCS6490 |
| **Arm Ethos-U** | INT8 (and 16×8 act/wt modes) | **No** | Arm blog: Ethos-U is 8-bit integer; TFLite INT8 + Vela |
| **Apple ANE / CoreML** | Weight 4/8-bit compress; runtime often float compute | **No** 1-bit XNOR | coremltools `linear_quantize_weights` n=4/8 |
| **This repo / LCE** | Packed binary on **CPU** | Yes | Local + Larq CE |

## Decision tree entry

```
Deploying on phone NPU?
├─ Stock SDK path → INT8 (or INT4 weights) via QNN / CoreML / Ethos+Vela
├─ Need BitNet ternary on Hexagon → budget custom kernels (non-trivial)
└─ Need classic W+A binary CNN → prefer CPU LCE or FPGA FINN, not stock NPU
```

## Gap status

**G_NPU / dim #15:** CLOSED-BY-PROXY with primary vendor documentation (above).
No further uncertainty for product thesis.
