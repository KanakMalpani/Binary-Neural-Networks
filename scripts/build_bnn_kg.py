#!/usr/bin/env python3
"""Build Binary Neural Network knowledge graph artifacts.

Generates:
  knowledge_graph/bnn_kg.json
  knowledge_graph/bnn_kg.graphml

Run from repo root:  python scripts/build_bnn_kg.py
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "knowledge_graph"


def N(
    id: str,
    label: str,
    type: str,
    summary: str,
    sources: list[str],
    confidence: float,
    status: str = "established",
    **extra: Any,
) -> dict[str, Any]:
    node = {
        "id": id,
        "label": label,
        "type": type,
        "summary": summary,
        "sources": sources,
        "confidence": round(confidence, 2),
        "status": status,
    }
    node.update(extra)
    return node


def E(
    source: str,
    target: str,
    relation: str,
    evidence: list[str],
    weight: float = 1.0,
) -> dict[str, Any]:
    return {
        "source": source,
        "target": target,
        "relation": relation,
        "evidence": evidence,
        "weight": weight,
    }


def build_nodes() -> list[dict[str, Any]]:
    return [
        # ── Thesis / meta ──────────────────────────────────────────────
        N(
            "thesis_lock",
            "Thesis Lock: Packed CPU/edge XNOR + honest STE",
            "Claim",
            "Speedups come from packed kernels on CPU/edge; never claim GPU 32× from sign(); "
            "compression 32× ≠ latency; STE trains latents, inference uses pack+popcount.",
            ["ROADMAP.md", "AGENTS.md", "docs/08_ADR.md", "docs/05_PERFECTED_CONCEPT.md"],
            1.0,
            status="locked",
        ),
        N(
            "dual_metric_culture",
            "Honest Dual-Metric Culture",
            "Concept",
            "Always report theoretical word reduction / weight compression separately from "
            "measured wall-clock S_compute and S_e2e; never advertise 32×/64× as latency.",
            ["docs/06_CALCULATED_SPEEDUP_MODEL.md", "docs/FAIR_EVAL_PROTOCOL.md", "results/SUMMARY.md"],
            1.0,
        ),
        N(
            "bnn_lab_system",
            "bnn-lab (Binary Neural Network Optimiser Lab)",
            "System",
            "Installable PyTorch lab (v0.3.0): packed XNOR kernels, wrap/optimise, .bnnpack, "
            "vision/audio/seq lanes, repro gates, bridges to BitNet/GGUF/torchao.",
            ["README.md", "ROADMAP.md", "pyproject.toml"],
            1.0,
            version="0.3.0",
        ),
        N(
            "world_class_v1",
            "World-Class Optimiser v1.0 Bar",
            "Decision",
            "Acceptance gates WC-A/K/O/R/D/P in ROADMAP §1; until all pass, claim only lab/beta.",
            ["ROADMAP.md#1-definition-of-world-class-acceptance-bar"],
            0.95,
            status="target",
        ),
        # ── First principles ───────────────────────────────────────────
        N(
            "xnor_popcount_identity",
            "XNOR–Popcount Dot-Product Identity",
            "Concept",
            "For ±1 vectors with bit0↦+1, bit1↦−1: ⟨w,x⟩ = N − 2·popcount(w⊕x). "
            "Pad bits encode +1 so they do not inflate disagreement count.",
            ["docs/01_FIRST_PRINCIPLES.md", "docs/35_BINARY_MATH_EFFECTIVENESS.md", "bnn/math/"],
            1.0,
        ),
        N(
            "amdahl_law",
            "Amdahl Bound on End-to-End Speedup",
            "Concept",
            "S_e2e = 1/((1−f)+f/S_k). Non-binary layers (softmax, LN, attn, embed) cap e2e gains "
            "even when kernel S_k is large.",
            ["docs/01_FIRST_PRINCIPLES.md", "docs/06_CALCULATED_SPEEDUP_MODEL.md", "results/SUMMARY.md"],
            1.0,
        ),
        N(
            "bandwidth_bound",
            "Memory-Bandwidth Bound Inference",
            "Concept",
            "Large models / batch-1 decode often limited by DRAM bytes/s, not peak FLOPs. "
            "32× denser weights can move kernels from DRAM-bound to L2/L3-bound.",
            ["docs/01_FIRST_PRINCIPLES.md", "docs/14_HARDWARE_AND_ENERGY.md"],
            0.95,
        ),
        N(
            "compression_32x",
            "Exact 32× Weight Compression (uint64 pack)",
            "Metric",
            "Aligned binary pack: FP32 4 bytes → 1 bit = 32.00× exact. Gate in golden_floors.json.",
            ["tests/golden_floors.json", "results/benchmark.json", "AGENTS.md"],
            1.0,
        ),
        N(
            "theoretical_word_reduction_64x",
            "Theoretical ~64× Word-Op Reduction",
            "Metric",
            "64 binary MACs per uint64 XNOR+popcount vs scalar FP32 MAC counting — "
            "NOT a wall-clock claim.",
            ["docs/06_CALCULATED_SPEEDUP_MODEL.md", "results/benchmark.json"],
            1.0,
        ),
        N(
            "energy_proxy_ept",
            "Energy Proxy E = P · t",
            "Metric",
            "Board Joules closed-by-proxy: measured wrap latency × assumed power brackets + lit anchors. "
            "RAPL optional on Linux; Windows stays proxy.",
            ["results/energy_bound.json", "docs/14_HARDWARE_AND_ENERGY.md", "docs/spikes/RAPL_ENERGY_SPIKE.md"],
            0.9,
        ),
        # ── Failure modes ──────────────────────────────────────────────
        N(
            "fake_binary_sign",
            "Fake Binary: sign() + FP GEMM",
            "FailureMode",
            "PyTorch w.sign() @ x.sign() still uses FP32 GEMM — no packing → no speedup, "
            "often slowdown (extra sign kernels). Negative control in benches.",
            ["docs/03_FAILURE_ANALYSIS.md", "results/benchmark.json", "docs/06_CALCULATED_SPEEDUP_MODEL.md"],
            1.0,
        ),
        N(
            "accuracy_collapse",
            "Accuracy Collapse from Information Loss",
            "FailureMode",
            "sign destroys magnitude; deep binary stacks lose capacity (historically 10–20+ pp ImageNet).",
            ["docs/03_FAILURE_ANALYSIS.md", "docs/02_SOTA_SURVEY.md"],
            0.95,
        ),
        N(
            "ste_mismatch",
            "STE Gradient Mismatch",
            "FailureMode",
            "True ∂sign/∂x = 0 a.e.; STE pretends identity on |x|≤1 → suboptimal minima risk.",
            ["docs/03_FAILURE_ANALYSIS.md", "docs/13_TRAINING_QAT_DISTILL.md", "results/math_ste_compare.json"],
            0.95,
        ),
        N(
            "gpu_tensor_core_reality",
            "GPU Tensor Cores Beat Naive XNOR",
            "FailureMode",
            "On A100/H100, FP16/BF16/FP8/INT8 Tensor Cores dominate; unoptimized binary popcount "
            "rarely beats cuDNN/cuBLAS end-to-end.",
            ["docs/03_FAILURE_ANALYSIS.md", "docs/14_HARDWARE_AND_ENERGY.md", "arXiv:1911.04477"],
            0.95,
        ),
        N(
            "first_last_layer_trap",
            "Binarizing Stem/Head Trap",
            "FailureMode",
            "Stem/head are small compute but critical for accuracy; always keep FP or ≥8-bit.",
            ["docs/03_FAILURE_ANALYSIS.md", "bnn/models.py"],
            1.0,
        ),
        N(
            "ptq_ternary_llm_wipe",
            "PTQ-Only Ternary LLM Quality Wipe",
            "FailureMode",
            "Absmean PTQ of Llama→ternary without distill/CPT destroys chat quality. Need BitDistill-scale path.",
            ["docs/18_DECISION_TREE_AND_COMPLETE_ROADMAP.md", "docs/12_WRAPPER_AND_EXISTING_MODELS.md", "arXiv:2510.13998"],
            0.9,
        ),
        N(
            "npu_no_native_1bit",
            "Stock NPU: No Native 1-bit",
            "FailureMode",
            "Qualcomm HTP / Apple ANE / Arm Ethos-U are INT8/INT4/FP16-first; native XNOR not drop-in.",
            ["docs/20_NPU_VENDOR_CLOSURE.md", "docs/14_HARDWARE_AND_ENERGY.md"],
            1.0,
        ),
        # ── Classic papers / methods ───────────────────────────────────
        N(
            "paper_bnn_2016",
            "Binarized Neural Networks (Courbariaux et al. 2016)",
            "Paper",
            "Foundational W+A ±1 with STE; GPU kernel demo era.",
            ["arXiv:1602.02830", "docs/02_SOTA_SURVEY.md"],
            1.0,
            arxiv="1602.02830",
        ),
        N(
            "paper_xnornet",
            "XNOR-Net (Rastegari et al. ECCV 2016)",
            "Paper",
            "Channel-wise scaling α; claims ~58× faster conv ops and 32× memory on ImageNet AlexNet class.",
            ["arXiv:1603.05279", "docs/02_SOTA_SURVEY.md"],
            1.0,
            arxiv="1603.05279",
        ),
        N(
            "paper_bireal",
            "Bi-Real Net (Liu et al. ECCV 2018 / IJCV)",
            "Paper",
            "FP residual shortcuts around binary convs + ApproxSign; ResNet-18 class ~56.4% ImageNet top-1.",
            ["arXiv:1808.00278", "arXiv:1811.01335", "docs/02_SOTA_SURVEY.md"],
            1.0,
            arxiv="1808.00278",
        ),
        N(
            "paper_reactnet",
            "ReActNet (Liu et al. ECCV 2020)",
            "Paper",
            "RSign/RPReLU learnable distribution reshape + distillation; ~69.4% ImageNet top-1, "
            "~3 pp gap to FP MobileNet-scale twin at 87M OPs.",
            ["arXiv:2003.03488", "docs/02_SOTA_SURVEY.md", "docs/13_TRAINING_QAT_DISTILL.md"],
            1.0,
            arxiv="2003.03488",
        ),
        N(
            "paper_irnet",
            "IR-Net (Qin et al. CVPR 2020)",
            "Paper",
            "Information Retention Network; Error Decay Estimator (EDE) / tanh-soft STE schedule.",
            ["arXiv:1909.10788", "docs/13_TRAINING_QAT_DISTILL.md", "docs/35_BINARY_MATH_EFFECTIVENESS.md"],
            0.95,
            arxiv="1909.10788",
        ),
        N(
            "paper_xnornetpp",
            "XNOR-Net++ (Bulat & Tzimiropoulos BMVC 2019)",
            "Paper",
            "Learned fused scale factors instead of analytic α; up to ~6% ImageNet gain vs XNOR-Net.",
            ["arXiv:1909.13863", "docs/02_SOTA_SURVEY.md"],
            0.9,
            arxiv="1909.13863",
        ),
        N(
            "paper_bibert",
            "BiBERT: Accurate Fully Binarized BERT (Qin et al. 2022)",
            "Paper",
            "Full 1-bit W+embed+A BERT via Bi-Attention + Direction-Matching Distillation; "
            "~56× FLOPs / ~31× size vs FP BERT.",
            ["arXiv:2203.06390", "docs/02_SOTA_SURVEY.md"],
            0.95,
            arxiv="2203.06390",
        ),
        N(
            "paper_bitnet",
            "BitNet: Scaling 1-bit Transformers (Wang et al. 2023)",
            "Paper",
            "BitLinear drop-in for 1-bit weight Transformers trained from scratch; scaling law akin to FP.",
            ["arXiv:2310.11453", "docs/02_SOTA_SURVEY.md"],
            1.0,
            arxiv="2310.11453",
        ),
        N(
            "paper_bitnet_b158",
            "BitNet b1.58 (Ma et al. 2024)",
            "Paper",
            "Ternary {−1,0,1} weights; matches FP16 LLaMA-class from ~3B+ at equal tokens; "
            "memory ~3.55×, latency ~2.71× at 3B; 0 enables feature gating.",
            ["arXiv:2402.17764", "docs/02_SOTA_SURVEY.md", "docs/01_FIRST_PRINCIPLES.md"],
            1.0,
            arxiv="2402.17764",
        ),
        N(
            "paper_bitnet_cpp",
            "1-bit AI Infra / bitnet.cpp (Wang et al. 2024)",
            "Paper",
            "CPU ternary kernels: x86 2.37–6.17×, ARM 1.37–5.07× vs FP; energy −55–82% reported.",
            ["arXiv:2410.16144", "docs/23_BITNET_CPP_BRIDGE.md", "docs/02_SOTA_SURVEY.md"],
            1.0,
            arxiv="2410.16144",
        ),
        N(
            "paper_bitnet_a48",
            "BitNet a4.8 (Wang et al. 2024)",
            "Paper",
            "4-bit activations for 1-bit LLMs via hybrid quant+sparsify; faster INT4/FP4 kernels.",
            ["arXiv:2411.04965"],
            0.9,
            arxiv="2411.04965",
        ),
        N(
            "paper_bitnet_2b4t",
            "BitNet b1.58 2B4T Technical Report (2025)",
            "Paper",
            "First open native 1-bit LLM at 2B params trained on 4T tokens; HF weights + GPU/CPU inference.",
            ["arXiv:2504.12285", "docs/02_SOTA_SURVEY.md"],
            0.95,
            arxiv="2504.12285",
        ),
        N(
            "paper_bitdistill",
            "BitNet Distillation / BitDistill (Wu et al. 2025)",
            "Paper",
            "Convert off-the-shelf FP LLMs → 1.58-bit for downstream tasks via SubLN + MHA distill + CPT; "
            "up to 10× memory, 2.65× CPU inference.",
            ["arXiv:2510.13998", "docs/18_DECISION_TREE_AND_COMPLETE_ROADMAP.md"],
            0.95,
            arxiv="2510.13998",
        ),
        N(
            "paper_sparse_bitnet",
            "Sparse-BitNet (2026)",
            "Paper",
            "1.58-bit + N:M semi-structured sparsity friendly to Sparse Tensor Cores; "
            "extra ~1.3× claimed in survey notes — treat measured numbers as OpenGap until lab reproduces.",
            ["arXiv:2603.05168", "docs/02_SOTA_SURVEY.md"],
            0.7,
            arxiv="2603.05168",
            status="literature",
        ),
        N(
            "paper_larq_ce",
            "Larq Compute Engine (Bannink et al. 2020)",
            "Paper",
            "Fastest open BNN inference engine of its era; ARM BGEMM 8.5–18.5× vs FP on Pixel-class.",
            ["arXiv:2011.09398", "docs/02_SOTA_SURVEY.md", "docs/16_ECOSYSTEM_AND_TOOLING.md"],
            0.95,
            arxiv="2011.09398",
        ),
        N(
            "paper_finn",
            "FINN (Umuroglu et al. FPGA'17)",
            "Paper",
            "FPGA dataflow for BNNs: Brevitas QAT → FINN IR → HLS; MNIST SFC-max ~12.3M FPS prototypes.",
            ["docs/14_HARDWARE_AND_ENERGY.md", "docs/16_ECOSYSTEM_AND_TOOLING.md"],
            0.9,
            note="Classic FPGA'17; arXiv companion commonly cited as 1612.07119",
        ),
        N(
            "paper_litespark",
            "Litespark Inference (2026)",
            "Paper",
            "SIMD ternary CPU kernels; survey cites ~18× Apple Silicon / extreme x86 vs naive PyTorch — "
            "not reproduced in this lab (OpenGap for local numbers).",
            ["arXiv:2605.06485", "docs/02_SOTA_SURVEY.md"],
            0.65,
            arxiv="2605.06485",
            status="literature",
        ),
        N(
            "paper_awq",
            "AWQ: Activation-aware Weight Quantization",
            "Paper",
            "Production GPU INT4 path for LLMs; default alternative when binary wrap is wrong tool.",
            ["docs/24_GPU_INT4_FP8_LANE.md", "docs/18_DECISION_TREE_AND_COMPLETE_ROADMAP.md"],
            0.95,
        ),
        # ── Novel local papers ─────────────────────────────────────────
        N(
            "paper_b1_honest_speedup",
            "B1 — Stop Claiming 32× (Honest Speedup Accounting)",
            "Paper",
            "Novel claim: mandatory dual metrics + fake-binary negative control + Amdahl/energy proxies; "
            "compression/op-count ≠ wall-clock. Idea vault only.",
            [
                "docs/32_NOVEL_PAPER_CANDIDATES.md",
                "C:/00 Research Papers/BINARY_NEURAL_SERIES_README.md",
                "C:/00 Research Papers/Stop Claiming 32x Honest Speedup Accounting for Binary Neural Networks/README.md",
            ],
            0.85,
            status="idea_vault",
        ),
        N(
            "paper_b2_packed_xnor",
            "B2 — Packed XNOR on Commodity CPUs (Productization + Goldens)",
            "Paper",
            "Novel claim: MSVC/portable packed XNOR recipe + fair pre-pack benches + machine-checkable "
            "golden gates for humans/AI agents. Idea vault only.",
            [
                "docs/32_NOVEL_PAPER_CANDIDATES.md",
                "C:/00 Research Papers/Packed XNOR on Commodity CPUs Reproducible Productization and Golden Gates/README.md",
            ],
            0.85,
            status="idea_vault",
        ),
        N(
            "paper_b3_when_not",
            "B3 — When Not to Binarize (Decision Tree + Hybrid Wrap)",
            "Paper",
            "Novel claim: goal×hardware decision tree + hybrid FFN wrap evidence for honest wrapping "
            "vs INT4/FP8/GGUF / do-not-binarize. Idea vault only.",
            [
                "docs/32_NOVEL_PAPER_CANDIDATES.md",
                "C:/00 Research Papers/When Not to Binarize Decision Tree for Hybrid Low-Bit Wrapping/README.md",
            ],
            0.85,
            status="idea_vault",
        ),
        # ── Algorithms ─────────────────────────────────────────────────
        N(
            "algo_ste_clip",
            "Clipped STE (BinaryNet)",
            "Algorithm",
            "Backward: 1_|x|≤1. Default in bnn.ste binary_sign.",
            ["docs/13_TRAINING_QAT_DISTILL.md", "bnn/ste.py", "results/math_ste_compare.json"],
            1.0,
        ),
        N(
            "algo_approx_sign",
            "ApproxSign (Bi-Real)",
            "Algorithm",
            "Piecewise-polynomial / tent surrogate 2−2|x| on [−1,1]; better match to sharp teacher in lab compare.",
            ["arXiv:1808.00278", "docs/13_TRAINING_QAT_DISTILL.md", "results/math_ste_compare.json"],
            0.95,
        ),
        N(
            "algo_irnet_ede",
            "IR-Net EDE / TanhSoft",
            "Algorithm",
            "kt(1−tanh²(tx)) schedule; set_sign_mode('tanh_soft') / irnet_ede_schedule.",
            ["arXiv:1909.10788", "docs/13_TRAINING_QAT_DISTILL.md", "bnn/ste.py"],
            0.9,
        ),
        N(
            "algo_rsign_rprelu",
            "RSign / RPReLU (ReActNet)",
            "Algorithm",
            "Channel-wise learnable thresholds/slopes for activation distribution reshape; "
            "documented, not default in this repo.",
            ["arXiv:2003.03488", "docs/13_TRAINING_QAT_DISTILL.md"],
            0.9,
            status="documented_not_default",
        ),
        N(
            "algo_xnor_gemm",
            "Packed uint64 XNOR+Popcount GEMM",
            "Algorithm",
            "Pre-pack weights once; XOR+popcount accumulate; native MSVC/GCC/portable SIMD + NumPy fallback.",
            ["bnn/kernels/", "docs/41_PORTABLE_SIMD_KERNEL.md", "results/benchmark.json"],
            1.0,
        ),
        N(
            "algo_ternary_bitplane",
            "Ternary Bitplane Pack / GEMM",
            "Algorithm",
            "2-bit / bitplane ternary path for {−1,0,+1}; ~16× compression vs FP32 when packed.",
            ["bnn/ternary_pack.py", "results/ternary_pack.json", "tests/golden_floors.json"],
            0.9,
        ),
        N(
            "algo_hybrid_ffn_wrap",
            "Hybrid FFN-Only Wrap Protocol",
            "Algorithm",
            "Replace FFN Linear with packed binary after short STE QAT; skip embed/attn/lm_head.",
            ["results/hybrid_ffn_wrap.json", "docs/12_WRAPPER_AND_EXISTING_MODELS.md", "docs/33_ULTRA_WRAP_LAYER.md"],
            0.95,
        ),
        N(
            "algo_amdahl_calculator",
            "Amdahl / Speedup Calculators",
            "Algorithm",
            "Machine-checked helpers in bnn.math for S_e2e(f,S_k) and bandwidth bounds.",
            ["bnn/math/", "docs/35_BINARY_MATH_EFFECTIVENESS.md"],
            1.0,
        ),
        # ── Lab systems / tools ────────────────────────────────────────
        N(
            "sys_packed_kernels",
            "Native Packed Kernel Runtime",
            "System",
            "Win/Linux/macOS/ARM portable SIMD dispatch (AVX-512→AVX2→NEON→scalar); OpenMP threads; err=0 vs ±1 FP.",
            ["docs/41_PORTABLE_SIMD_KERNEL.md", "results/benchmark.json", "docs/34_COMPUTE_SPEEDUP.md"],
            1.0,
        ),
        N(
            "sys_optimise_api",
            "bnn.optimise / Public Optimiser API",
            "System",
            "Stable entry: calibrate → policy → optional QAT → pack → report; preferred over legacy wrap --ultra.",
            ["docs/api/optimise.md", "bnn/optimise.py", "docs/adr/0001_public_optimiser_api.md"],
            0.95,
        ),
        N(
            "sys_ultra_wrap",
            "Ultra Wrap Layer",
            "System",
            "Hybrid binary/ternary wrap with calibration, sensitivity, layer search sketch.",
            ["docs/33_ULTRA_WRAP_LAYER.md", "results/ultra_wrap.json", "bnn/wrap/"],
            0.9,
        ),
        N(
            "sys_bnnpack",
            ".bnnpack Codec v1",
            "System",
            "Portable packed weight container encode/decode CLI; v2 schema deferred (moonshot).",
            ["docs/api/codec.md", "bnn/codec/packfile.py", "docs/BNNPACK_V2_DESIGN.md"],
            0.9,
        ),
        N(
            "sys_repro_gates",
            "bnn repro + Golden Floors",
            "System",
            "Machine-checkable gates: compression 32×, native err=0, MNIST/CIFAR/audio floors.",
            ["AGENTS.md", "tests/golden_floors.json", "REPRODUCIBILITY.md", "scripts/repro_all.py"],
            1.0,
        ),
        N(
            "sys_seq_encdec",
            "Encoder–Decoder Seq Lane",
            "System",
            "Toy seq2seq with binary-capable layers; tutorial 06; pedagogy not production NLP.",
            ["docs/36_ENCODER_DECODER_AND_NEXT.md", "docs/tutorials/06_encoder_decoder.md", "results/seq2seq_encoder_decoder.json"],
            0.9,
        ),
        N(
            "sys_vision_cifar",
            "Vision / CIFAR Bi-Real Lane",
            "System",
            "CIFAR-10 Bi-Real CNN: FP 71.14% vs binary 61.14% (10 pp gap); ImageNet protocol stub only.",
            ["results/image_cifar.json", "docs/tutorials/04_image_cifar.md", "bnn/vision/"],
            0.95,
        ),
        N(
            "sys_audio_synth",
            "Audio Synthetic Tones Lane",
            "System",
            "Synthetic tone CNN canary (not ASR); binary can match/exceed FP on this toy task.",
            ["results/audio_synth.json", "docs/tutorials/05_audio.md", "bnn/audio/"],
            0.9,
        ),
        N(
            "sys_energy_module",
            "bnn.energy (Proxy + RAPL Spike)",
            "System",
            "Energy bound from wrap latency; Linux RAPL probe; Windows CLOSED-BY-PROXY.",
            ["bnn/energy/", "results/energy_bound.json", "results/energy_rapl_spike.json", "docs/spikes/RAPL_ENERGY_SPIKE.md"],
            0.9,
        ),
        N(
            "tool_bitnet_cpp",
            "bitnet.cpp",
            "Tool",
            "Microsoft official ternary LLM CPU(+GPU) inference; bridge recipes in scripts/bridges.",
            ["docs/23_BITNET_CPP_BRIDGE.md", "arXiv:2410.16144", "third_party/BITNET_PIN.md"],
            0.95,
        ),
        N(
            "tool_gguf_llamacpp",
            "GGUF / llama.cpp",
            "Tool",
            "Default local CPU LLM path (Q4_K_M etc.) when checkpoint is not BitNet.",
            ["docs/22_HF_TO_GGUF_GUIDE.md", "docs/18_DECISION_TREE_AND_COMPLETE_ROADMAP.md"],
            0.95,
        ),
        N(
            "tool_torchao",
            "torchao (FP8/INT4/INT8)",
            "Tool",
            "PyTorch native quant stack for GPU serve; bridge JSON in results/bridge_gpu_torchao.json.",
            ["docs/24_GPU_INT4_FP8_LANE.md", "results/bridge_gpu_torchao.json"],
            0.9,
        ),
        N(
            "tool_vllm_trt",
            "vLLM / TensorRT / SGLang",
            "Tool",
            "Production NVIDIA LLM serve; prefer FP8/AWQ-INT4 over classic BNN simulation.",
            ["docs/18_DECISION_TREE_AND_COMPLETE_ROADMAP.md", "docs/24_GPU_INT4_FP8_LANE.md"],
            0.9,
        ),
        N(
            "tool_larq",
            "Larq + Larq Compute Engine",
            "Tool",
            "TF/Keras BNN train + ARM packed deploy; Larq repo archived 2026 but still usable.",
            ["arXiv:2011.09398", "docs/16_ECOSYSTEM_AND_TOOLING.md", "docs/02_SOTA_SURVEY.md"],
            0.85,
        ),
        N(
            "tool_brevitas_finn",
            "Brevitas → FINN",
            "Tool",
            "PyTorch QAT export to FPGA BNN dataflow; doc-only in this repo (non-goal to ship).",
            ["docs/14_HARDWARE_AND_ENERGY.md", "docs/18_DECISION_TREE_AND_COMPLETE_ROADMAP.md"],
            0.85,
        ),
        N(
            "tool_tflite_openvino_ort",
            "TFLite / OpenVINO / ORT INT8",
            "Tool",
            "Default edge/CPU INT8 paths when retrain-for-binary is unavailable.",
            ["docs/18_DECISION_TREE_AND_COMPLETE_ROADMAP.md", "docs/16_ECOSYSTEM_AND_TOOLING.md"],
            0.9,
        ),
        # ── Hardware ───────────────────────────────────────────────────
        N(
            "hw_cpu_x86_popcnt",
            "x86 POPCNT / Portable SIMD CPU",
            "Hardware",
            "Primary honest demo target for this lab; MSVC __popcnt64 + OpenMP + AVX2/512 dispatch.",
            ["docs/14_HARDWARE_AND_ENERGY.md", "docs/41_PORTABLE_SIMD_KERNEL.md", "results/benchmark.json"],
            1.0,
        ),
        N(
            "hw_arm_neon",
            "ARM NEON / CNT",
            "Hardware",
            "Edge phone CPU path; LCE and bitnet.cpp report multi-×; portable NEON in docs/41.",
            ["docs/41_PORTABLE_SIMD_KERNEL.md", "docs/spikes/ARM_NEON_SPIKE.md", "arXiv:2011.09398"],
            0.9,
        ),
        N(
            "hw_nvidia_gpu",
            "NVIDIA Datacenter GPU (Tensor Cores)",
            "Hardware",
            "Prefer INT4/FP8 serve; forbid claiming 32× from sign() BNNs.",
            ["docs/14_HARDWARE_AND_ENERGY.md", "docs/24_GPU_INT4_FP8_LANE.md", "ROADMAP.md"],
            1.0,
        ),
        N(
            "hw_phone_npu",
            "Phone NPU (HTP / ANE / Ethos-U)",
            "Hardware",
            "INT8-first stock SDKs; custom Hexagon for BitNet ternary is non-trivial.",
            ["docs/20_NPU_VENDOR_CLOSURE.md"],
            1.0,
        ),
        N(
            "hw_fpga_finn",
            "FPGA via FINN",
            "Hardware",
            "Native binary datapath; extreme FPS/W on small nets; LLM-scale still research.",
            ["docs/14_HARDWARE_AND_ENERGY.md"],
            0.85,
        ),
        # ── Metrics / results ──────────────────────────────────────────
        N(
            "result_kernel_speedups",
            "Measured Kernel Speedups (benchmark.json)",
            "Result",
            "Prepacked compute vs NumPy FP32: ~12× (2048), ~24× (4096), ~29× (8192); err=0; "
            "fake_binary slower than torch FP (~1.4× slower ratio).",
            ["results/benchmark.json", "results/SUMMARY.md"],
            1.0,
        ),
        N(
            "result_openmp_scaling",
            "OpenMP Thread Scaling Curve",
            "Result",
            "Thread scaling recorded per shape; gains taper / can regress at 8 threads on some shapes "
            "(overhead / memory). Dual-metric: report threads with S_compute.",
            ["results/benchmark.json", "docs/34_COMPUTE_SPEEDUP.md"],
            0.95,
        ),
        N(
            "result_mnist",
            "MNIST Accuracies (committed)",
            "Result",
            "fp32_mlp 97.67%, binary_mlp 96.36%, ternary_mlp 97.16% — within golden floors.",
            ["results/train_results.json", "tests/golden_floors.json"],
            1.0,
        ),
        N(
            "result_cifar",
            "CIFAR-10 Bi-Real Gap",
            "Result",
            "FP CNN 71.14% vs Bi-Real binary 61.14% (10.00 pp) — canary not ImageNet SOTA.",
            ["results/image_cifar.json", "tests/golden_floors.json"],
            1.0,
        ),
        N(
            "result_audio",
            "Audio Synth Accuracies",
            "Result",
            "FP 94.5% vs binary 96.0% on synthetic tones; NOT production ASR.",
            ["results/audio_synth.json", "tests/golden_floors.json"],
            0.95,
        ),
        N(
            "result_wrap_e2e",
            "Wrap Demo E2E Latency",
            "Result",
            "FP 34.14 ms → wrapped 7.09 ms (e2e ~4.82×); weight compression 32×; cosine ~0.31 without QAT.",
            ["results/wrap_demo.json", "results/SUMMARY.md"],
            1.0,
        ),
        N(
            "result_hybrid_ffn",
            "Hybrid FFN Wrap Evidence",
            "Result",
            "Replaced ffn_fc1/fc2 only; 32× on replaced weights; skipped attn/embed/lm_head; "
            "STE sketch→packed closed architecture gap.",
            ["results/hybrid_ffn_wrap.json"],
            0.95,
        ),
        N(
            "result_energy_bound",
            "Energy Bound Proxy Result",
            "Result",
            "Latency-only same-power energy reduction ~4.82×; with assumed P brackets ~6.74×; "
            "board_joules_status CLOSED-BY-PROXY.",
            ["results/energy_bound.json"],
            0.9,
        ),
        N(
            "result_ste_compare",
            "STE / ApproxSign / TanhSoft Compare",
            "Result",
            "Gradient cosines and short train curves; ApproxSign closer to sharp teacher (0.83 vs STE 0.60).",
            ["results/math_ste_compare.json", "docs/35_BINARY_MATH_EFFECTIVENESS.md"],
            0.95,
        ),
        N(
            "metric_s_compute",
            "S_compute (Prepacked Kernel Speedup)",
            "Metric",
            "Wall-clock of packed GEMM vs FP baseline with weights pre-packed once.",
            ["docs/06_CALCULATED_SPEEDUP_MODEL.md", "docs/FAIR_EVAL_PROTOCOL.md"],
            1.0,
        ),
        N(
            "metric_s_e2e",
            "S_e2e (End-to-End Speedup)",
            "Metric",
            "Whole-forward wall-clock including act pack / non-binary layers.",
            ["docs/06_CALCULATED_SPEEDUP_MODEL.md", "results/wrap_demo.json"],
            1.0,
        ),
        # ── Datasets ───────────────────────────────────────────────────
        N(
            "data_mnist",
            "MNIST",
            "Dataset",
            "Primary tiny accuracy canary; floors in golden_floors.json.",
            ["docs/DATASET_CARDS.md", "tests/golden_floors.json"],
            1.0,
        ),
        N(
            "data_cifar10",
            "CIFAR-10",
            "Dataset",
            "Vision Bi-Real proxy; subset+epochs committed for repro.",
            ["docs/DATASET_CARDS.md", "results/image_cifar.json"],
            1.0,
        ),
        N(
            "data_audio_synth",
            "Synthetic Audio Tones",
            "Dataset",
            "Pedagogy dataset; not LibriSpeech/ASR.",
            ["docs/DATASET_CARDS.md", "results/audio_synth.json"],
            0.95,
        ),
        N(
            "data_imagenet",
            "ImageNet (protocol stub only)",
            "Dataset",
            "Full ImageNet SOTA is explicit non-goal; protocol folder stub for future.",
            ["docs/imagenet_protocol.md", "docs/MOONSHOT_DEFERRALS.md", "ROADMAP.md"],
            0.8,
            status="non_goal_gate",
        ),
        # ── Decision tree ──────────────────────────────────────────────
        N(
            "decision_wrap_tree",
            "Goal × Hardware Decision Tree",
            "Decision",
            "Central practitioner tree: GPU→FP8/AWQ; CPU LLM→bitnet.cpp/GGUF; edge vision→Bi-Real/LCE/FINN "
            "or INT8; phone NPU→INT8; convert LLM→BitDistill; research→this repo.",
            ["docs/18_DECISION_TREE_AND_COMPLETE_ROADMAP.md", "docs/25_ONEPAGER.md", "paper_b3_when_not"],
            1.0,
        ),
        N(
            "decision_prefer_optimise",
            "Prefer bnn optimise over wrap --ultra",
            "Decision",
            "Public API / docs should guide users to optimise; ultra remains legacy surface.",
            ["AGENTS.md", "docs/GUIDE_E2E.md", "docs/adr/0001_public_optimiser_api.md"],
            1.0,
        ),
        N(
            "decision_onnx_defer",
            "Defer Full ONNX Custom-Op Runtime",
            "Decision",
            "Keep .bnnpack + bridges; revisit ORT custom ops only if consumer demands measured dual metrics.",
            ["docs/MOONSHOT_DEFERRALS.md"],
            0.95,
        ),
        N(
            "decision_no_gpu_32x",
            "Forbidden: GPU 32× from sign()",
            "Decision",
            "Immutable non-goal / thesis lock forever.",
            ["ROADMAP.md", "AGENTS.md"],
            1.0,
            status="locked",
        ),
        # ── Roadmap / open gaps ────────────────────────────────────────
        N(
            "roadmap_v030",
            "v0.3.0 Lab Complete (Phases C+D)",
            "Decision",
            "Portable SIMD delivered; WC-K3 met; public optimiser preview; community OSS files present.",
            ["docs/40_ROADMAP_E2E_SESSION.md", "CHANGELOG.md", "ROADMAP.md"],
            0.95,
        ),
        N(
            "gap_wasm",
            "OpenGap: WASM SIMD Demo",
            "OpenGap",
            "Optional browser pedagogy (W2.T06); not blocking v1.0 core.",
            ["docs/MOONSHOT_DEFERRALS.md", "docs/spikes/WASM_SIMD.md"],
            0.8,
            status="open",
        ),
        N(
            "gap_bnnpack_v2",
            "OpenGap: .bnnpack v2 + safetensors",
            "OpenGap",
            "Schema ADR + hashes + ternary meta + safetensors export still deferred.",
            ["docs/MOONSHOT_DEFERRALS.md", "docs/BNNPACK_V2_DESIGN.md"],
            0.85,
            status="open",
        ),
        N(
            "gap_distill_integration",
            "OpenGap: Distill Integration (W3.T08)",
            "OpenGap",
            "distill_sketch.py exists; full BitDistill-scale integration not shipped.",
            ["docs/MOONSHOT_DEFERRALS.md", "results/distill_sketch.json"],
            0.85,
            status="open",
        ),
        N(
            "gap_pypi_trusted",
            "OpenGap: First PyPI Trusted Publisher Upload",
            "OpenGap",
            "wheels.yml + docs ready; first bnn-lab upload via Trusted Publishing pending manual link.",
            ["docs/PYPI_PUBLISH.md", "ROADMAP.md"],
            0.9,
            status="open",
        ),
        N(
            "gap_litespark_local",
            "OpenGap: Local Litespark / Sparse-BitNet Numbers",
            "OpenGap",
            "Survey cites extreme SIMD/sparse gains; this lab has not reproduced — no invented metrics.",
            ["docs/02_SOTA_SURVEY.md", "arXiv:2605.06485", "arXiv:2603.05168"],
            0.7,
            status="open",
        ),
        N(
            "gap_imagenet_sota",
            "OpenGap / Non-goal: Full ImageNet SOTA Schedule",
            "OpenGap",
            "Explicitly accepted non-goal for gates; protocol stub only.",
            ["ROADMAP.md", "docs/MOONSHOT_DEFERRALS.md"],
            0.95,
            status="accepted_non_goal",
        ),
        N(
            "gap_rapl_windows",
            "OpenGap: Windows Board Joules (RAPL)",
            "OpenGap",
            "Linux RAPL spike delivered; Windows remains energy-proxy; no fake RAPL claims.",
            ["docs/spikes/RAPL_ENERGY_SPIKE.md", "results/energy_rapl_spike.json"],
            0.9,
            status="closed_by_proxy_on_windows",
        ),
        N(
            "gap_venue_submit",
            "OpenGap: Venue Submission of B1–B3",
            "OpenGap",
            "Idea vault + sources only; no venue LaTeX yet; claims must match measured goldens.",
            ["docs/32_NOVEL_PAPER_CANDIDATES.md", "docs/PUBLICATION_PLAN.md"],
            0.85,
            status="open",
        ),
        N(
            "gap_reactnet_in_repo",
            "OpenGap: Full ReActNet RSign/RPReLU in bnn.ste",
            "OpenGap",
            "Documented in training docs; not default implementation — ApproxSign/STE shipped.",
            ["docs/13_TRAINING_QAT_DISTILL.md", "arXiv:2003.03488"],
            0.8,
            status="open",
        ),
        # ── Concepts (training / wrap) ─────────────────────────────────
        N(
            "concept_ste_vs_inference",
            "STE Training ≠ Inference Throughput",
            "Concept",
            "Training keeps FP latents + surrogate grads; measured speed needs packed inference kernels.",
            ["ROADMAP.md", "docs/05_PERFECTED_CONCEPT.md", "docs/13_TRAINING_QAT_DISTILL.md"],
            1.0,
        ),
        N(
            "concept_fp_shortcuts",
            "Full-Precision Residual Shortcuts",
            "Concept",
            "Bi-Real/ReActNet pattern: restore information past sign bottlenecks.",
            ["arXiv:1808.00278", "arXiv:2003.03488", "docs/03_FAILURE_ANALYSIS.md"],
            1.0,
        ),
        N(
            "concept_ternary_zero_gate",
            "Ternary Zero as Feature Gate",
            "Concept",
            "BitNet b1.58 insight: explicit 0 enables filtering and closes quality gap vs pure ±1.",
            ["arXiv:2402.17764", "docs/02_SOTA_SURVEY.md"],
            0.95,
        ),
        N(
            "concept_prepack_once",
            "Fair Bench: Pre-pack Weights Once",
            "Concept",
            "Re-packing every forward creates false slowdowns; goldens require prepacked timing.",
            ["docs/06_CALCULATED_SPEEDUP_MODEL.md", "paper_b2_packed_xnor", "results/benchmark.json"],
            1.0,
        ),
        N(
            "org_microsoft_bitnet",
            "Microsoft BitNet Team",
            "PersonOrg",
            "BitNet / b1.58 / bitnet.cpp / BitDistill / 2B4T line of work.",
            ["arXiv:2310.11453", "arXiv:2402.17764", "arXiv:2410.16144", "arXiv:2510.13998"],
            0.95,
        ),
        N(
            "org_plumerai_larq",
            "Plumerai / Larq",
            "PersonOrg",
            "Larq training + Larq Compute Engine BNN deploy stack.",
            ["arXiv:2011.09398", "docs/16_ECOSYSTEM_AND_TOOLING.md"],
            0.85,
        ),
        # ── Extra classic / LLM methods ────────────────────────────────
        N(
            "paper_binaryconnect",
            "BinaryConnect (Courbariaux et al. 2015)",
            "Paper",
            "Binary weights only, FP activations — memory win, limited compute win.",
            ["docs/02_SOTA_SURVEY.md", "docs/13_TRAINING_QAT_DISTILL.md"],
            0.95,
        ),
        N(
            "paper_dorefa",
            "DoReFa-Net (Zhou et al. 2016)",
            "Paper",
            "Flexible multi-bit QAT for weights and activations.",
            ["docs/02_SOTA_SURVEY.md", "docs/13_TRAINING_QAT_DISTILL.md"],
            0.9,
        ),
        N(
            "paper_fbi_llm",
            "FBI-LLM: Fully Binarized LLMs (Ma et al. 2024)",
            "Paper",
            "Scale fully binarized LLMs from scratch via autoregressive distillation.",
            ["arXiv:2407.07093"],
            0.8,
            arxiv="2407.07093",
            status="literature",
        ),
        N(
            "paper_bitnet_v2",
            "BitNet v2 (Wang et al. 2025)",
            "Paper",
            "H-BitLinear with online Hadamard transform for native 4-bit activations on 1-bit LLMs.",
            ["arXiv:2504.18415"],
            0.85,
            arxiv="2504.18415",
        ),
        N(
            "method_tmac",
            "T-MAC / LUT Ternary Kernels",
            "Method",
            "Lookup-table / conditional-add kernels for ternary matmul on CPU; complementary to bitnet.cpp.",
            ["docs/02_SOTA_SURVEY.md", "docs/16_ECOSYSTEM_AND_TOOLING.md"],
            0.75,
            status="literature",
        ),
        N(
            "method_surge",
            "SURGE / DPGC Learnable Surrogate Gradients",
            "Method",
            "ICML 2026 learnable dual-path surrogate gradients addressing STE mismatch; not in-repo.",
            ["docs/02_SOTA_SURVEY.md", "docs/03_FAILURE_ANALYSIS.md", "docs/13_TRAINING_QAT_DISTILL.md"],
            0.7,
            status="literature",
        ),
        N(
            "method_absmean_ptq",
            "Absmean PTQ for Ternary",
            "Method",
            "Naive absmean PTQ of FP→ternary — insufficient for chat LLMs; antithesis of BitDistill.",
            ["docs/18_DECISION_TREE_AND_COMPLETE_ROADMAP.md"],
            0.9,
        ),
        # ── Lab APIs / modalities ──────────────────────────────────────
        N(
            "sys_guide_e2e",
            "GUIDE_E2E User Narrative",
            "System",
            "Primary human path: install → repro → optimise → encode/decode → modalities → metrics.",
            ["docs/GUIDE_E2E.md", "docs/39_GUIDE_E2E_COMPLETION.md", "AGENTS.md"],
            1.0,
        ),
        N(
            "sys_wrap_policy",
            "Auto Wrap Policy (binary/ternary/skip)",
            "System",
            "policy.py chooses binary vs ternary vs skip with reasons; sensitivity + layer search sketches.",
            ["bnn/wrap/policy.py", "docs/42_QAT_AND_LAYER_SEARCH.md", "results/ultra_wrap.json"],
            0.9,
        ),
        N(
            "sys_calibrate_qat",
            "Calibration + Short QAT Path",
            "System",
            "wrap/calibrate.py + wrap/qat.py; short STE QAT improves cosine vs cold PTQ on demos.",
            ["bnn/wrap/calibrate.py", "bnn/wrap/qat.py", "docs/42_QAT_AND_LAYER_SEARCH.md", "docs/api/wrap.md"],
            0.9,
        ),
        N(
            "sys_profile_pareto",
            "Profile + Pareto + Memory Report",
            "System",
            "Dual-metric profiling, Pareto fronts, memory footprint report (arena measured & declined).",
            ["bnn/profile.py", "docs/43_MEMORY_FOOTPRINT.md", "results/profile.json"],
            0.9,
        ),
        N(
            "sys_hf_optimiser",
            "Hugging Face Optimiser UX",
            "System",
            "Tutorial 08 + optional hf tests: load → calibrate → policy → encode → report.",
            ["docs/tutorials/08_HF_OPTIMISER.md", "results/hf_tiny_wrap.json", "ROADMAP.md"],
            0.9,
        ),
        N(
            "sys_bridge_recipes",
            "Bridge Recipes (torchao / llama.cpp / bitnet)",
            "System",
            "scripts/bridges/* with committed JSON results — first-class 'use this instead when…'.",
            [
                "docs/23_BITNET_CPP_BRIDGE.md",
                "docs/24_GPU_INT4_FP8_LANE.md",
                "results/bridge_gpu_torchao.json",
                "results/bridge_cpu_llamacpp_bitnet.json",
            ],
            0.9,
        ),
        N(
            "result_ultra_wrap",
            "Ultra Wrap Suite Goldens",
            "Result",
            "Binary compression 32×; ternary 16×; ternary cosine ≥0.85; hybrid cosine ≥0.55 floors.",
            ["results/ultra_wrap.json", "tests/golden_floors.json"],
            0.95,
        ),
        N(
            "result_robustness_fgsm",
            "FGSM Robustness Canary",
            "Result",
            "Binary MLP drop comparable to FP on FGSM canary — not a robustness SOTA claim.",
            ["results/robustness_fgsm.json", "results/SUMMARY.md", "docs/17_EVALUATION_ROBUSTNESS_ECONOMICS.md"],
            0.85,
        ),
        N(
            "result_ternary_pack",
            "Ternary Pack Compression Result",
            "Result",
            "Committed ternary pack measurements; floor 16× exact when packed.",
            ["results/ternary_pack.json", "tests/golden_floors.json"],
            0.9,
        ),
        N(
            "metric_cosine_vs_fp",
            "Output Cosine vs FP Reference",
            "Metric",
            "Drop-in honesty metric; low cosine without QAT expected — never claim transparent wrap.",
            ["results/wrap_demo.json", "tests/golden_floors.json", "docs/FAIR_EVAL_PROTOCOL.md"],
            1.0,
        ),
        N(
            "metric_energy_proxy_ratio",
            "energy_proxy Relative Ratio",
            "Metric",
            "Pareto field: fp=1.0, binary=E_bin/E_fp from proxy or RAPL spike.",
            ["results/energy_bound.json", "docs/spikes/RAPL_ENERGY_SPIKE.md"],
            0.9,
        ),
        N(
            "concept_sim_vs_packed",
            "Sim Mode vs Packed Mode Separation",
            "Concept",
            "Trainable STE simulation must not be timed as acceleration; packed mode is the speed path.",
            ["docs/03_FAILURE_ANALYSIS.md", "docs/05_PERFECTED_CONCEPT.md"],
            1.0,
        ),
        N(
            "concept_word_vs_wallclock",
            "Op-Count Speedup vs Wall-Clock",
            "Concept",
            "Explicit contradiction pair: theoretical word reduction / packing factor ≠ measured latency.",
            ["results/SUMMARY.md", "docs/06_CALCULATED_SPEEDUP_MODEL.md", "paper_b1_honest_speedup"],
            1.0,
        ),
        N(
            "hw_wasm",
            "WebAssembly SIMD (deferred)",
            "Hardware",
            "Browser pedagogy target; spike notes exist; not shipped as gate.",
            ["docs/spikes/WASM_SIMD.md", "docs/MOONSHOT_DEFERRALS.md"],
            0.7,
            status="deferred",
        ),
        N(
            "decision_int8_npu_first",
            "INT8-First on Stock Phone NPU",
            "Decision",
            "Product rule from vendor closure: stock SDK → INT8/INT4; 1-bit → CPU LCE/lab or FPGA FINN.",
            ["docs/20_NPU_VENDOR_CLOSURE.md", "docs/18_DECISION_TREE_AND_COMPLETE_ROADMAP.md"],
            1.0,
        ),
        N(
            "decision_fair_eval",
            "Fair Eval Protocol (published shapes only)",
            "Decision",
            "No invented goldens; same shapes / same conclusions; floats need not be bit-identical.",
            ["docs/FAIR_EVAL_PROTOCOL.md", "AGENTS.md", "tests/golden_floors.json"],
            1.0,
        ),
        N(
            "gap_layer_search_full",
            "OpenGap: Full Layer-wise Binary/Ternary Search",
            "OpenGap",
            "Sensitivity scores exist (W3.T05); full search W3.T06 still enrichable.",
            ["docs/MOONSHOT_DEFERRALS.md", "docs/42_QAT_AND_LAYER_SEARCH.md"],
            0.8,
            status="open",
        ),
        N(
            "gap_bitnet_submodule",
            "OpenGap: bitnet.cpp as Git Submodule",
            "OpenGap",
            "Vendor pin/size; keep bridge recipe scripts instead of vendoring full tree.",
            ["docs/MOONSHOT_DEFERRALS.md", "third_party/BITNET_PIN.md"],
            0.85,
            status="open",
        ),
        N(
            "gap_fbi_llm_repro",
            "OpenGap: FBI-LLM / Full Binary LLM Repro",
            "OpenGap",
            "Literature path for fully binary LLMs; not a lab gate; cite only.",
            ["arXiv:2407.07093", "ROADMAP.md non-goals"],
            0.75,
            status="open",
        ),
        N(
            "tool_executorch",
            "ExecuTorch / Mobile Deployment",
            "Tool",
            "Mobile deploy path typically torchao INT4/8 — not classic 1-bit XNOR.",
            ["docs/14_HARDWARE_AND_ENERGY.md", "docs/16_ECOSYSTEM_AND_TOOLING.md"],
            0.8,
        ),
        N(
            "dataset_glue_downstream",
            "Downstream NLP Tasks (GLUE-class)",
            "Dataset",
            "BitDistill evaluates task-specific ternary FT; not run as lab golden.",
            ["arXiv:2510.13998"],
            0.7,
            status="literature",
        ),
    ]


def build_edges() -> list[dict[str, Any]]:
    return [
        # Thesis structure
        E("thesis_lock", "dual_metric_culture", "requires", ["ROADMAP.md", "AGENTS.md"], 1.0),
        E("thesis_lock", "algo_xnor_gemm", "requires", ["ROADMAP.md"], 1.0),
        E("thesis_lock", "decision_no_gpu_32x", "part_of", ["ROADMAP.md"], 1.0),
        E("thesis_lock", "fake_binary_sign", "contradicts", ["docs/03_FAILURE_ANALYSIS.md"], 1.0),
        E("bnn_lab_system", "thesis_lock", "implements", ["README.md", "ROADMAP.md"], 1.0),
        E("bnn_lab_system", "world_class_v1", "part_of", ["ROADMAP.md"], 0.8),
        E("dual_metric_culture", "compression_32x", "measured_on", ["results/SUMMARY.md"], 1.0),
        E("dual_metric_culture", "metric_s_compute", "requires", ["docs/FAIR_EVAL_PROTOCOL.md"], 1.0),
        E("dual_metric_culture", "metric_s_e2e", "requires", ["docs/FAIR_EVAL_PROTOCOL.md"], 1.0),
        E("compression_32x", "theoretical_word_reduction_64x", "contradicts", [
            "results/SUMMARY.md — same table: do not claim either as e2e latency"
        ], 0.9),
        E("theoretical_word_reduction_64x", "metric_s_e2e", "contradicts", [
            "docs/06 — op-count ≠ wall-clock"
        ], 1.0),
        # First principles
        E("xnor_popcount_identity", "algo_xnor_gemm", "derived_from", ["docs/35_BINARY_MATH_EFFECTIVENESS.md"], 1.0),
        E("xnor_popcount_identity", "compression_32x", "improves", ["docs/01_FIRST_PRINCIPLES.md"], 0.8),
        E("amdahl_law", "metric_s_e2e", "requires", ["docs/06_CALCULATED_SPEEDUP_MODEL.md"], 1.0),
        E("amdahl_law", "result_wrap_e2e", "explains", ["results/SUMMARY.md"], 0.9),
        E("bandwidth_bound", "compression_32x", "improves", ["docs/01_FIRST_PRINCIPLES.md"], 0.85),
        E("bandwidth_bound", "hw_cpu_x86_popcnt", "recommends_for", ["docs/14_HARDWARE_AND_ENERGY.md"], 0.8),
        E("algo_amdahl_calculator", "amdahl_law", "implements", ["bnn/math/"], 1.0),
        # Failures
        E("fake_binary_sign", "result_kernel_speedups", "measured_on", [
            "results/benchmark.json fake_binary_vs_torch_fp32 ~1.44 (slower)"
        ], 1.0),
        E("fake_binary_sign", "algo_xnor_gemm", "alternative_to", ["docs/03_FAILURE_ANALYSIS.md"], 1.0),
        E("fake_binary_sign", "gpu_tensor_core_reality", "part_of", ["docs/03_FAILURE_ANALYSIS.md"], 0.7),
        E("accuracy_collapse", "concept_fp_shortcuts", "blocked_by", ["docs/03_FAILURE_ANALYSIS.md"], 0.9),
        E("accuracy_collapse", "first_last_layer_trap", "part_of", ["docs/03_FAILURE_ANALYSIS.md"], 0.8),
        E("ste_mismatch", "algo_ste_clip", "requires", ["docs/13_TRAINING_QAT_DISTILL.md"], 0.7),
        E("ste_mismatch", "algo_approx_sign", "improves", ["results/math_ste_compare.json"], 0.85),
        E("ste_mismatch", "algo_irnet_ede", "improves", ["docs/13_TRAINING_QAT_DISTILL.md"], 0.8),
        E("ptq_ternary_llm_wipe", "paper_bitdistill", "blocked_by", ["arXiv:2510.13998"], 0.9),
        E("npu_no_native_1bit", "hw_phone_npu", "measured_on", ["docs/20_NPU_VENDOR_CLOSURE.md"], 1.0),
        E("npu_no_native_1bit", "tool_tflite_openvino_ort", "recommends_for", ["docs/20_NPU_VENDOR_CLOSURE.md"], 0.9),
        E("gpu_tensor_core_reality", "tool_torchao", "recommends_for", ["docs/24_GPU_INT4_FP8_LANE.md"], 0.95),
        E("gpu_tensor_core_reality", "paper_awq", "recommends_for", ["docs/18_DECISION_TREE_AND_COMPLETE_ROADMAP.md"], 0.9),
        # Papers lineage
        E("paper_bnn_2016", "algo_ste_clip", "introduces", ["arXiv:1602.02830"], 1.0),
        E("paper_xnornet", "xnor_popcount_identity", "popularizes", ["arXiv:1603.05279"], 0.9),
        E("paper_xnornet", "paper_xnornetpp", "improved_by", ["arXiv:1909.13863"], 0.8),
        E("paper_bireal", "concept_fp_shortcuts", "introduces", ["arXiv:1808.00278"], 1.0),
        E("paper_bireal", "algo_approx_sign", "introduces", ["arXiv:1808.00278"], 1.0),
        E("paper_bireal", "sys_vision_cifar", "implements", ["results/image_cifar.json"], 0.85),
        E("paper_reactnet", "algo_rsign_rprelu", "introduces", ["arXiv:2003.03488"], 1.0),
        E("paper_reactnet", "paper_bireal", "improves", ["arXiv:2003.03488"], 0.9),
        E("paper_reactnet", "gap_reactnet_in_repo", "not_fully_implemented_in", ["docs/13_TRAINING_QAT_DISTILL.md"], 0.8),
        E("paper_irnet", "algo_irnet_ede", "introduces", ["arXiv:1909.10788"], 1.0),
        E("paper_bibert", "paper_bnn_2016", "cites", ["arXiv:2203.06390"], 0.7),
        E("paper_bitnet", "org_microsoft_bitnet", "authored_by", ["arXiv:2310.11453"], 1.0),
        E("paper_bitnet_b158", "paper_bitnet", "improves", ["arXiv:2402.17764"], 1.0),
        E("paper_bitnet_b158", "concept_ternary_zero_gate", "introduces", ["arXiv:2402.17764"], 1.0),
        E("paper_bitnet_b158", "algo_ternary_bitplane", "motivates", ["docs/02_SOTA_SURVEY.md"], 0.7),
        E("paper_bitnet_cpp", "tool_bitnet_cpp", "implements", ["arXiv:2410.16144"], 1.0),
        E("paper_bitnet_cpp", "paper_bitnet_b158", "requires", ["arXiv:2410.16144"], 0.9),
        E("paper_bitnet_a48", "paper_bitnet_b158", "improves", ["arXiv:2411.04965"], 0.85),
        E("paper_bitnet_2b4t", "paper_bitnet_b158", "derived_from", ["arXiv:2504.12285"], 0.9),
        E("paper_bitdistill", "paper_bitnet_b158", "requires", ["arXiv:2510.13998"], 0.9),
        E("paper_bitdistill", "ptq_ternary_llm_wipe", "mitigates", ["arXiv:2510.13998"], 0.95),
        E("paper_sparse_bitnet", "paper_bitnet_b158", "improves", ["arXiv:2603.05168"], 0.7),
        E("paper_sparse_bitnet", "gap_litespark_local", "blocked_by", ["GAPS: not locally reproduced"], 0.8),
        E("paper_larq_ce", "tool_larq", "implements", ["arXiv:2011.09398"], 1.0),
        E("paper_larq_ce", "hw_arm_neon", "measured_on", ["arXiv:2011.09398"], 0.9),
        E("paper_finn", "tool_brevitas_finn", "implements", ["docs/14_HARDWARE_AND_ENERGY.md"], 0.9),
        E("paper_finn", "hw_fpga_finn", "recommends_for", ["docs/14_HARDWARE_AND_ENERGY.md"], 1.0),
        E("paper_litespark", "gap_litespark_local", "blocked_by", ["docs/02_SOTA_SURVEY.md"], 0.85),
        E("paper_awq", "tool_torchao", "alternative_to", ["docs/24_GPU_INT4_FP8_LANE.md"], 0.7),
        E("paper_awq", "tool_vllm_trt", "recommends_for", ["docs/18_DECISION_TREE_AND_COMPLETE_ROADMAP.md"], 0.85),
        # Novel papers ↔ lab
        E("paper_b1_honest_speedup", "dual_metric_culture", "derived_from", ["docs/32_NOVEL_PAPER_CANDIDATES.md"], 1.0),
        E("paper_b1_honest_speedup", "fake_binary_sign", "requires", ["B1 README"], 1.0),
        E("paper_b1_honest_speedup", "result_kernel_speedups", "measured_on", ["results/benchmark.json"], 1.0),
        E("paper_b1_honest_speedup", "energy_proxy_ept", "requires", ["results/energy_bound.json"], 0.9),
        E("paper_b1_honest_speedup", "gap_venue_submit", "blocked_by", ["docs/32_NOVEL_PAPER_CANDIDATES.md"], 0.8),
        E("paper_b2_packed_xnor", "sys_packed_kernels", "derived_from", ["docs/32_NOVEL_PAPER_CANDIDATES.md"], 1.0),
        E("paper_b2_packed_xnor", "sys_repro_gates", "requires", ["AGENTS.md", "tests/golden_floors.json"], 1.0),
        E("paper_b2_packed_xnor", "concept_prepack_once", "requires", ["B2 README"], 1.0),
        E("paper_b2_packed_xnor", "result_mnist", "measured_on", ["results/train_results.json"], 0.7),
        E("paper_b3_when_not", "decision_wrap_tree", "derived_from", ["docs/18_DECISION_TREE_AND_COMPLETE_ROADMAP.md"], 1.0),
        E("paper_b3_when_not", "algo_hybrid_ffn_wrap", "measured_on", ["results/hybrid_ffn_wrap.json"], 1.0),
        E("paper_b3_when_not", "npu_no_native_1bit", "requires", ["docs/20_NPU_VENDOR_CLOSURE.md"], 0.9),
        E("paper_b3_when_not", "tool_gguf_llamacpp", "recommends_for", ["docs/18_DECISION_TREE_AND_COMPLETE_ROADMAP.md"], 0.8),
        # Algorithms ↔ systems
        E("algo_xnor_gemm", "sys_packed_kernels", "implements", ["bnn/kernels/", "results/benchmark.json"], 1.0),
        E("algo_xnor_gemm", "result_kernel_speedups", "measured_on", ["results/benchmark.json"], 1.0),
        E("algo_xnor_gemm", "hw_cpu_x86_popcnt", "requires", ["docs/41_PORTABLE_SIMD_KERNEL.md"], 0.95),
        E("algo_ternary_bitplane", "sys_ultra_wrap", "part_of", ["results/ultra_wrap.json"], 0.7),
        E("algo_hybrid_ffn_wrap", "result_hybrid_ffn", "measured_on", ["results/hybrid_ffn_wrap.json"], 1.0),
        E("algo_hybrid_ffn_wrap", "sys_ultra_wrap", "part_of", ["docs/33_ULTRA_WRAP_LAYER.md"], 0.8),
        E("algo_ste_clip", "concept_ste_vs_inference", "part_of", ["docs/13_TRAINING_QAT_DISTILL.md"], 0.8),
        E("algo_approx_sign", "result_ste_compare", "measured_on", ["results/math_ste_compare.json"], 1.0),
        E("algo_irnet_ede", "result_ste_compare", "measured_on", ["results/math_ste_compare.json"], 0.9),
        E("algo_rsign_rprelu", "gap_reactnet_in_repo", "blocked_by", ["docs/13_TRAINING_QAT_DISTILL.md"], 0.85),
        # Systems
        E("sys_optimise_api", "sys_ultra_wrap", "improves", ["docs/adr/0001_public_optimiser_api.md"], 0.85),
        E("sys_optimise_api", "decision_prefer_optimise", "implements", ["AGENTS.md"], 1.0),
        E("sys_optimise_api", "sys_bnnpack", "requires", ["docs/api/optimise.md"], 0.7),
        E("sys_bnnpack", "gap_bnnpack_v2", "blocked_by", ["docs/MOONSHOT_DEFERRALS.md"], 0.8),
        E("sys_repro_gates", "result_mnist", "measured_on", ["tests/golden_floors.json"], 1.0),
        E("sys_repro_gates", "result_cifar", "measured_on", ["tests/golden_floors.json"], 1.0),
        E("sys_repro_gates", "result_audio", "measured_on", ["tests/golden_floors.json"], 1.0),
        E("sys_repro_gates", "compression_32x", "requires", ["AGENTS.md"], 1.0),
        E("sys_vision_cifar", "data_cifar10", "measured_on", ["results/image_cifar.json"], 1.0),
        E("sys_vision_cifar", "paper_bireal", "derived_from", ["docs/tutorials/04_image_cifar.md"], 0.9),
        E("sys_audio_synth", "data_audio_synth", "measured_on", ["results/audio_synth.json"], 1.0),
        E("sys_seq_encdec", "bnn_lab_system", "part_of", ["docs/36_ENCODER_DECODER_AND_NEXT.md"], 0.8),
        E("sys_energy_module", "energy_proxy_ept", "implements", ["bnn/energy/", "results/energy_bound.json"], 1.0),
        E("sys_energy_module", "result_energy_bound", "measured_on", ["results/energy_bound.json"], 1.0),
        E("sys_energy_module", "gap_rapl_windows", "blocked_by", ["docs/spikes/RAPL_ENERGY_SPIKE.md"], 0.7),
        E("sys_packed_kernels", "result_openmp_scaling", "measured_on", ["results/benchmark.json"], 0.9),
        E("sys_packed_kernels", "hw_arm_neon", "supports", ["docs/41_PORTABLE_SIMD_KERNEL.md"], 0.85),
        E("bnn_lab_system", "sys_optimise_api", "part_of", ["ROADMAP.md"], 0.9),
        E("bnn_lab_system", "sys_packed_kernels", "part_of", ["ROADMAP.md"], 1.0),
        E("bnn_lab_system", "sys_repro_gates", "part_of", ["AGENTS.md"], 1.0),
        # Decision tree edges
        E("decision_wrap_tree", "hw_nvidia_gpu", "recommends_for", ["docs/18 — FP8/AWQ path"], 0.9),
        E("decision_wrap_tree", "tool_bitnet_cpp", "recommends_for", ["docs/18 — BitNet checkpoint"], 0.9),
        E("decision_wrap_tree", "tool_gguf_llamacpp", "recommends_for", ["docs/18 — non-BitNet CPU LLM"], 0.9),
        E("decision_wrap_tree", "tool_larq", "recommends_for", ["docs/18 — edge vision retrainable"], 0.8),
        E("decision_wrap_tree", "tool_brevitas_finn", "recommends_for", ["docs/18 — FPGA"], 0.8),
        E("decision_wrap_tree", "bnn_lab_system", "recommends_for", ["docs/18 — research/teach XNOR"], 1.0),
        E("decision_wrap_tree", "paper_bitdistill", "recommends_for", ["docs/18 — convert HF LLM to 1.58"], 0.85),
        E("decision_wrap_tree", "tool_tflite_openvino_ort", "recommends_for", ["docs/18 — cannot retrain"], 0.85),
        E("decision_no_gpu_32x", "fake_binary_sign", "forbids", ["ROADMAP.md"], 1.0),
        E("decision_onnx_defer", "gap_bnnpack_v2", "related_to", ["docs/MOONSHOT_DEFERRALS.md"], 0.6),
        # Roadmap / gaps
        E("roadmap_v030", "world_class_v1", "part_of", ["ROADMAP.md"], 0.8),
        E("roadmap_v030", "sys_packed_kernels", "implements", ["docs/40_ROADMAP_E2E_SESSION.md"], 0.9),
        E("world_class_v1", "gap_pypi_trusted", "blocked_by", ["ROADMAP.md scorecard"], 0.7),
        E("world_class_v1", "gap_distill_integration", "blocked_by", ["docs/MOONSHOT_DEFERRALS.md"], 0.7),
        E("world_class_v1", "gap_bnnpack_v2", "blocked_by", ["docs/MOONSHOT_DEFERRALS.md"], 0.6),
        E("gap_wasm", "sys_packed_kernels", "alternative_to", ["docs/spikes/WASM_SIMD.md"], 0.5),
        E("gap_imagenet_sota", "data_imagenet", "blocked_by", ["ROADMAP.md non-goals"], 0.9),
        E("gap_venue_submit", "paper_b1_honest_speedup", "blocked_by", ["docs/PUBLICATION_PLAN.md"], 0.8),
        E("gap_venue_submit", "paper_b2_packed_xnor", "blocked_by", ["docs/PUBLICATION_PLAN.md"], 0.8),
        E("gap_venue_submit", "paper_b3_when_not", "blocked_by", ["docs/PUBLICATION_PLAN.md"], 0.8),
        # Results links
        E("result_kernel_speedups", "metric_s_compute", "measured_on", ["results/benchmark.json"], 1.0),
        E("result_wrap_e2e", "metric_s_e2e", "measured_on", ["results/wrap_demo.json"], 1.0),
        E("result_wrap_e2e", "compression_32x", "measured_on", ["results/wrap_demo.json"], 1.0),
        E("result_energy_bound", "result_wrap_e2e", "derived_from", ["results/energy_bound.json"], 1.0),
        E("result_mnist", "data_mnist", "measured_on", ["results/train_results.json"], 1.0),
        E("result_cifar", "accuracy_collapse", "exemplifies", ["results/image_cifar.json 10pp gap"], 0.7),
        E("result_openmp_scaling", "sys_packed_kernels", "measured_on", ["results/benchmark.json thread_scaling"], 0.9),
        # Org
        E("org_microsoft_bitnet", "paper_bitnet", "authored", ["arXiv:2310.11453"], 1.0),
        E("org_microsoft_bitnet", "tool_bitnet_cpp", "maintains", ["https://github.com/microsoft/BitNet"], 1.0),
        E("org_plumerai_larq", "tool_larq", "maintains", ["arXiv:2011.09398"], 0.9),
        # Hardware recommendations
        E("hw_cpu_x86_popcnt", "algo_xnor_gemm", "recommends_for", ["docs/14_HARDWARE_AND_ENERGY.md"], 1.0),
        E("hw_nvidia_gpu", "decision_no_gpu_32x", "requires", ["ROADMAP.md"], 1.0),
        E("hw_phone_npu", "npu_no_native_1bit", "exhibits", ["docs/20_NPU_VENDOR_CLOSURE.md"], 1.0),
        E("hw_fpga_finn", "paper_finn", "enables", ["docs/14_HARDWARE_AND_ENERGY.md"], 0.9),
        # Concept links
        E("concept_ste_vs_inference", "thesis_lock", "part_of", ["ROADMAP.md"], 1.0),
        E("concept_fp_shortcuts", "sys_vision_cifar", "implements", ["bnn/vision/"], 0.85),
        E("concept_prepack_once", "result_kernel_speedups", "requires", ["results/benchmark.json"], 1.0),
        E("concept_ternary_zero_gate", "paper_bitnet_b158", "derived_from", ["arXiv:2402.17764"], 1.0),
        # Bridges as alternatives
        E("tool_bitnet_cpp", "bnn_lab_system", "alternative_to", ["docs/23_BITNET_CPP_BRIDGE.md"], 0.75),
        E("tool_gguf_llamacpp", "bnn_lab_system", "alternative_to", ["docs/22_HF_TO_GGUF_GUIDE.md"], 0.7),
        E("tool_torchao", "algo_xnor_gemm", "alternative_to", ["docs/24_GPU_INT4_FP8_LANE.md"], 0.8),
        E("tool_vllm_trt", "fake_binary_sign", "alternative_to", ["docs/18_DECISION_TREE_AND_COMPLETE_ROADMAP.md"], 0.85),
        # Expanded cluster edges
        E("paper_binaryconnect", "paper_bnn_2016", "part_of", ["docs/02_SOTA_SURVEY.md"], 0.8),
        E("paper_binaryconnect", "compression_32x", "improves", ["docs/02 — weight-only memory"], 0.5),
        E("paper_dorefa", "paper_bnn_2016", "improves", ["docs/02_SOTA_SURVEY.md"], 0.6),
        E("paper_fbi_llm", "paper_bitnet", "alternative_to", ["arXiv:2407.07093"], 0.7),
        E("paper_fbi_llm", "gap_fbi_llm_repro", "blocked_by", ["ROADMAP non-goals"], 0.8),
        E("paper_bitnet_v2", "paper_bitnet_a48", "improves", ["arXiv:2504.18415"], 0.85),
        E("paper_bitnet_v2", "org_microsoft_bitnet", "part_of", ["arXiv:2504.18415"], 0.9),
        E("method_tmac", "tool_bitnet_cpp", "alternative_to", ["docs/02_SOTA_SURVEY.md"], 0.7),
        E("method_tmac", "algo_ternary_bitplane", "related_to", ["docs/16_ECOSYSTEM_AND_TOOLING.md"], 0.5),
        E("method_surge", "ste_mismatch", "improves", ["docs/03_FAILURE_ANALYSIS.md"], 0.75),
        E("method_surge", "algo_ste_clip", "alternative_to", ["docs/13_TRAINING_QAT_DISTILL.md"], 0.7),
        E("method_absmean_ptq", "ptq_ternary_llm_wipe", "part_of", ["docs/18_DECISION_TREE_AND_COMPLETE_ROADMAP.md"], 1.0),
        E("method_absmean_ptq", "paper_bitdistill", "contradicts", ["arXiv:2510.13998 — CPT+distill needed"], 0.95),
        E("sys_guide_e2e", "sys_optimise_api", "recommends_for", ["docs/GUIDE_E2E.md"], 1.0),
        E("sys_guide_e2e", "sys_repro_gates", "requires", ["AGENTS.md"], 1.0),
        E("sys_guide_e2e", "bnn_lab_system", "part_of", ["docs/39_GUIDE_E2E_COMPLETION.md"], 0.9),
        E("sys_wrap_policy", "sys_optimise_api", "part_of", ["docs/42_QAT_AND_LAYER_SEARCH.md"], 0.9),
        E("sys_wrap_policy", "algo_hybrid_ffn_wrap", "recommends_for", ["bnn/wrap/policy.py"], 0.8),
        E("sys_wrap_policy", "gap_layer_search_full", "blocked_by", ["docs/MOONSHOT_DEFERRALS.md"], 0.7),
        E("sys_calibrate_qat", "metric_cosine_vs_fp", "improves", ["docs/42_QAT_AND_LAYER_SEARCH.md"], 0.9),
        E("sys_calibrate_qat", "algo_ste_clip", "requires", ["bnn/wrap/qat.py"], 0.8),
        E("sys_calibrate_qat", "result_ultra_wrap", "measured_on", ["results/ultra_wrap.json"], 0.85),
        E("sys_profile_pareto", "dual_metric_culture", "implements", ["docs/FAIR_EVAL_PROTOCOL.md"], 0.9),
        E("sys_profile_pareto", "metric_energy_proxy_ratio", "measured_on", ["results/energy_bound.json"], 0.8),
        E("sys_hf_optimiser", "sys_optimise_api", "implements", ["docs/tutorials/08_HF_OPTIMISER.md"], 0.9),
        E("sys_hf_optimiser", "sys_bnnpack", "requires", ["docs/api/codec.md"], 0.7),
        E("sys_bridge_recipes", "tool_torchao", "implements", ["results/bridge_gpu_torchao.json"], 0.9),
        E("sys_bridge_recipes", "tool_bitnet_cpp", "implements", ["results/bridge_cpu_llamacpp_bitnet.json"], 0.9),
        E("sys_bridge_recipes", "decision_wrap_tree", "part_of", ["docs/18_DECISION_TREE_AND_COMPLETE_ROADMAP.md"], 0.85),
        E("result_ultra_wrap", "sys_ultra_wrap", "measured_on", ["results/ultra_wrap.json"], 1.0),
        E("result_ultra_wrap", "algo_ternary_bitplane", "measured_on", ["tests/golden_floors.json ternary 16×"], 0.9),
        E("result_robustness_fgsm", "result_mnist", "measured_on", ["results/robustness_fgsm.json"], 0.7),
        E("result_ternary_pack", "algo_ternary_bitplane", "measured_on", ["results/ternary_pack.json"], 1.0),
        E("metric_cosine_vs_fp", "result_wrap_e2e", "measured_on", ["results/wrap_demo.json cosine~0.31"], 1.0),
        E("metric_cosine_vs_fp", "dual_metric_culture", "part_of", ["docs/FAIR_EVAL_PROTOCOL.md"], 0.8),
        E("metric_energy_proxy_ratio", "energy_proxy_ept", "derived_from", ["results/energy_bound.json"], 1.0),
        E("concept_sim_vs_packed", "fake_binary_sign", "contradicts", ["docs/03_FAILURE_ANALYSIS.md"], 1.0),
        E("concept_sim_vs_packed", "algo_xnor_gemm", "requires", ["docs/05_PERFECTED_CONCEPT.md"], 1.0),
        E("concept_word_vs_wallclock", "theoretical_word_reduction_64x", "contradicts", ["results/SUMMARY.md"], 1.0),
        E("concept_word_vs_wallclock", "metric_s_compute", "requires", ["docs/06_CALCULATED_SPEEDUP_MODEL.md"], 0.9),
        E("concept_word_vs_wallclock", "paper_b1_honest_speedup", "part_of", ["docs/32_NOVEL_PAPER_CANDIDATES.md"], 1.0),
        E("hw_wasm", "gap_wasm", "part_of", ["docs/MOONSHOT_DEFERRALS.md"], 1.0),
        E("decision_int8_npu_first", "hw_phone_npu", "recommends_for", ["docs/20_NPU_VENDOR_CLOSURE.md"], 1.0),
        E("decision_int8_npu_first", "npu_no_native_1bit", "derived_from", ["docs/20_NPU_VENDOR_CLOSURE.md"], 1.0),
        E("decision_fair_eval", "sys_repro_gates", "requires", ["docs/FAIR_EVAL_PROTOCOL.md"], 1.0),
        E("decision_fair_eval", "thesis_lock", "part_of", ["AGENTS.md"], 1.0),
        E("gap_layer_search_full", "sys_wrap_policy", "blocked_by", ["docs/MOONSHOT_DEFERRALS.md"], 0.75),
        E("gap_bitnet_submodule", "tool_bitnet_cpp", "blocked_by", ["third_party/BITNET_PIN.md"], 0.8),
        E("gap_fbi_llm_repro", "paper_fbi_llm", "blocked_by", ["ROADMAP.md"], 0.8),
        E("tool_executorch", "tool_torchao", "requires", ["docs/14_HARDWARE_AND_ENERGY.md"], 0.7),
        E("tool_executorch", "hw_phone_npu", "recommends_for", ["docs/16_ECOSYSTEM_AND_TOOLING.md"], 0.6),
        E("dataset_glue_downstream", "paper_bitdistill", "measured_on", ["arXiv:2510.13998"], 0.85),
        E("paper_awq", "method_absmean_ptq", "alternative_to", ["docs/24 — AWQ for GPU INT4"], 0.75),
        E("sys_energy_module", "metric_energy_proxy_ratio", "implements", ["docs/spikes/RAPL_ENERGY_SPIKE.md"], 0.9),
        E("bnn_lab_system", "sys_guide_e2e", "part_of", ["docs/GUIDE_E2E.md"], 0.9),
        E("bnn_lab_system", "sys_bridge_recipes", "part_of", ["ROADMAP.md WC-P2"], 0.85),
        E("org_microsoft_bitnet", "paper_bitdistill", "authored", ["arXiv:2510.13998"], 0.95),
        E("org_microsoft_bitnet", "paper_bitnet_2b4t", "authored", ["arXiv:2504.12285"], 0.95),
    ]


# Alias edges that used free-form relation names → normalize a few
_RELATION_ALIASES = {
    "explains": "derived_from",
    "popularizes": "cites",
    "improved_by": "improves",
    "introduces": "derived_from",
    "motivates": "derived_from",
    "mitigates": "improves",
    "not_fully_implemented_in": "blocked_by",
    "supports": "implements",
    "forbids": "contradicts",
    "related_to": "part_of",
    "exemplifies": "measured_on",
    "authored": "part_of",
    "authored_by": "part_of",
    "maintains": "implements",
    "exhibits": "measured_on",
    "enables": "recommends_for",
}


def normalize_edges(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for e in edges:
        rel = e["relation"]
        e2 = dict(e)
        e2["relation"] = _RELATION_ALIASES.get(rel, rel)
        if rel != e2["relation"]:
            e2["relation_original"] = rel
        out.append(e2)
    return out


def to_graphml(nodes: list[dict], edges: list[dict]) -> str:
    g = ET.Element("graphml", xmlns="http://graphml.graphdrawing.org/xmlns")
    for key, kind, attr_name, attr_type in [
        ("label", "node", "label", "string"),
        ("type", "node", "type", "string"),
        ("summary", "node", "summary", "string"),
        ("confidence", "node", "confidence", "double"),
        ("status", "node", "status", "string"),
        ("sources", "node", "sources", "string"),
        ("relation", "edge", "relation", "string"),
        ("evidence", "edge", "evidence", "string"),
        ("weight", "edge", "weight", "double"),
    ]:
        ET.SubElement(g, "key", id=key, **{"for": kind, "attr.name": attr_name, "attr.type": attr_type})
    graph = ET.SubElement(g, "graph", id="bnn_kg", edgedefault="directed")
    for n in nodes:
        ne = ET.SubElement(graph, "node", id=n["id"])
        for k in ("label", "type", "summary", "confidence", "status"):
            d = ET.SubElement(ne, "data", key=k)
            d.text = str(n.get(k, ""))
        d = ET.SubElement(ne, "data", key="sources")
        d.text = " | ".join(n.get("sources", []))
    for i, e in enumerate(edges):
        ee = ET.SubElement(
            graph,
            "edge",
            id=f"e{i}",
            source=e["source"],
            target=e["target"],
        )
        d = ET.SubElement(ee, "data", key="relation")
        d.text = e["relation"]
        d = ET.SubElement(ee, "data", key="evidence")
        d.text = " | ".join(e.get("evidence", []))
        d = ET.SubElement(ee, "data", key="weight")
        d.text = str(e.get("weight", 1.0))
    # pretty print
    from xml.dom import minidom

    rough = ET.tostring(g, encoding="utf-8")
    return minidom.parseString(rough).toprettyxml(indent="  ", encoding="utf-8").decode("utf-8")


def main() -> None:
    nodes = build_nodes()
    edges = normalize_edges(build_edges())
    ids = {n["id"] for n in nodes}
    dangling = [
        e for e in edges if e["source"] not in ids or e["target"] not in ids
    ]
    if dangling:
        raise SystemExit(f"Dangling edges: {dangling[:5]}")

    # Fix accidental node id used as source string in decision_wrap_tree sources
    for n in nodes:
        n["sources"] = [s for s in n["sources"] if s != "paper_b3_when_not"]
        if n["id"] == "decision_wrap_tree":
            n["sources"].append("docs/32_NOVEL_PAPER_CANDIDATES.md")

    meta = {
        "name": "Binary Neural Networks Lab Knowledge Graph",
        "version": "1.0.0",
        "repo": "https://github.com/KanakMalpani/Binary-Neural-Networks",
        "thesis_lock": (
            "Packed CPU/edge XNOR-popcount + honest STE; never claim GPU 32× from sign(); "
            "32× compression ≠ 32× latency."
        ),
        "generated_by": "scripts/build_bnn_kg.py",
        "node_count": len(nodes),
        "edge_count": len(edges),
        "node_types": sorted({n["type"] for n in nodes}),
        "relations": sorted({e["relation"] for e in edges}),
    }
    graph = {"meta": meta, "nodes": nodes, "edges": edges}

    OUT.mkdir(parents=True, exist_ok=True)
    json_path = OUT / "bnn_kg.json"
    json_path.write_text(json.dumps(graph, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    graphml_path = OUT / "bnn_kg.graphml"
    graphml_path.write_text(to_graphml(nodes, edges), encoding="utf-8")
    print(f"Wrote {json_path} ({len(nodes)} nodes, {len(edges)} edges)")
    print(f"Wrote {graphml_path}")


if __name__ == "__main__":
    main()
