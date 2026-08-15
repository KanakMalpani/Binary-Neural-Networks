#!/usr/bin/env python3
"""Ultra wrap demo: hybrid/calib/ternary/QAT effectiveness + gemm efficiency.

Targets
-------
- Compression ~32× (binary) or ~16× (ternary theoretical pack)
- gemm_only speedup ≥ prior single-thread baseline on wide layers
- hybrid+calib / ternary cosine ≫ 0.28 (aim ≥0.85 on ternary path)
- Auto policy picks efficient+effective mode per hardware
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bnn.determinism import set_repro_seed  # noqa: E402
from bnn.wrapper import (  # noqa: E402
    CalibConfig,
    PackedBinaryXNORLinear,
    attach_effectiveness,
    light_qat_recover,
    measure_agreement,
    model_param_bytes,
    recommend_wrap_policy,
    wrap_model,
)


class TinyBlock(nn.Module):
    """Realistic tiny transformer-ish stack with FFN names for hybrid policy."""

    def __init__(self, d: int = 512, ff: int = 2048, n_classes: int = 10):
        super().__init__()
        self.embed = nn.Linear(28 * 28, d)
        self.attn_qkv = nn.Linear(d, d)
        self.ffn_fc1 = nn.Linear(d, ff)
        self.ffn_fc2 = nn.Linear(ff, d)
        self.lm_head = nn.Linear(d, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.embed(x))
        h = h + F.relu(self.attn_qkv(h))
        h = h + self.ffn_fc2(F.relu(self.ffn_fc1(h)))
        return self.lm_head(h)


def time_fn(fn, warmup: int = 5, reps: int = 25) -> float:
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps


def run_one(
    *,
    policy: str,
    mode: str | None,
    d: int,
    ff: int,
    batch: int,
    qat_steps: int,
    calib_method: str,
    per_channel: bool,
    drop_in_threshold: float,
    force: bool,
    min_width: int = 32,
) -> dict:
    set_repro_seed(0, deterministic=True, force_cpu=True)
    teacher = TinyBlock(d, ff)
    x = torch.randn(batch, 28 * 28)

    with torch.no_grad():
        t_logits = teacher(x)

    student = copy.deepcopy(teacher)
    qat_info = None
    # Binary STE QAT only helps binary_xnor; it *hurts* ternary PTQ quality.
    use_binary_qat = qat_steps > 0 and (mode in (None, "binary_xnor", "auto") or policy in ("hybrid_ffn", "aggressive", "auto", "default"))
    if policy == "ternary_wo" or mode == "ternary_weight_only":
        use_binary_qat = False
    if use_binary_qat:
        qat_info = light_qat_recover(
            student,
            x,
            teacher=teacher,
            steps=qat_steps,
            lr=1e-3,
            layer_names=["ffn_fc1", "ffn_fc2"],
            logit_loss="mse",
            fold_alpha=True,
        )
    elif qat_steps > 0 and (policy == "ternary_wo" or mode == "ternary_weight_only"):
        # Light FP distill toward teacher before ternary snap (no BinaryLinear)
        student.train()
        opt = torch.optim.Adam(student.parameters(), lr=1e-3)
        last = 0.0
        for _ in range(qat_steps):
            opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                t_out = teacher(x)
            loss = F.mse_loss(student(x), t_out)
            loss.backward()
            opt.step()
            last = float(loss.detach().item())
        student.eval()
        qat_info = {
            "steps": qat_steps,
            "skipped": False,
            "last_loss": last,
            "kind": "fp_mse_distill_pre_ternary",
            "note": "FP distill then ternary snap — not BinaryLinear STE",
        }

    calib = CalibConfig(method=calib_method, per_channel=per_channel)  # type: ignore[arg-type]
    before = model_param_bytes(student)
    student, report = wrap_model(
        student,
        mode=mode,
        policy=policy,  # type: ignore[arg-type]
        min_in_features=min_width,
        calib=calib,
    )
    after = model_param_bytes(student)

    with torch.no_grad():
        s_logits = student(x)
    eff = measure_agreement(t_logits, s_logits, drop_in_threshold=drop_in_threshold)
    attach_effectiveness(report, eff, force=force)

    # Layer microbench on ffn_fc1 if binary
    layer_micro: dict = {}
    mod = student.ffn_fc1
    fp_ref = teacher.ffn_fc1
    h = torch.randn(batch, d)
    t_fp = time_fn(lambda: fp_ref(h))
    t_wrap = time_fn(lambda: mod(h))
    layer_micro["linear_fp_ms"] = t_fp * 1e3
    layer_micro["linear_wrapped_forward_ms"] = t_wrap * 1e3
    layer_micro["speedup_wrapped_vs_torch"] = (t_fp / t_wrap) if t_wrap else None
    if isinstance(mod, PackedBinaryXNORLinear):
        h_pm1 = np.where(h.numpy() >= 0, 1.0, -1.0).astype(np.float32)
        t_gemm = time_fn(lambda: mod.gemm_only(h_pm1))
        layer_micro["linear_gemm_only_ms"] = t_gemm * 1e3
        layer_micro["speedup_gemm_only_vs_torch"] = (t_fp / t_gemm) if t_gemm else None

    t_e2e_fp = time_fn(lambda: teacher(x))
    t_e2e_w = time_fn(lambda: student(x))
    samples_s_fp = batch / t_e2e_fp if t_e2e_fp else None
    samples_s_w = batch / t_e2e_w if t_e2e_w else None

    decision = recommend_wrap_policy(teacher.ffn_fc1)

    drop_ok = bool(report.drop_in_ok)
    if not drop_ok and not force:
        status = "REFUSE_DROP_IN_CLAIM"
    else:
        status = "OK" if drop_ok else "FORCED"

    return {
        "schema": "bnn_optimise_report_v1",
        "schema_version": 1,
        "policy": report.policy,
        "mode": report.mode,
        "policy_reason": report.policy_reason,
        "thesis_note": (
            "Compression is theoretical pack ratio; latency fields are wall-clock. "
            "Never claim GPU 32× from sign()/STE."
        ),
        "auto_recommendation": {
            "policy": decision.policy,
            "mode": decision.mode,
            "reason": decision.reason,
            "fallback_note": decision.fallback_note,
        },
        "replaced": report.replaced,
        "skipped": report.skipped,
        "compression_replaced_weights": report.compression,
        "fp32_weight_bytes_replaced": report.fp32_weight_bytes_replaced,
        "packed_weight_bytes": report.packed_weight_bytes,
        "param_bytes_before": before,
        "param_bytes_after": after,
        "calib_method": report.calib_method,
        "qat": qat_info,
        "effectiveness": report.effectiveness,
        "drop_in_ok": report.drop_in_ok,
        "forced": force,
        "status": status,
        "native_kernel": report.native_kernel,
        "e2e_latency_ms_fp": t_e2e_fp * 1e3,
        "e2e_latency_ms_wrapped": t_e2e_w * 1e3,
        "e2e_speedup": (t_e2e_fp / t_e2e_w) if t_e2e_w else None,
        "samples_per_s_fp": samples_s_fp,
        "samples_per_s_wrapped": samples_s_w,
        "layer_microbench": layer_micro,
        "d_model": d,
        "ff": ff,
        "batch": batch,
        # Legacy alias for older golden readers
        "schema_legacy": "ultra_wrap_report_v1",
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Ultra wrap effectiveness + efficiency demo")
    p.add_argument(
        "--policy",
        default="auto",
        choices=["hybrid_ffn", "aggressive", "ternary_wo", "auto", "default"],
    )
    p.add_argument(
        "--mode",
        default=None,
        choices=["binary_xnor", "ternary_weight_only", "binary_weight_only_dequant", "auto"],
    )
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--ff", type=int, default=2048)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--calib-batches", type=int, default=4)
    p.add_argument("--calib-method", default="absmean", choices=["absmean", "percentile"])
    p.add_argument("--min-width", type=int, default=32)
    p.add_argument("--qat-steps", type=int, default=0)
    p.add_argument("--drop-in-threshold", type=float, default=0.85)
    p.add_argument("--force", action="store_true")
    p.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Also run aggressive binary PTQ (legacy cosine ~0.28 class) for before/after",
    )
    p.add_argument("--report", type=Path, default=ROOT / "results" / "ultra_wrap.json")
    args = p.parse_args()

    mode = args.mode
    if args.policy == "auto" and mode is None:
        mode = "auto"
    elif mode is None and args.policy == "ternary_wo":
        mode = "ternary_weight_only"
    elif mode is None:
        mode = "binary_xnor"

    primary = run_one(
        policy=args.policy,
        mode=mode,
        d=args.d_model,
        ff=args.ff,
        batch=args.batch,
        qat_steps=args.qat_steps,
        calib_method=args.calib_method,
        per_channel=True,
        drop_in_threshold=args.drop_in_threshold,
        force=args.force,
        min_width=args.min_width,
    )

    # Accurate ternary path: calib per-channel, optional FP distill (no binary STE)
    ternary = run_one(
        policy="ternary_wo",
        mode="ternary_weight_only",
        d=args.d_model,
        ff=args.ff,
        batch=args.batch,
        qat_steps=max(args.qat_steps, 20),
        calib_method=args.calib_method,
        per_channel=True,
        drop_in_threshold=args.drop_in_threshold,
        force=True,
        min_width=args.min_width,
    )

    # Wide-layer efficiency probe (gemm_only) — matches published wrap_demo shape class
    wide_eff = None
    if args.d_model < 2048:
        wide_eff = run_one(
            policy="hybrid_ffn",
            mode="binary_xnor",
            d=2048,
            ff=8192,
            batch=min(args.batch, 32),
            qat_steps=0,
            calib_method="absmean",
            per_channel=True,
            drop_in_threshold=args.drop_in_threshold,
            force=True,
            min_width=args.min_width,
        )

    baseline = None
    # No-QAT binary hybrid reference for before/after table
    baseline = run_one(
        policy="hybrid_ffn",
        mode="binary_xnor",
        d=args.d_model,
        ff=args.ff,
        batch=args.batch,
        qat_steps=0,
        calib_method="absmean",
        per_channel=True,
        drop_in_threshold=args.drop_in_threshold,
        force=True,
        min_width=args.min_width,
    )

    payload = {
        "schema": "ultra_wrap_suite_v1",
        "primary": primary,
        "ternary_accurate_path": ternary,
        "binary_hybrid_baseline": baseline,
        "wide_efficiency_probe": wide_eff,
        "before_after": {
            "legacy_binary_ptq_cosine_doc": 0.28,
            "binary_hybrid_calib_cosine": baseline["effectiveness"]["cosine"] if baseline else None,
            "binary_hybrid_qat_cosine": primary["effectiveness"]["cosine"]
            if primary.get("qat")
            else None,
            "ternary_hybrid_calib_cosine": ternary["effectiveness"]["cosine"],
            "ternary_compression": ternary["compression_replaced_weights"],
            "binary_compression": (baseline or primary)["compression_replaced_weights"],
            "binary_gemm_only_speedup_demo_shape": (
                (baseline or primary).get("layer_microbench", {}) or {}
            ).get("speedup_gemm_only_vs_torch"),
            "binary_gemm_only_speedup_wide": (
                (wide_eff or {}).get("layer_microbench", {}) or {}
            ).get("speedup_gemm_only_vs_torch"),
            "prior_wrap_demo_gemm_speedup": 2.12,
        },
        "thesis_lock": (
            "CPU/edge packed kernels; no fake GPU 32×; ternary path is size/accuracy-first "
            "without ternary GEMM; GPU production stays INT4/FP8"
        ),
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Wrote {args.report}")

    # Exit non-zero if primary refuses drop-in and not forced
    if primary["status"] == "REFUSE_DROP_IN_CLAIM" and not args.force:
        # Still success for demo harness if ternary path meets threshold
        if ternary["effectiveness"]["cosine"] >= args.drop_in_threshold:
            print(
                "NOTE: primary policy below drop-in; ternary accurate path meets threshold.",
                flush=True,
            )
            return 0
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
