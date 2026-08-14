#!/usr/bin/env python3
"""Wave S2 spike: hybrid/binary QAT on the committed ultra_wrap TinyBlock shape.

AND-gate (without --force): cosine >= 0.85 AND e2e >= 1.5x vs FP.
Does not overwrite results/ultra_wrap.json. Not a new golden shape.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bnn.determinism import set_repro_seed  # noqa: E402
from bnn.wrap import (  # noqa: E402
    CalibConfig,
    DistillConfig,
    distill_binary_student,
    light_qat_recover,
    measure_agreement,
    wrap_model,
)


class TinyBlock(nn.Module):
    """Committed ultra_wrap shape — keep in sync with scripts/ultra_wrap_demo.py."""

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

AND_COSINE = 0.85
AND_E2E = 1.5


def _e2e_speedup(teacher: nn.Module, student: nn.Module, x: torch.Tensor) -> float:
    t_fp = time_fn(lambda: teacher(x), warmup=3, reps=12)
    t_w = time_fn(lambda: student(x), warmup=3, reps=12)
    return float(t_fp / t_w) if t_w else 0.0


def _wrap_hybrid(student: nn.Module) -> nn.Module:
    wrapped, _ = wrap_model(
        student,
        mode="binary_xnor",
        policy="hybrid_ffn",
        min_in_features=32,
        calib=CalibConfig(method="absmean", per_channel=True),
        inplace=True,
    )
    return wrapped


def run_recipe(name: str, *, qat_kwargs: dict | None, distill_cfg: DistillConfig | None) -> dict:
    set_repro_seed(0, deterministic=True, force_cpu=True)
    teacher = TinyBlock(512, 2048)
    x = torch.randn(64, 28 * 28)
    student = copy.deepcopy(teacher)
    qat_info: dict | None = None
    distill_info: dict | None = None
    t0 = time.perf_counter()
    if qat_kwargs is not None:
        qat_info = light_qat_recover(
            student,
            x,
            teacher=teacher,
            layer_names=["ffn_fc1", "ffn_fc2"],
            **qat_kwargs,
        )
    if distill_cfg is not None:
        batches = [x, torch.randn(64, 28 * 28), torch.randn(64, 28 * 28)]
        d = distill_binary_student(
            student, teacher, batches, cfg=distill_cfg
        )
        distill_info = d.to_dict()
    student.eval()
    with torch.no_grad():
        cos_prewrap = float(measure_agreement(teacher(x), student(x)).cosine)
    wrapped = _wrap_hybrid(student)
    with torch.no_grad():
        eff = measure_agreement(teacher(x), wrapped(x), drop_in_threshold=AND_COSINE)
    e2e = _e2e_speedup(teacher, wrapped, x)
    elapsed = time.perf_counter() - t0
    row = {
        "recipe": name,
        "cosine": eff.cosine,
        "cosine_prewrap": cos_prewrap,
        "kl_div": eff.kl_div,
        "top1_agreement": eff.top1_agreement,
        "drop_in_ok": eff.drop_in_ok,
        "e2e_speedup": e2e,
        "and_gate": bool(eff.cosine >= AND_COSINE and e2e >= AND_E2E),
        "forced": False,
        "seconds": elapsed,
        "qat": qat_info,
        "distill": distill_info,
        "replaced_expected": ["ffn_fc1", "ffn_fc2"],
        "shape": {"d_model": 512, "ff": 2048, "batch": 64},
        "thesis_note": (
            "Packed CPU XNOR; cosine and e2e are measured. "
            "Never GPU 32x from sign()."
        ),
    }
    print(
        f"{name:40s}  cos={eff.cosine:.4f}  prewrap={cos_prewrap:.4f}  "
        f"e2e={e2e:.3f}x  drop_in={eff.drop_in_ok}  AND={row['and_gate']}  "
        f"({elapsed:.1f}s)",
        flush=True,
    )
    return row


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "docs" / "spikes" / "WRAP_HYBRID_085_MEASURED.json",
    )
    p.add_argument("--quick", action="store_true", help="Fewer recipes (smoke)")
    args = p.parse_args()

    recipes: list[tuple[str, dict | None, DistillConfig | None]] = [
        ("ptq_hybrid", None, None),
        (
            "legacy_kd_nofold_200",
            {"steps": 200, "lr": 1e-3, "logit_loss": "kd", "fold_alpha": False},
            None,
        ),
        (
            "mse_fold_200",
            {"steps": 200, "lr": 1e-3, "logit_loss": "mse", "fold_alpha": True},
            None,
        ),
        (
            "cosine_fold_200",
            {"steps": 200, "lr": 1e-3, "logit_loss": "cosine", "fold_alpha": True},
            None,
        ),
        (
            "mse_fold_approx_200",
            {
                "steps": 200,
                "lr": 1e-3,
                "logit_loss": "mse",
                "fold_alpha": True,
                "sign_mode": "approx",
            },
            None,
        ),
        (
            "mse_fold_hidden_200",
            {
                "steps": 200,
                "lr": 1e-3,
                "logit_loss": "mse",
                "fold_alpha": True,
                "hidden_mse": 1.0,
            },
            None,
        ),
        (
            "mse_fold_wonly_200",
            {
                "steps": 200,
                "lr": 1e-3,
                "logit_loss": "mse",
                "fold_alpha": True,
                "binarize_activations": False,
            },
            None,
        ),
    ]
    if not args.quick:
        recipes.extend(
            [
                (
                    "mse_fold_500",
                    {"steps": 500, "lr": 1e-3, "logit_loss": "mse", "fold_alpha": True},
                    None,
                ),
                (
                    "mse_fold_approx_hidden_400",
                    {
                        "steps": 400,
                        "lr": 5e-4,
                        "logit_loss": "mse",
                        "fold_alpha": True,
                        "hidden_mse": 1.0,
                        "sign_mode": "approx",
                    },
                    None,
                ),
                (
                    "distill_mse_fold_200",
                    None,
                    DistillConfig(
                        steps=200,
                        lr=1e-3,
                        logit_loss="mse",
                        fold_alpha=True,
                        layer_names=["ffn_fc1", "ffn_fc2"],
                    ),
                ),
            ]
        )

    rows = []
    for name, qat_kwargs, distill_cfg in recipes:
        rows.append(run_recipe(name, qat_kwargs=qat_kwargs, distill_cfg=distill_cfg))

    any_and = any(r["and_gate"] for r in rows)
    payload = {
        "schema": "wrap_hybrid_085_spike_v1",
        "and_gate": {"cosine_min": AND_COSINE, "e2e_min": AND_E2E, "force": False},
        "shape": "ultra_wrap TinyBlock d=512 ff=2048 batch=64 (committed)",
        "passed": any_and,
        "rows": rows,
        "thesis_lock": "CPU packed XNOR; no GPU 32x from sign(); no new golden shapes.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}  AND-gate passed={any_and}", flush=True)
    return 0 if any_and else 2


if __name__ == "__main__":
    raise SystemExit(main())
