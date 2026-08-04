#!/usr/bin/env python3
"""W3.T08 distill demo: measured cosine uplift vs cold PTQ wrap.

Honest dual-metric demo — cosine/agreement are measured; compression from
``wrap_model`` is theoretical pack ratio. Never claim GPU 32× from ``sign()``.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bnn.determinism import set_repro_seed  # noqa: E402
from bnn.wrap import (  # noqa: E402
    DistillConfig,
    attach_effectiveness,
    distill_binary_student,
    measure_agreement,
    wrap_model,
)


class TinyBlock(nn.Module):
    def __init__(self, d: int = 128, ff: int = 512, n_classes: int = 10):
        super().__init__()
        self.embed = nn.Linear(64, d)
        self.attn_qkv = nn.Linear(d, d)
        self.ffn_fc1 = nn.Linear(d, ff)
        self.ffn_fc2 = nn.Linear(ff, d)
        self.lm_head = nn.Linear(d, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.embed(x))
        h = h + F.relu(self.attn_qkv(h))
        h = h + self.ffn_fc2(F.relu(self.ffn_fc1(h)))
        return self.lm_head(h)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=80)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "distill_wrap_demo.json",
    )
    args = p.parse_args()

    set_repro_seed(args.seed, deterministic=True, force_cpu=True)
    teacher = TinyBlock()
    x = torch.randn(args.batch, 64)
    batches = [x, torch.randn(args.batch, 64), torch.randn(args.batch, 64)]

    cold = copy.deepcopy(teacher)
    cold_wrapped, cold_report = wrap_model(
        cold, policy="hybrid_ffn", min_in_features=32, inplace=True
    )
    with torch.no_grad():
        eff_ptq = measure_agreement(teacher(x), cold_wrapped(x))
    attach_effectiveness(cold_report, eff_ptq)

    warm = copy.deepcopy(teacher)
    with torch.no_grad():
        warm.ffn_fc1.weight.mul_(0.4)
        warm.ffn_fc2.weight.mul_(0.4)
    distill = distill_binary_student(
        warm,
        teacher,
        batches,
        cfg=DistillConfig(steps=args.steps, lr=5e-3, temperature=2.0),
    )
    warm_wrapped, warm_report = wrap_model(
        warm, policy="hybrid_ffn", min_in_features=32, inplace=True
    )
    with torch.no_grad():
        eff_qat = measure_agreement(teacher(x), warm_wrapped(x))
    attach_effectiveness(warm_report, eff_qat)

    payload = {
        "protocol": "distill_binary_student → wrap_model vs cold PTQ wrap",
        "cold_ptq": {
            "cosine": eff_ptq.cosine,
            "drop_in_ok": eff_ptq.drop_in_ok,
            "theoretical_compression": cold_report.compression,
            "policy_reason": cold_report.policy_reason,
            "effectiveness": cold_report.effectiveness,
        },
        "distill_then_wrap": {
            "cosine": eff_qat.cosine,
            "drop_in_ok": eff_qat.drop_in_ok,
            "theoretical_compression": warm_report.compression,
            "distill": distill.to_dict(),
            "effectiveness": warm_report.effectiveness,
        },
        "cosine_uplift_wrap": float(eff_qat.cosine - eff_ptq.cosine),
        "thesis_note": (
            "Cosine is measured agreement; compression_* is theoretical pack "
            "ratio. Never claim GPU 32× from sign()/STE."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
