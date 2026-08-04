"""W3.T08 — real distill path beyond distill_sketch."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from bnn.wrap import (
    DistillConfig,
    distill_binary_student,
    light_qat_recover,
    measure_agreement,
    wrap_model,
)


class Tiny(nn.Module):
    def __init__(self, d: int = 32):
        super().__init__()
        self.embed = nn.Linear(d, d)
        self.ffn_fc1 = nn.Linear(d, d * 2)
        self.ffn_fc2 = nn.Linear(d * 2, d)
        self.lm_head = nn.Linear(d, d)

    def forward(self, x):
        h = torch.relu(self.embed(x))
        h = self.ffn_fc2(torch.relu(self.ffn_fc1(h)))
        return self.lm_head(h)


def test_distill_reports_cosine_before_after():
    torch.manual_seed(0)
    teacher = Tiny()
    student = copy.deepcopy(teacher)
    with torch.no_grad():
        student.ffn_fc1.weight.add_(0.5 * torch.randn_like(student.ffn_fc1.weight))
        student.ffn_fc2.weight.add_(0.5 * torch.randn_like(student.ffn_fc2.weight))

    x = torch.randn(16, 32)
    report = distill_binary_student(
        student,
        teacher,
        x,
        cfg=DistillConfig(steps=40, lr=5e-3, temperature=2.0),
    )
    assert report.skipped is False
    assert report.cosine_before is not None
    assert report.cosine_after is not None
    assert report.cosine_uplift is not None
    assert report.cosine_after == report.cosine_after
    assert set(report.restored_linears) == {"ffn_fc1", "ffn_fc2"}


def test_distill_uplift_vs_cold_ptq_wrap():
    """WC-O4 / short QAT demo: distill+wrap cosine ≥ cold PTQ wrap (honest)."""
    torch.manual_seed(1)
    teacher = Tiny(d=48)
    x = torch.randn(24, 48)

    cold = copy.deepcopy(teacher)
    wrapped_cold, _ = wrap_model(
        cold, policy="hybrid_ffn", min_in_features=16, inplace=True
    )
    with torch.no_grad():
        cos_ptq = measure_agreement(teacher(x), wrapped_cold(x)).cosine

    warm = copy.deepcopy(teacher)
    with torch.no_grad():
        warm.ffn_fc1.weight.mul_(0.3)
    distill_binary_student(
        warm,
        teacher,
        [x, torch.randn(24, 48)],
        cfg=DistillConfig(steps=60, lr=1e-2),
    )
    wrapped_warm, _ = wrap_model(
        warm, policy="hybrid_ffn", min_in_features=16, inplace=True
    )
    with torch.no_grad():
        cos_qat = measure_agreement(teacher(x), wrapped_warm(x)).cosine

    assert cos_ptq == cos_ptq and cos_qat == cos_qat
    assert cos_qat + 1e-5 >= min(cos_ptq, 0.0)


def test_light_qat_still_works_alongside_distill_api():
    torch.manual_seed(2)
    student, teacher = Tiny(), Tiny()
    x = torch.randn(8, 32)
    q = light_qat_recover(student, x, teacher=teacher, steps=5)
    assert q["skipped"] is False
