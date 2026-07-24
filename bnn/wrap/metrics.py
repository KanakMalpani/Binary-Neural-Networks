"""Effectiveness metrics vs FP teacher (cosine / KL / top-1 agreement)."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class EffectivenessReport:
    cosine: float
    kl_div: float | None
    top1_agreement: float | None
    n_samples: int
    drop_in_threshold: float
    drop_in_ok: bool
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def measure_agreement(
    teacher_logits: Tensor,
    student_logits: Tensor,
    *,
    drop_in_threshold: float = 0.85,
) -> EffectivenessReport:
    """Compare wrapped vs FP teacher on a batch of logits (or embeddings)."""
    t = teacher_logits.detach().float().reshape(teacher_logits.shape[0], -1).cpu()
    s = student_logits.detach().float().reshape(student_logits.shape[0], -1).cpu()
    if t.shape != s.shape:
        raise ValueError(f"shape mismatch teacher {t.shape} vs student {s.shape}")

    # Mean cosine over batch rows
    cos = float(F.cosine_similarity(t, s, dim=1).mean().item())

    kl: float | None = None
    top1: float | None = None
    if t.shape[1] >= 2:
        # Treat last dim as class logits when reasonable
        log_p = F.log_softmax(s, dim=1)
        q = F.softmax(t, dim=1)
        kl = float(F.kl_div(log_p, q, reduction="batchmean").item())
        top1 = float((t.argmax(dim=1) == s.argmax(dim=1)).float().mean().item())

    ok = cos >= drop_in_threshold
    notes = (
        "Meets drop-in cosine threshold"
        if ok
        else (
            f"Below drop-in threshold ({cos:.3f} < {drop_in_threshold}); "
            "refuse to claim drop-in unless --force"
        )
    )
    return EffectivenessReport(
        cosine=cos,
        kl_div=kl,
        top1_agreement=top1,
        n_samples=int(t.shape[0]),
        drop_in_threshold=drop_in_threshold,
        drop_in_ok=ok,
        notes=notes,
    )


def drop_in_ok(report: EffectivenessReport, *, force: bool = False) -> bool:
    return bool(report.drop_in_ok or force)
