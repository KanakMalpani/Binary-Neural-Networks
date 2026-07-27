"""CLI / script: STE vs ApproxSign vs tanh-soft on a tiny binary MLP."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bnn.layers import BinaryLinear  # noqa: E402
from bnn.ste import (  # noqa: E402
    approx_sign_grad_numpy,
    clip_weights_,
    gradient_cosine,
    set_sign_mode,
    ste_grad_numpy,
    tanh_soft_grad_numpy,
)


def _grad_field_compare(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 0.8, size=10_000)
    ste = ste_grad_numpy(x)
    approx = approx_sign_grad_numpy(x)
    soft = tanh_soft_grad_numpy(x, t=1.0, k=1.0)
    # Proxy "true" soft sign derivative: sech^2 of a sharp tanh (teacher)
    teacher = tanh_soft_grad_numpy(x, t=5.0, k=1.0)
    return {
        "cosine_ste_vs_approx": gradient_cosine(ste, approx),
        "cosine_ste_vs_tanh_soft": gradient_cosine(ste, soft),
        "cosine_approx_vs_tanh_soft": gradient_cosine(approx, soft),
        "cosine_ste_vs_sharp_teacher": gradient_cosine(ste, teacher),
        "cosine_approx_vs_sharp_teacher": gradient_cosine(approx, teacher),
        "cosine_tanh_soft_vs_sharp_teacher": gradient_cosine(soft, teacher),
    }


class TinyBinMLP(nn.Module):
    def __init__(self, d_in: int = 32, d_h: int = 64, n_out: int = 4):
        super().__init__()
        self.fc1 = BinaryLinear(d_in, d_h, bias=False)
        self.fc2 = BinaryLinear(d_h, n_out, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.fc1(x))


def _train_curve(mode: str, steps: int = 80, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)  # noqa: NPY002 — legacy global RNG seeded intentionally
    set_sign_mode("ste" if mode == "ste" else mode if mode != "approx_sign" else "approx")
    if mode == "approx_sign":
        set_sign_mode("approx")
    elif mode == "tanh_soft":
        set_sign_mode("tanh_soft", t=1.5, k=1.0)
    else:
        set_sign_mode("ste")

    model = TinyBinMLP()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    # Synthetic linearly separable-ish data in ±1 space
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(256, 32, generator=g)
    y = (x[:, :4].sum(dim=1) > 0).long() % 4
    losses: list[float] = []
    for _ in range(steps):
        opt.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        opt.step()
        clip_weights_(model)
        losses.append(float(loss.detach()))
    with torch.no_grad():
        pred = model(x).argmax(dim=1)
        acc = float((pred == y).float().mean())
    return {
        "mode": mode,
        "final_loss": losses[-1],
        "final_acc": acc,
        "loss_curve": losses[:: max(1, steps // 10)],
    }


def main() -> int:
    out = {
        "gradient_cosine": _grad_field_compare(),
        "train": {
            "ste": _train_curve("ste"),
            "approx_sign": _train_curve("approx_sign"),
            "tanh_soft": _train_curve("tanh_soft"),
        },
        "notes": (
            "Tiny synthetic MLP — not a SOTA claim. ApproxSign / tanh-soft "
            "often align better with a sharp soft-sign teacher than clipped STE."
        ),
    }
    path = ROOT / "results" / "math_ste_compare.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"Wrote {path}")
    # Reset mode for other tests
    set_sign_mode("ste")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
