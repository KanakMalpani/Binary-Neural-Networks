"""Model smoke + BN momentum gate."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from bnn.models import build_model


@pytest.mark.parametrize(
    "name",
    ["fp32_mlp", "binary_mlp", "ternary_mlp", "fp32_cnn", "binary_cnn"],
)
def test_build_forward(name):
    m = build_model(name, hidden=64)
    x = torch.randn(4, 1, 28, 28)
    y = m(x)
    assert y.shape == (4, 10)
    assert torch.isfinite(y).all()


def test_bn_momentum_0_9():
    m = build_model("binary_cnn", hidden=32)
    moms = [
        mod.momentum
        for mod in m.modules()
        if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d))
    ]
    assert moms, "expected BatchNorm"
    assert all(abs(m_ - 0.9) < 1e-6 for m_ in moms)
