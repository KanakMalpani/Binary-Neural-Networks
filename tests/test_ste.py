"""STE unit tests."""

from __future__ import annotations

import torch

from bnn.layers import BinaryLinear
from bnn.ste import binary_sign, clip_weights_, ternary_weight


def test_binary_sign_pm1():
    x = torch.tensor([-2.0, -0.1, 0.0, 0.5, 3.0])
    y = binary_sign(x)
    # non-positive → -1, positive → +1
    assert torch.equal(y, torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0]))


def test_ternary_weight_values():
    w = torch.tensor([0.0, 0.01, 1.0, -1.0, 2.0])
    q = ternary_weight(w)
    assert set(q.unique().tolist()).issubset({-1.0, 0.0, 1.0})


def test_clip_weights_on_binary_linear():
    layer = BinaryLinear(8, 4)
    with torch.no_grad():
        layer.weight.fill_(5.0)
    clip_weights_(layer)
    assert float(layer.weight.max()) <= 1.0 + 1e-5
