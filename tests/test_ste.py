"""STE unit tests."""

from __future__ import annotations

import torch

from bnn.layers import BinaryLinear
from bnn.ste import (
    binary_sign,
    binary_sign_approx,
    clip_weights_,
    get_binary_sign_fn,
    set_approx_sign,
    ternary_weight,
)


def test_binary_sign_pm1():
    x = torch.tensor([-2.0, -0.1, 0.0, 0.5, 3.0])
    y = binary_sign(x)
    # non-positive → -1, positive → +1
    assert torch.equal(y, torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0]))


def test_approx_sign_forward_and_switch():
    x = torch.tensor([-0.5, 0.25], requires_grad=True)
    y = binary_sign_approx(x)
    assert torch.equal(y.detach(), torch.tensor([-1.0, 1.0]))
    y.sum().backward()
    assert x.grad is not None
    set_approx_sign(True)
    try:
        assert get_binary_sign_fn() is binary_sign_approx
    finally:
        set_approx_sign(False)
    assert get_binary_sign_fn() is binary_sign


def test_ternary_weight_values():
    w = torch.tensor([0.0, 0.01, 1.0, -1.0, 2.0])
    q = ternary_weight(w)
    assert set(q.unique().tolist()).issubset({-1.0, 0.0, 1.0})


def test_clip_weights_on_binary_linear():
    layer = BinaryLinear(8, 4)
    with torch.no_grad():
        layer.weight.fill_(5.0)
    clip_weights_(layer)
    assert float(layer.weight.detach().max()) <= 1.0 + 1e-5
