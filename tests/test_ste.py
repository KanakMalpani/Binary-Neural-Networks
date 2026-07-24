"""STE unit tests."""

from __future__ import annotations

import torch

from bnn.layers import BinaryLinear
from bnn.ste import (
    binary_sign,
    binary_sign_approx,
    binary_sign_tanh_soft,
    clip_weights_,
    get_binary_sign_fn,
    get_sign_mode,
    irnet_ede_schedule,
    set_approx_sign,
    set_sign_mode,
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
        assert get_sign_mode() == "approx"
    finally:
        set_approx_sign(False)
    assert get_binary_sign_fn() is binary_sign


def test_tanh_soft_and_ede_schedule():
    x = torch.tensor([-0.5, 0.25], requires_grad=True)
    y = binary_sign_tanh_soft(x, t=1.0, k=1.0)
    assert torch.equal(y.detach(), torch.tensor([-1.0, 1.0]))
    y.sum().backward()
    assert x.grad is not None
    # Grad should be k t (1-tanh^2(t x)) > 0 near 0
    assert float(x.grad[1]) > 0
    t0, k0 = irnet_ede_schedule(0, 100)
    t1, k1 = irnet_ede_schedule(100, 100)
    assert t1 > t0
    set_sign_mode("tanh_soft", t=2.0, k=1.0)
    try:
        assert get_sign_mode() == "tanh_soft"
        fn = get_binary_sign_fn()
        z = torch.tensor([0.1], requires_grad=True)
        fn(z).sum().backward()
        assert z.grad is not None
    finally:
        set_sign_mode("ste")


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
