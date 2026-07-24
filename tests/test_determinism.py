"""Determinism helper smoke."""

from __future__ import annotations

import torch

from bnn.determinism import set_repro_seed


def test_set_repro_seed_returns_status():
    status = set_repro_seed(123, deterministic=True, force_cpu=True)
    assert status["seed"] == 123
    assert status["force_cpu"] is True
    a = torch.randn(4)
    set_repro_seed(123, deterministic=True, force_cpu=True)
    b = torch.randn(4)
    assert torch.allclose(a, b)
