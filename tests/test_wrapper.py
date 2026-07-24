"""Wrapper smoke tests."""

from __future__ import annotations

import torch
import torch.nn as nn

from bnn.wrapper import wrap_linear_modules, wrap_model


def _mlp():
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )


def test_wrap_compression():
    m = _mlp()
    # Named modules via Sequential indices — skip first/last by min features + manual
    _, report = wrap_linear_modules(
        m,
        mode="binary_xnor",
        skip_name_substr=(),
        min_in_features=100,
    )
    assert report.replaced
    assert report.compression >= 30.0


def test_wrap_model_hybrid_policy():
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Linear(64, 64)
            self.attn_qkv = nn.Linear(64, 64)
            self.ffn_fc1 = nn.Linear(64, 256)
            self.ffn_fc2 = nn.Linear(256, 64)
            self.lm_head = nn.Linear(64, 10)

        def forward(self, x):
            h = self.embed(x) + self.attn_qkv(x)
            return self.lm_head(self.ffn_fc2(self.ffn_fc1(h)))

    m = Tiny()
    _, report = wrap_model(m, policy="hybrid_ffn", min_in_features=32)
    assert "ffn_fc1" in report.replaced
    assert "ffn_fc2" in report.replaced
    assert "embed" not in report.replaced
    assert "attn_qkv" not in report.replaced
    x = torch.randn(2, 64)
    y = m(x)
    assert y.shape == (2, 10)
