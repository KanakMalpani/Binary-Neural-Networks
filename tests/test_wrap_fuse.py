"""W3.T09 — BN fuse on the wrap / optimiser prep path."""

from __future__ import annotations

import torch
import torch.nn as nn

from bnn.layers import BiRealBlock
from bnn.wrap import fuse_bn_for_wrap_, fuse_linear_bn1d_, wrap_model


def test_fuse_linear_bn1d_preserves_eval_output():
    torch.manual_seed(0)
    lin = nn.Linear(16, 8)
    bn = nn.BatchNorm1d(8)
    x = torch.randn(32, 16)
    for _ in range(8):
        bn.train()
        _ = bn(lin(torch.randn(32, 16)))
    bn.eval()
    lin.eval()
    y0 = bn(lin(x)).detach().clone()
    fuse_linear_bn1d_(lin, bn)
    y1 = bn(lin(x))
    assert torch.allclose(y0, y1, atol=1e-5, rtol=1e-5)


def test_fuse_bn_for_wrap_linear_pairs():
    torch.manual_seed(1)

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.ffn_fc1 = nn.Linear(32, 64)
            self.bn1 = nn.BatchNorm1d(64)
            self.ffn_fc2 = nn.Linear(64, 32)
            self.bn2 = nn.BatchNorm1d(32)

        def forward(self, x):
            h = self.bn1(self.ffn_fc1(x))
            return self.bn2(self.ffn_fc2(torch.relu(h)))

    m = MLP()
    for _ in range(5):
        m.train()
        _ = m(torch.randn(16, 32))
    m.eval()
    x = torch.randn(8, 32)
    y0 = m(x).detach().clone()
    report = fuse_bn_for_wrap_(m, bireal=False, linear_bn=True)
    assert "ffn_fc1" in report.linear_bn_pairs
    assert "ffn_fc2" in report.linear_bn_pairs
    y1 = m(x)
    assert torch.allclose(y0, y1, atol=1e-5, rtol=1e-5)


def test_wrap_model_fuse_bn_flag_records_fuse():
    torch.manual_seed(2)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.ffn_fc1 = nn.Linear(64, 128)
            self.bn1 = nn.BatchNorm1d(128)
            self.ffn_fc2 = nn.Linear(128, 64)
            self.lm_head = nn.Linear(64, 4)

        def forward(self, x):
            h = torch.relu(self.bn1(self.ffn_fc1(x)))
            return self.lm_head(self.ffn_fc2(h))

    m = Net()
    for _ in range(4):
        m.train()
        _ = m(torch.randn(8, 64))
    m.eval()
    _, report = wrap_model(
        m, policy="hybrid_ffn", min_in_features=32, fuse_bn=True
    )
    assert report.fuse is not None
    assert "ffn_fc1" in report.fuse["linear_bn_pairs"]
    assert report.replaced


def test_fuse_bn_bireal_count():
    torch.manual_seed(0)
    block = BiRealBlock(8)
    for _ in range(5):
        block.train()
        _ = block(torch.randn(4, 8, 7, 7))
    block.eval()
    root = nn.Sequential(block)
    report = fuse_bn_for_wrap_(root, bireal=True, linear_bn=False)
    assert report.bireal_blocks == 1
    assert block._bn_fused is True
