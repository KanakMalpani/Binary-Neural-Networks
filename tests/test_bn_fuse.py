"""BN fuse helper smoke (eval path only)."""

from __future__ import annotations

import torch

from bnn.layers import BiRealBlock, fuse_binary_conv_bn_


def test_fuse_bn_preserves_eval_output():
    torch.manual_seed(0)
    block = BiRealBlock(8)
    block.train()
    x = torch.randn(4, 8, 7, 7)
    # Populate running stats
    for _ in range(5):
        _ = block(torch.randn(4, 8, 7, 7))
    block.eval()
    y0 = block(x).detach().clone()
    fuse_binary_conv_bn_(block)
    y1 = block(x)
    assert torch.allclose(y0, y1, atol=1e-5, rtol=1e-5)
    assert block._bn_fused is True
