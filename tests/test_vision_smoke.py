"""Vision smoke: forward + 1-epoch tiny subset (uses local CIFAR if present)."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from bnn.ste import set_approx_sign
from bnn.vision.models import FP32CIFARCNN, BinaryCIFARCNN, TinyBinaryViT, build_vision_model
from bnn.wrapper import wrap_conv_modules

ROOT = Path(__file__).resolve().parents[1]


def test_vision_forward_shapes():
    set_approx_sign(False)
    x = torch.randn(2, 3, 32, 32)
    for m in (FP32CIFARCNN(32), BinaryCIFARCNN(32), TinyBinaryViT(dim=32, depth=1)):
        y = m(x)
        assert y.shape == (2, 10)


def test_vision_approx_sign_forward():
    set_approx_sign(True)
    try:
        m = BinaryCIFARCNN(16)
        y = m(torch.randn(1, 3, 32, 32))
        assert y.shape == (1, 10)
    finally:
        set_approx_sign(False)


def test_build_vision_model():
    assert isinstance(build_vision_model("binary_cifar", channels=16), BinaryCIFARCNN)
    assert isinstance(build_vision_model("tiny_vit_binary", dim=32, depth=1), TinyBinaryViT)


def test_wrap_binary_conv_compression():
    m = BinaryCIFARCNN(16)
    _, report = wrap_conv_modules(m, skip_name_substr=("stem", "head", "skip"), min_weight_elems=64)
    assert report.replaced
    # Small kernels pad to 64-bit words → compression < 32×; still a clear size win
    assert report.compression >= 15.0
    y = m(torch.randn(1, 3, 32, 32))
    assert y.shape == (1, 10)


def test_vision_one_epoch_tiny_subset():
    """1-epoch smoke on tiny CIFAR subset (offline — data already in repo)."""
    from bnn.cifar import get_cifar10_loaders
    from bnn.ste import clip_weights_

    data_dir = ROOT / "data"
    train_loader, test_loader = get_cifar10_loaders(
        data_dir, batch_size=32, train_subset=128, seed=0
    )
    model = BinaryCIFARCNN(16)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    x, y = next(iter(train_loader))
    opt.zero_grad(set_to_none=True)
    loss = loss_fn(model(x), y)
    loss.backward()
    opt.step()
    clip_weights_(model)
    model.eval()
    with torch.no_grad():
        logits = model(next(iter(test_loader))[0][:8])
    assert logits.shape[1] == 10
    assert torch.isfinite(logits).all()
