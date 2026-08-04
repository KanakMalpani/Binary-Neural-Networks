"""Vision smoke: forward + 1-epoch tiny subset (synthetic — no network)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from bnn.ste import set_approx_sign
from bnn.vision.models import (
    FP32CIFARCNN,
    BinaryCIFARCNN,
    ResNetBiReal,
    TinyBinaryViT,
    build_vision_model,
)
from bnn.wrapper import wrap_conv_modules


def test_vision_forward_shapes():
    set_approx_sign(False)
    x = torch.randn(2, 3, 32, 32)
    for m in (
        FP32CIFARCNN(32),
        BinaryCIFARCNN(32),
        TinyBinaryViT(dim=32, depth=1),
        build_vision_model("resnet_bireal_cifar", width=8),
    ):
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
    assert isinstance(build_vision_model("resnet_bireal_cifar", width=8), ResNetBiReal)


def test_wrap_binary_conv_compression():
    m = BinaryCIFARCNN(16)
    _, report = wrap_conv_modules(m, skip_name_substr=("stem", "head", "skip"), min_weight_elems=64)
    assert report.replaced
    # Small kernels pad to 64-bit words → compression < 32×; still a clear size win
    assert report.compression >= 15.0
    y = m(torch.randn(1, 3, 32, 32))
    assert y.shape == (1, 10)


def test_vision_one_epoch_tiny_subset():
    """1-epoch smoke on CIFAR-shaped synthetic batches (no network / no data/).

    Real CIFAR downloads belong in ``scripts/train_*.py`` and ``@pytest.mark.slow``
    jobs — fast CI must not hit Toronto CDN (truncated downloads fail runners).
    """
    from bnn.ste import clip_weights_

    g = torch.Generator().manual_seed(0)
    train = DataLoader(
        TensorDataset(
            torch.randn(128, 3, 32, 32, generator=g),
            torch.randint(0, 10, (128,), generator=g),
        ),
        batch_size=32,
        shuffle=True,
    )
    test = DataLoader(
        TensorDataset(
            torch.randn(32, 3, 32, 32, generator=g),
            torch.randint(0, 10, (32,), generator=g),
        ),
        batch_size=16,
    )
    model = BinaryCIFARCNN(16)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    x, y = next(iter(train))
    opt.zero_grad(set_to_none=True)
    loss = loss_fn(model(x), y)
    loss.backward()
    opt.step()
    clip_weights_(model)
    model.eval()
    with torch.no_grad():
        logits = model(next(iter(test))[0][:8])
    assert logits.shape[1] == 10
    assert torch.isfinite(logits).all()


@pytest.mark.slow
def test_vision_one_epoch_real_cifar(allow_network):
    """Same smoke against real CIFAR-10 — skipped when the CDN is unreachable.

    The synthetic test above proves the training step; this proves the real
    loader still feeds it. It is `slow`-marked so a flaky Toronto CDN can never
    redden the default CI job, and skips (rather than fails) on a download
    problem, which is an environment fault and not a defect in this repo.
    """
    from bnn.cifar import CifarDownloadError, get_cifar10_loaders
    from bnn.ste import clip_weights_

    root = Path(__file__).resolve().parents[1]
    try:
        train_loader, test_loader = get_cifar10_loaders(
            root / "data", batch_size=32, train_subset=128, seed=0
        )
    except CifarDownloadError as err:
        pytest.skip(f"CIFAR-10 unavailable: {err}")

    model = BinaryCIFARCNN(16)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    x, y = next(iter(train_loader))
    assert x.shape[1:] == (3, 32, 32)
    assert int(y.min()) >= 0 and int(y.max()) <= 9
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
