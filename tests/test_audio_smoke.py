"""Audio smoke: synthetic features + 1-step train (no network)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from bnn.audio.data import get_audio_loaders, synthesize_tone
from bnn.audio.features import waveform_to_features
from bnn.audio.models import BinaryAudioCNN, FP32AudioCNN, build_audio_model
from bnn.ste import clip_weights_, set_approx_sign


def test_waveform_features_shape():
    w = synthesize_tone(3, rng=np.random.default_rng(0))
    feats = waveform_to_features(w, n_mels=40, target_frames=32)
    assert feats.shape == (40, 32)
    assert np.isfinite(feats).all()


def test_audio_loaders_offline():
    train, test, meta = get_audio_loaders(
        batch_size=16, n_train=64, n_test=16, n_classes=4, seed=1, cache_dir=None
    )
    assert meta["source"] == "synthetic_tones"
    x, y = next(iter(train))
    assert x.ndim == 4 and x.shape[1] == 1
    assert y.dtype == torch.int64


def test_audio_forward():
    set_approx_sign(False)
    x = torch.randn(2, 1, 40, 32)
    assert FP32AudioCNN(4, 16)(x).shape == (2, 4)
    assert BinaryAudioCNN(4, 16)(x).shape == (2, 4)
    assert build_audio_model("binary_mlp", n_classes=4)(x).shape == (2, 4)


def test_audio_one_epoch_smoke():
    train, test, _ = get_audio_loaders(
        batch_size=16, n_train=48, n_test=16, n_classes=4, seed=0, cache_dir=None
    )
    model = BinaryAudioCNN(n_classes=4, channels=16)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    x, y = next(iter(train))
    opt.zero_grad(set_to_none=True)
    loss = loss_fn(model(x), y)
    loss.backward()
    opt.step()
    clip_weights_(model)
    model.eval()
    with torch.no_grad():
        logits = model(next(iter(test))[0])
    assert logits.shape[1] == 4
    assert torch.isfinite(logits).all()
