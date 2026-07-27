"""Scale calibration, deterministic-mode wiring, and audio feature extraction."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from bnn.audio.features import mel_like_filterbank, stft_mag, waveform_to_features
from bnn.determinism import set_repro_seed
from bnn.wrap.calibrate import (
    CalibConfig,
    absmean_scale,
    calibrate_linear_scales,
    percentile_scale,
    scale_from_weight,
)

# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------

def test_absmean_scale_per_channel_shape_and_positivity():
    w = torch.randn(8, 32)
    s = absmean_scale(w, per_channel=True)
    assert s.shape == (8,)
    assert (s > 0).all()


def test_absmean_scale_global_is_scalar():
    s = absmean_scale(torch.randn(8, 32), per_channel=False)
    assert s.ndim == 0
    assert s > 0


def test_absmean_scale_never_returns_zero_for_dead_channel():
    """A clamp floor keeps downstream division finite."""
    w = torch.zeros(4, 16)
    assert (absmean_scale(w) > 0).all()


def test_percentile_scale_per_channel_and_global():
    w = torch.randn(6, 64)
    per = percentile_scale(w, 99.0, per_channel=True)
    glob = percentile_scale(w, 99.0, per_channel=False)
    assert per.shape == (6,)
    assert glob.ndim == 0
    assert (per > 0).all() and glob > 0


def test_percentile_scale_is_monotonic_in_percentile():
    w = torch.randn(4, 128)
    lo = percentile_scale(w, 50.0, per_channel=False)
    hi = percentile_scale(w, 99.0, per_channel=False)
    assert hi >= lo


def test_scale_from_weight_dispatches_on_method():
    w = torch.randn(4, 32)
    a = scale_from_weight(w, CalibConfig(method="absmean", per_channel=False))
    p = scale_from_weight(w, CalibConfig(method="percentile", per_channel=False))
    assert torch.allclose(a, absmean_scale(w, per_channel=False))
    assert torch.allclose(p, percentile_scale(w, 99.0, per_channel=False))


def test_scale_from_weight_defaults_to_absmean():
    w = torch.randn(4, 32)
    assert torch.allclose(scale_from_weight(w), absmean_scale(w))


def test_calibrate_without_activations_is_weight_only():
    w = torch.randn(8, 32)
    assert torch.allclose(calibrate_linear_scales(w), absmean_scale(w))


def test_calibrate_with_activations_stays_finite_and_per_channel():
    """Activation-aware nudge must not change shape or blow up the scale."""
    w = torch.randn(8, 32)
    acts = [torch.randn(4, 32) for _ in range(3)]
    alpha = calibrate_linear_scales(w, activation_batches=acts)
    assert alpha.shape == (8,)
    assert torch.isfinite(alpha).all()
    assert (alpha > 0).all()
    # Weight scale stays primary: the act factor is clamped to [0.5, 2].
    ratio = alpha / absmean_scale(w)
    assert float(ratio.min()) >= 0.5 - 1e-6
    assert float(ratio.max()) <= 2.0 + 1e-6


def test_calibrate_scalar_alpha_with_activations():
    w = torch.randn(8, 32)
    alpha = calibrate_linear_scales(
        w,
        cfg=CalibConfig(per_channel=False),
        activation_batches=[torch.randn(4, 32)],
    )
    assert alpha.ndim == 0
    assert torch.isfinite(alpha) and alpha > 0


def test_calibrate_respects_max_batches():
    w = torch.randn(4, 16)
    many = [torch.randn(2, 16) for _ in range(20)]
    alpha = calibrate_linear_scales(
        w, cfg=CalibConfig(max_batches=2), activation_batches=many
    )
    assert torch.isfinite(alpha).all()


# --------------------------------------------------------------------------
# determinism
# --------------------------------------------------------------------------

def test_deterministic_mode_sets_cudnn_flags_and_reports():
    status = set_repro_seed(0, deterministic=True)
    assert status["device_policy"] == "cpu"
    assert "torch_deterministic" in status
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
    # The caveat note must survive into result JSON.
    assert any("nondeterministic" in n for n in status["notes"])


def test_force_cpu_false_reports_auto_device_policy():
    status = set_repro_seed(0, force_cpu=False, deterministic=False)
    assert status["device_policy"] == "auto"
    assert status["torch_deterministic"] is False


# --------------------------------------------------------------------------
# audio features
# --------------------------------------------------------------------------

def test_stft_mag_shape_and_non_negative():
    wav = np.sin(np.linspace(0, 20 * np.pi, 2000)).astype(np.float32)
    mag = stft_mag(wav)
    assert mag.ndim == 2
    assert (mag >= 0).all()
    assert np.isfinite(mag).all()


def test_mel_like_filterbank_reduces_frequency_bins():
    """Layout is (F, T); pooling collapses F to n_mels and preserves T."""
    wav = np.sin(np.linspace(0, 20 * np.pi, 2000)).astype(np.float32)
    mag = stft_mag(wav)
    mel = mel_like_filterbank(mag, n_mels=16)
    assert mel.shape == (16, mag.shape[1])
    assert mel.shape[0] < mag.shape[0], "pooling should reduce frequency bins"
    assert np.isfinite(mel).all()


def test_waveform_to_features_is_deterministic():
    wav = np.sin(np.linspace(0, 10 * np.pi, 1500)).astype(np.float32)
    a = waveform_to_features(wav)
    b = waveform_to_features(wav)
    np.testing.assert_allclose(a, b)
    assert np.isfinite(a).all()


def test_waveform_to_features_handles_short_signal():
    """A clip shorter than one window must not raise."""
    out = waveform_to_features(np.zeros(64, dtype=np.float32))
    assert np.isfinite(out).all()


@pytest.mark.parametrize("n_mels", [8, 40])
def test_mel_bins_configurable(n_mels: int):
    mag = stft_mag(np.random.default_rng(0).standard_normal(3000).astype(np.float32))
    out = mel_like_filterbank(mag, n_mels=n_mels)
    assert out.shape[0] == n_mels
    assert out.shape[1] == mag.shape[1]


def test_mel_filterbank_more_bins_than_frequencies_still_works():
    """n_mels above the FFT bin count must not produce empty rows or crash."""
    mag = stft_mag(np.zeros(300, dtype=np.float32))
    out = mel_like_filterbank(mag, n_mels=mag.shape[0] + 8)
    assert out.shape[0] == mag.shape[0] + 8
    assert np.isfinite(out).all()
