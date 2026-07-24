"""Audio feature front-end: STFT magnitude (numpy) with optional torchaudio."""

from __future__ import annotations

import numpy as np


def stft_mag(
    wave: np.ndarray,
    *,
    sr: int = 16000,
    n_fft: int = 256,
    hop: int = 128,
) -> np.ndarray:
    """Return log-magnitude spectrogram (F, T) for a 1-D float waveform."""
    wave = np.asarray(wave, dtype=np.float32).reshape(-1)
    if wave.size < n_fft:
        wave = np.pad(wave, (0, n_fft - wave.size))
    # Frame
    n_frames = 1 + (wave.size - n_fft) // hop
    if n_frames < 1:
        n_frames = 1
        wave = np.pad(wave, (0, n_fft))
    frames = np.lib.stride_tricks.as_strided(
        wave,
        shape=(n_frames, n_fft),
        strides=(wave.strides[0] * hop, wave.strides[0]),
        writeable=False,
    ).copy()
    window = np.hanning(n_fft).astype(np.float32)
    frames *= window
    spec = np.fft.rfft(frames, axis=1)
    mag = np.abs(spec).T.astype(np.float32)  # F, T
    return np.log1p(mag)


def mel_like_filterbank(mag: np.ndarray, n_mels: int = 40) -> np.ndarray:
    """Cheap triangular pooling over frequency (not a full Mel scale — portable)."""
    f, t = mag.shape
    edges = np.linspace(0, f, n_mels + 2).astype(int)
    out = np.zeros((n_mels, t), dtype=np.float32)
    for i in range(n_mels):
        a, b, c = edges[i], edges[i + 1], edges[i + 2]
        if b <= a:
            b = a + 1
        if c <= b:
            c = b + 1
        left = mag[a:b]
        right = mag[b:c]
        if left.size:
            out[i] += left.mean(axis=0)
        if right.size:
            out[i] += right.mean(axis=0)
        out[i] *= 0.5
    return out


def waveform_to_features(
    wave: np.ndarray,
    *,
    sr: int = 16000,
    n_mels: int = 40,
    target_frames: int = 32,
) -> np.ndarray:
    """(n_mels, target_frames) float32 features."""
    mag = stft_mag(wave, sr=sr)
    feats = mel_like_filterbank(mag, n_mels=n_mels)
    # Time pad/crop
    t = feats.shape[1]
    if t < target_frames:
        feats = np.pad(feats, ((0, 0), (0, target_frames - t)))
    else:
        feats = feats[:, :target_frames]
    # Normalize
    mu = feats.mean()
    sd = feats.std() + 1e-5
    return ((feats - mu) / sd).astype(np.float32)
