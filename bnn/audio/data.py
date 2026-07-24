"""Synthetic tone / digit-like audio classification dataset (offline-friendly)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .features import waveform_to_features

CLASS_FREQS_HZ = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25]  # C4–C5


def synthesize_tone(
    class_id: int,
    *,
    sr: int = 16000,
    duration_s: float = 0.5,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    rng = rng or np.random.default_rng()
    f0 = CLASS_FREQS_HZ[class_id % len(CLASS_FREQS_HZ)]
    # Slight jitter + harmonics for realism
    f0 *= float(rng.uniform(0.98, 1.02))
    t = np.arange(int(sr * duration_s), dtype=np.float32) / sr
    wave = 0.6 * np.sin(2 * np.pi * f0 * t)
    wave += 0.25 * np.sin(2 * np.pi * (2 * f0) * t)
    wave += 0.05 * rng.normal(size=t.shape).astype(np.float32)
    # Fade
    fade = min(256, wave.size // 10)
    wave[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)
    wave[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
    return wave.astype(np.float32)


def build_synthetic_audio_arrays(
    *,
    n_train: int = 800,
    n_test: int = 200,
    n_classes: int = 8,
    seed: int = 0,
    n_mels: int = 40,
    frames: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    def make(n):
        xs, ys = [], []
        for i in range(n):
            y = int(i % n_classes)
            w = synthesize_tone(y, rng=rng)
            xs.append(waveform_to_features(w, n_mels=n_mels, target_frames=frames))
            ys.append(y)
        return np.stack(xs), np.array(ys, dtype=np.int64)

    return (*make(n_train), *make(n_test))


def get_audio_loaders(
    *,
    batch_size: int = 64,
    n_train: int = 800,
    n_test: int = 200,
    n_classes: int = 8,
    seed: int = 0,
    cache_dir: Path | None = None,
) -> tuple[DataLoader, DataLoader, dict]:
    """Always works offline via synthetic tones. Optional NPZ cache."""
    cache_dir = Path(cache_dir).expanduser().resolve() if cache_dir else None
    cache = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        # Filename is constructed from ints only — no user path segments
        fname = f"synth_audio_{int(n_train)}_{int(n_test)}_{int(n_classes)}_{int(seed)}.npz"
        cache = (cache_dir / fname).resolve()
        if not str(cache).startswith(str(cache_dir)):
            raise ValueError(f"audio cache path escapes cache_dir: {cache}")
    if cache is not None and cache.exists():
        z = np.load(cache, allow_pickle=False)
        x_tr, y_tr, x_te, y_te = z["x_tr"], z["y_tr"], z["x_te"], z["y_te"]
    else:
        x_tr, y_tr, x_te, y_te = build_synthetic_audio_arrays(
            n_train=n_train, n_test=n_test, n_classes=n_classes, seed=seed
        )
        if cache is not None:
            np.savez_compressed(cache, x_tr=x_tr, y_tr=y_tr, x_te=x_te, y_te=y_te)

    # Add channel dim for CNN: B,1,M,T
    x_tr_t = torch.from_numpy(x_tr).unsqueeze(1)
    x_te_t = torch.from_numpy(x_te).unsqueeze(1)
    meta = {
        "source": "synthetic_tones",
        "n_classes": n_classes,
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "feature_shape": list(x_tr.shape[1:]),
        "note": (
            "Synthetic musical-tone spectrograms (offline). "
            "Production ASR → INT8 Whisper/ORT; classic BNN ASR is research-grade."
        ),
    }
    train_ds = TensorDataset(x_tr_t, torch.from_numpy(y_tr))
    test_ds = TensorDataset(x_te_t, torch.from_numpy(y_te))
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
        meta,
    )
