"""CIFAR-10 loader without torchvision.

Sources (first hit wins):
1. ``data/cifar10_hf/{train,test}.npz`` (HF ``datasets`` dump) — preferred
2. Official pickle batches under ``cifar-10-batches-py`` (trusted CDN only)

Security: pickle is only used for the official CIFAR-10 batch format after
download from the Toronto CDN (or a pre-placed tarball under ``data_dir``).
Prefer NPZ caches; never point ``data_dir`` at untrusted pickle trees.
"""

from __future__ import annotations

import pickle
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(1, 3, 1, 1)
_STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32).reshape(1, 3, 1, 1)


def _from_hf_npz(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    hf = data_dir / "cifar10_hf"
    tr, te = hf / "train.npz", hf / "test.npz"
    if not (tr.exists() and te.exists()):
        return None
    train = np.load(tr)
    test = np.load(te)
    x_train = train["images"].astype(np.float32) / 255.0
    y_train = train["labels"].astype(np.int64)
    x_test = test["images"].astype(np.float32) / 255.0
    y_test = test["labels"].astype(np.int64)
    return x_train, y_train, x_test, y_test


def _download_pickle(data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    extract = data_dir / "cifar-10-batches-py"
    if extract.exists() and (extract / "data_batch_1").exists():
        return extract
    candidates = [
        data_dir / "cifar-10-python-full.tar.gz",
        data_dir / "cifar-10-python.tar.gz",
    ]
    tgz = next((c for c in candidates if c.exists() and c.stat().st_size > 150_000_000), None)
    if tgz is None:
        tgz = data_dir / "cifar-10-python.tar.gz"
        print(f"Downloading {URL} ...", flush=True)
        urllib.request.urlretrieve(URL, tgz)
        if tgz.stat().st_size < 150_000_000:
            raise RuntimeError(
                f"CIFAR download truncated ({tgz.stat().st_size} bytes). "
                "Prefer: pip install datasets && dump to data/cifar10_hf/"
            )
    print(f"Extracting {tgz} ...", flush=True)
    with tarfile.open(tgz, "r:gz") as tar:
        tar.extractall(data_dir)
    if not (extract / "data_batch_1").exists():
        raise RuntimeError(f"CIFAR extract missing batches under {extract}")
    return extract


def _load_batch(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load one official CIFAR-10 pickle batch (structure-validated).

    Prefer NPZ via ``cifar10_hf/`` — pickle is only for the Toronto CDN layout.
    """
    path = Path(path).resolve()
    # Only accept known batch filenames under a cifar-10-batches-py tree
    if path.name not in {*(f"data_batch_{i}" for i in range(1, 6)), "test_batch"}:
        raise ValueError(f"Unexpected CIFAR batch name: {path.name}")
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="bytes")  # noqa: S301 — official CIFAR-10 only
    if not isinstance(d, dict) or b"data" not in d or b"labels" not in d:
        raise ValueError(f"Not a CIFAR-10 batch dict: {path}")
    data = np.asarray(d[b"data"])
    if data.ndim != 2 or data.shape[1] != 3072:
        raise ValueError(f"Bad CIFAR data shape {data.shape} in {path}")
    raw = data.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    labels = np.array(d[b"labels"], dtype=np.int64)
    if labels.shape[0] != raw.shape[0]:
        raise ValueError("CIFAR labels/data length mismatch")
    return raw, labels


def get_cifar10_loaders(
    data_dir: Path,
    batch_size: int,
    *,
    train_subset: int | None = None,
    seed: int = 0,
) -> tuple[DataLoader, DataLoader]:
    data_dir = Path(data_dir).expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    loaded = _from_hf_npz(data_dir)
    if loaded is not None:
        x_train, y_train, x_test, y_test = loaded
        print("CIFAR-10 source: data/cifar10_hf/*.npz", flush=True)
    else:
        root = _download_pickle(data_dir)
        xs, ys = [], []
        for i in range(1, 6):
            x, y = _load_batch(root / f"data_batch_{i}")
            xs.append(x)
            ys.append(y)
        x_train = np.concatenate(xs)
        y_train = np.concatenate(ys)
        x_test, y_test = _load_batch(root / "test_batch")
        print("CIFAR-10 source: pickle batches", flush=True)

    x_train = (x_train - _MEAN) / _STD
    x_test = (x_test - _MEAN) / _STD

    if train_subset is not None and train_subset < len(x_train):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(x_train), size=train_subset, replace=False)
        x_train, y_train = x_train[idx], y_train[idx]

    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    test_ds = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(test_ds, batch_size=256, shuffle=False),
    )
