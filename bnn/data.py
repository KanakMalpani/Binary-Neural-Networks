"""Minimal MNIST loader (no torchvision required)."""

from __future__ import annotations

import gzip
import struct
import urllib.request
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .paths import PathSecurityError, resolve_under

BASE = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def _safe_data_dir(data_dir: Path) -> Path:
    """Resolve data_dir; reject traversal via weird symlinks when nested under cwd."""
    data_dir = Path(data_dir).expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    # Ensure filenames below stay under this resolved root
    for name in FILES.values():
        try:
            resolve_under(data_dir, name)
        except PathSecurityError as exc:
            raise PathSecurityError(f"MNIST path rejected: {exc}") from exc
    return data_dir


def _download(data_dir: Path) -> None:
    data_dir = _safe_data_dir(data_dir)
    for name in FILES.values():
        dest = resolve_under(data_dir, name)
        if dest.exists():
            continue
        url = BASE + name
        print(f"Downloading {url}", flush=True)
        urllib.request.urlretrieve(url, dest)  # noqa: S310 — fixed CDN URL


def _read_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, 1, rows, cols)
    return data.astype(np.float32) / 255.0


def _read_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 2049
        return np.frombuffer(f.read(), dtype=np.uint8).astype(np.int64)


def default_num_workers() -> int:
    """Windows spawn + TensorDataset usually wants 0; override with BNN_NUM_WORKERS."""
    import os

    raw = os.environ.get("BNN_NUM_WORKERS")
    if raw is not None and raw.strip() != "":
        try:
            return max(0, int(raw))
        except ValueError:
            pass
    return 0 if os.name == "nt" else 2


def get_mnist_loaders(
    data_dir: Path, batch_size: int, num_workers: int | None = None
) -> tuple[DataLoader, DataLoader]:
    data_dir = _safe_data_dir(data_dir)
    _download(data_dir)
    mean, std = 0.1307, 0.3081
    x_train = (_read_images(resolve_under(data_dir, FILES["train_images"])) - mean) / std
    y_train = _read_labels(resolve_under(data_dir, FILES["train_labels"]))
    x_test = (_read_images(resolve_under(data_dir, FILES["test_images"])) - mean) / std
    y_test = _read_labels(resolve_under(data_dir, FILES["test_labels"]))

    nw = default_num_workers() if num_workers is None else num_workers
    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    test_ds = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=nw
    )
    test_loader = DataLoader(
        test_ds, batch_size=512, shuffle=False, num_workers=nw
    )
    return train_loader, test_loader
