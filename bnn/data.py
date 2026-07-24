"""Minimal MNIST loader (no torchvision required)."""

from __future__ import annotations

import gzip
import struct
import urllib.request
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

BASE = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def _download(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in FILES.values():
        dest = data_dir / name
        if dest.exists():
            continue
        url = BASE + name
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, dest)


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


def get_mnist_loaders(
    data_dir: Path, batch_size: int
) -> tuple[DataLoader, DataLoader]:
    _download(data_dir)
    mean, std = 0.1307, 0.3081
    x_train = (_read_images(data_dir / FILES["train_images"]) - mean) / std
    y_train = _read_labels(data_dir / FILES["train_labels"])
    x_test = (_read_images(data_dir / FILES["test_images"]) - mean) / std
    y_test = _read_labels(data_dir / FILES["test_labels"])

    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    test_ds = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)
    return train_loader, test_loader
