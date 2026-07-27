"""MNIST IDX parsing and loader wiring — synthetic files, never the network."""

from __future__ import annotations

import gzip
import struct
from pathlib import Path

import numpy as np
import pytest

from bnn.data import (
    FILES,
    _read_images,
    _read_labels,
    _safe_data_dir,
    default_num_workers,
    get_mnist_loaders,
)
from bnn.paths import PathSecurityError


def _write_images(path: Path, n: int = 4, rows: int = 3, cols: int = 3) -> np.ndarray:
    raw = np.arange(n * rows * cols, dtype=np.uint8).reshape(n, 1, rows, cols)
    with gzip.open(path, "wb") as f:
        f.write(struct.pack(">IIII", 2051, n, rows, cols))
        f.write(raw.tobytes())
    return raw


def _write_labels(path: Path, labels: list[int]) -> np.ndarray:
    arr = np.array(labels, dtype=np.uint8)
    with gzip.open(path, "wb") as f:
        f.write(struct.pack(">II", 2049, arr.size))
        f.write(arr.tobytes())
    return arr


def test_read_images_normalises_to_unit_range(tmp_path: Path):
    path = tmp_path / "img.gz"
    raw = _write_images(path, n=4, rows=3, cols=3)
    out = _read_images(path)
    assert out.shape == (4, 1, 3, 3)
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0
    np.testing.assert_allclose(out, raw.astype(np.float32) / 255.0)


def test_read_labels_returns_int64(tmp_path: Path):
    path = tmp_path / "lab.gz"
    _write_labels(path, [0, 3, 9, 1])
    out = _read_labels(path)
    assert out.dtype == np.int64
    assert out.tolist() == [0, 3, 9, 1]


def test_read_images_rejects_wrong_magic(tmp_path: Path):
    path = tmp_path / "bad.gz"
    with gzip.open(path, "wb") as f:
        f.write(struct.pack(">IIII", 1234, 1, 1, 1))
        f.write(b"\x00")
    with pytest.raises(AssertionError):
        _read_images(path)


def test_read_labels_rejects_wrong_magic(tmp_path: Path):
    path = tmp_path / "bad.gz"
    with gzip.open(path, "wb") as f:
        f.write(struct.pack(">II", 1234, 1))
        f.write(b"\x00")
    with pytest.raises(AssertionError):
        _read_labels(path)


def test_safe_data_dir_creates_and_resolves(tmp_path: Path):
    target = tmp_path / "nested" / "data"
    out = _safe_data_dir(target)
    assert out.is_dir()
    assert out == target.resolve()


def test_safe_data_dir_expands_user(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    out = _safe_data_dir(Path("~") / "bnn_data")
    assert out.is_dir()


def test_safe_data_dir_rejects_escaping_filename(tmp_path: Path, monkeypatch):
    """A tampered FILES entry must not be able to write outside the data root."""
    monkeypatch.setitem(FILES, "train_images", "../escaped.gz")
    with pytest.raises(PathSecurityError, match="MNIST path rejected"):
        _safe_data_dir(tmp_path / "data")


@pytest.mark.parametrize(
    "value,expected", [("0", 0), ("4", 4), ("-1", 0), ("abc", None), ("", None)]
)
def test_default_num_workers_env_override(monkeypatch, value, expected):
    monkeypatch.setenv("BNN_NUM_WORKERS", value)
    got = default_num_workers()
    if expected is None:
        assert got >= 0  # falls through to the platform default
    else:
        assert got == expected


def test_default_num_workers_unset_is_non_negative(monkeypatch):
    monkeypatch.delenv("BNN_NUM_WORKERS", raising=False)
    assert default_num_workers() >= 0


def test_get_mnist_loaders_reads_local_files_without_download(tmp_path: Path):
    """Pre-seeded files must short-circuit the download path entirely."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_images(data_dir / FILES["train_images"], n=8, rows=4, cols=4)
    _write_labels(data_dir / FILES["train_labels"], [0, 1, 2, 3, 4, 5, 6, 7])
    _write_images(data_dir / FILES["test_images"], n=4, rows=4, cols=4)
    _write_labels(data_dir / FILES["test_labels"], [0, 1, 2, 3])

    train, test = get_mnist_loaders(data_dir, batch_size=2, num_workers=0)
    xb, yb = next(iter(train))
    assert xb.shape == (2, 1, 4, 4)
    assert yb.shape == (2,)
    assert len(test.dataset) == 4
