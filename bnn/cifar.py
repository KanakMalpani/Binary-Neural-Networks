"""CIFAR-10 loader without torchvision.

Sources (first hit wins):
1. ``data/cifar10_hf/{train,test}.npz`` (HF ``datasets`` dump) — preferred
2. Official pickle batches under ``cifar-10-batches-py`` (trusted CDN only)

Security: pickle is only used for the official CIFAR-10 batch format after
download from the Toronto CDN (or a pre-placed tarball under ``data_dir``).
Prefer NPZ caches; never point ``data_dir`` at untrusted pickle trees.
"""

from __future__ import annotations

import hashlib
import os
import pickle
import shutil
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
# Published checksum for cifar-10-python.tar.gz. A size floor cannot tell a
# truncated archive from a corrupt one; this can.
MD5 = "c58f30108f718f92721af3b95e74349a"
MIN_BYTES = 150_000_000
DOWNLOAD_TIMEOUT_S = 60.0

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


class CifarDownloadError(RuntimeError):
    """Network / integrity failure fetching the CIFAR-10 archive.

    Distinct from a generic RuntimeError so callers (and CI smoke tests) can
    skip on an unreachable CDN without swallowing real bugs.
    """


def _md5(path: Path, *, chunk: int = 1 << 20) -> str:
    digest = hashlib.md5()  # noqa: S324 — integrity check, not a security control
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _looks_complete(path: Path) -> bool:
    """Cheap size gate first, then the exact checksum (md5 costs ~1s on 170 MB)."""
    return path.exists() and path.stat().st_size >= MIN_BYTES and _md5(path) == MD5


def _fetch_once(dest: Path, *, timeout: float) -> None:
    """Download to a sibling temp file, then atomically replace ``dest``.

    Writing straight to ``dest`` means an interrupted transfer leaves a partial
    archive behind that a later run can mistake for a cached copy. A temp file
    plus ``os.replace`` makes the cached path all-or-nothing.
    """
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.unlink(missing_ok=True)
    try:
        # urlretrieve has no timeout parameter, so a stalled socket would hang
        # a CI job until the runner's global limit.
        with (
            urllib.request.urlopen(URL, timeout=timeout) as resp,  # noqa: S310 — fixed CDN URL
            tmp.open("wb") as out,
        ):
            shutil.copyfileobj(resp, out, length=1 << 20)
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)


def _download_pickle(data_dir: Path, *, retries: int = 3, timeout: float | None = None) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    extract = data_dir / "cifar-10-batches-py"
    if extract.exists() and (extract / "data_batch_1").exists():
        return extract

    timeout = DOWNLOAD_TIMEOUT_S if timeout is None else timeout
    candidates = [
        data_dir / "cifar-10-python-full.tar.gz",
        data_dir / "cifar-10-python.tar.gz",
    ]
    # A pre-placed archive is trusted on size alone (offline mirrors legitimately
    # differ); only downloads we performed are checksum-gated.
    tgz = next((c for c in candidates if c.exists() and c.stat().st_size >= MIN_BYTES), None)

    if tgz is None:
        tgz = data_dir / "cifar-10-python.tar.gz"
        tgz.unlink(missing_ok=True)  # drop any truncated leftover
        last_err: BaseException | None = None
        for attempt in range(1, max(1, retries) + 1):
            try:
                print(f"Downloading {URL} (attempt {attempt}/{retries}) ...", flush=True)
                _fetch_once(tgz, timeout=timeout)
                if _looks_complete(tgz):
                    break
                raise CifarDownloadError(
                    f"CIFAR archive failed verification "
                    f"({tgz.stat().st_size} bytes, expected md5 {MD5})"
                )
            except (urllib.error.URLError, OSError, CifarDownloadError) as err:
                last_err = err
                tgz.unlink(missing_ok=True)
                print(f"  attempt {attempt} failed: {err}", flush=True)
                if attempt >= max(1, retries):
                    raise CifarDownloadError(
                        f"CIFAR download failed after {max(1, retries)} attempts. "
                        "Prefer: pip install datasets && dump to data/cifar10_hf/"
                    ) from last_err
                # Back off before retrying — an immediate retry usually hits the
                # same transient CDN condition.
                time.sleep(min(2.0 ** (attempt - 1), 8.0))

    print(f"Extracting {tgz} ...", flush=True)
    with tarfile.open(tgz, "r:gz") as tar:
        _safe_extract(tar, data_dir)
    if not (extract / "data_batch_1").exists():
        raise CifarDownloadError(f"CIFAR extract missing batches under {extract}")
    return extract


def _safe_extract(tar: tarfile.TarFile, dest: Path) -> None:
    """Extract with traversal protection (CVE-2007-4559).

    ``filter="data"`` does this natively but only landed in 3.11.4/3.12, and this
    package supports 3.11 — so fall back to checking each member by hand.
    """
    try:
        tar.extractall(dest, filter="data")
        return
    except TypeError:
        pass  # older 3.11 patch level: no filter= parameter
    root = dest.resolve()
    for member in tar.getmembers():
        target = (root / member.name).resolve()
        if not target.is_relative_to(root):
            raise CifarDownloadError(f"Refusing tar member outside {root}: {member.name}")
        if member.issym() or member.islnk():
            raise CifarDownloadError(f"Refusing link member in archive: {member.name}")
    tar.extractall(dest)  # noqa: S202 — every member validated above


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
    from bnn.data import default_num_workers

    nw = default_num_workers()
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=nw),
        DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=nw),
    )
