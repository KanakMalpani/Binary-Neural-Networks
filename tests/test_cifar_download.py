"""CIFAR download resilience: retries, integrity, atomicity, safe extraction.

Network is never touched — ``_fetch_once`` is monkeypatched. These assert the
behaviour that keeps CI green: a truncated transfer must be retried and must
never be cached as if it succeeded.
"""

from __future__ import annotations

import socket
import tarfile
from pathlib import Path

import pytest

from bnn import cifar
from bnn.cifar import MD5, MIN_BYTES, CifarDownloadError, _download_pickle, _md5, _safe_extract


def _good_bytes() -> bytes:
    return b"\x00" * MIN_BYTES


# --------------------------------------------------------------------------
# the network guard itself
# --------------------------------------------------------------------------

def test_network_guard_blocks_sockets_in_fast_tests():
    """Meta-test: the guard that keeps CI deterministic must actually fire.

    Matched by message rather than class: conftest is imported by pytest as a
    top-level module, so `from tests.conftest import ...` would create a second
    module object and a non-identical exception class.
    """
    with pytest.raises(RuntimeError, match="must not use the network"):
        socket.socket()
    with pytest.raises(RuntimeError, match="must not use the network"):
        socket.create_connection(("example.invalid", 80))


def test_allow_network_fixture_restores_sockets(allow_network):
    s = socket.socket()          # constructing must no longer raise
    s.close()


# --------------------------------------------------------------------------
# checksum helper
# --------------------------------------------------------------------------

def test_md5_matches_hashlib(tmp_path: Path):
    import hashlib

    p = tmp_path / "blob.bin"
    payload = b"binary neural networks" * 1000
    p.write_bytes(payload)
    assert _md5(p) == hashlib.md5(payload).hexdigest()  # noqa: S324


def test_md5_streams_large_files(tmp_path: Path):
    """Chunked read must not blow memory or change the digest."""
    import hashlib

    p = tmp_path / "big.bin"
    payload = b"x" * (5 << 20)
    p.write_bytes(payload)
    assert _md5(p, chunk=4096) == hashlib.md5(payload).hexdigest()  # noqa: S324


# --------------------------------------------------------------------------
# download retry / integrity
# --------------------------------------------------------------------------

def test_truncated_download_is_retried_then_reported(tmp_path: Path, monkeypatch):
    """A short archive must never be accepted, and must be retried."""
    attempts: list[int] = []

    def fake_fetch(dest: Path, *, timeout: float) -> None:
        attempts.append(1)
        dest.write_bytes(b"\x00" * 1024)  # far below MIN_BYTES

    monkeypatch.setattr(cifar, "_fetch_once", fake_fetch)
    monkeypatch.setattr(cifar.time, "sleep", lambda s: None)  # no real backoff

    with pytest.raises(CifarDownloadError, match="failed after 3 attempts"):
        _download_pickle(tmp_path, retries=3)
    assert len(attempts) == 3


def test_truncated_download_is_not_left_on_disk(tmp_path: Path, monkeypatch):
    """A cached partial file would poison every later run."""
    def fake_fetch(dest: Path, *, timeout: float) -> None:
        dest.write_bytes(b"\x00" * 1024)

    monkeypatch.setattr(cifar, "_fetch_once", fake_fetch)
    monkeypatch.setattr(cifar.time, "sleep", lambda s: None)
    with pytest.raises(CifarDownloadError):
        _download_pickle(tmp_path, retries=2)
    assert not (tmp_path / "cifar-10-python.tar.gz").exists()


def test_network_error_is_retried_and_wrapped(tmp_path: Path, monkeypatch):
    import urllib.error

    calls: list[int] = []

    def fake_fetch(dest: Path, *, timeout: float) -> None:
        calls.append(1)
        raise urllib.error.ContentTooShortError("truncated", None)

    monkeypatch.setattr(cifar, "_fetch_once", fake_fetch)
    monkeypatch.setattr(cifar.time, "sleep", lambda s: None)
    with pytest.raises(CifarDownloadError):
        _download_pickle(tmp_path, retries=2)
    assert len(calls) == 2


def test_retry_backs_off_between_attempts(tmp_path: Path, monkeypatch):
    """Immediate retries usually hit the same transient CDN state."""
    slept: list[float] = []

    def fake_fetch(dest: Path, *, timeout: float) -> None:
        raise OSError("connection reset")

    monkeypatch.setattr(cifar, "_fetch_once", fake_fetch)
    monkeypatch.setattr(cifar.time, "sleep", slept.append)
    with pytest.raises(CifarDownloadError):
        _download_pickle(tmp_path, retries=3)
    # One sleep between each pair of attempts, never after the last.
    assert len(slept) == 2
    assert slept == sorted(slept), "backoff should be non-decreasing"


def test_wrong_checksum_is_rejected_even_at_full_size(tmp_path: Path, monkeypatch):
    """Size alone cannot detect corruption; the md5 gate must."""
    def fake_fetch(dest: Path, *, timeout: float) -> None:
        dest.write_bytes(_good_bytes())  # right size, wrong content

    monkeypatch.setattr(cifar, "_fetch_once", fake_fetch)
    monkeypatch.setattr(cifar.time, "sleep", lambda s: None)
    with pytest.raises(CifarDownloadError):
        _download_pickle(tmp_path, retries=1)


def test_existing_extracted_tree_short_circuits(tmp_path: Path, monkeypatch):
    """Already-extracted batches must not trigger any download."""
    extract = tmp_path / "cifar-10-batches-py"
    extract.mkdir()
    (extract / "data_batch_1").write_bytes(b"x")

    def explode(*a, **k):
        raise AssertionError("must not download when batches already exist")

    monkeypatch.setattr(cifar, "_fetch_once", explode)
    assert _download_pickle(tmp_path) == extract


def test_preplaced_archive_is_used_without_download(tmp_path: Path, monkeypatch):
    """Offline mirrors are trusted on size — only our own fetches are md5-gated."""
    tgz = tmp_path / "cifar-10-python.tar.gz"
    tgz.write_bytes(_good_bytes())

    def explode(*a, **k):
        raise AssertionError("must not download when an archive is present")

    monkeypatch.setattr(cifar, "_fetch_once", explode)
    # Extraction fails (it is not a real tar), but only *after* skipping download.
    with pytest.raises((tarfile.ReadError, CifarDownloadError)):
        _download_pickle(tmp_path)


def test_published_md5_constant_is_the_canonical_one():
    """Guards against a careless edit silently disabling integrity checking."""
    assert MD5 == "c58f30108f718f92721af3b95e74349a"
    assert MIN_BYTES > 100_000_000


# --------------------------------------------------------------------------
# safe extraction (CVE-2007-4559)
# --------------------------------------------------------------------------

def test_safe_extract_allows_normal_members(tmp_path: Path):
    src = tmp_path / "payload.txt"
    src.write_text("ok", encoding="utf-8")
    archive = tmp_path / "a.tar"
    with tarfile.open(archive, "w") as tar:
        tar.add(src, arcname="cifar-10-batches-py/data_batch_1")

    dest = tmp_path / "out"
    dest.mkdir()
    with tarfile.open(archive) as tar:
        _safe_extract(tar, dest)
    assert (dest / "cifar-10-batches-py" / "data_batch_1").is_file()


def test_safe_extract_refuses_path_traversal(tmp_path: Path):
    src = tmp_path / "evil.txt"
    src.write_text("pwned", encoding="utf-8")
    archive = tmp_path / "evil.tar"
    with tarfile.open(archive, "w") as tar:
        tar.add(src, arcname="../escaped.txt")

    dest = tmp_path / "out"
    dest.mkdir()
    with (
        tarfile.open(archive) as tar,
        pytest.raises((CifarDownloadError, tarfile.TarError, ValueError)),
    ):
        _safe_extract(tar, dest)
    assert not (tmp_path / "escaped.txt").exists()
