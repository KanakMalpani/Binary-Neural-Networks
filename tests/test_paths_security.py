"""Path safety + packed validation fail-fast."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bnn.kernels.packed import binary_gemm_packed, pack_binary_pm1
from bnn.paths import PathSecurityError, resolve_under


def test_resolve_under_blocks_escape(tmp_path: Path):
    root = tmp_path / "data"
    root.mkdir()
    with pytest.raises(PathSecurityError):
        resolve_under(root, Path("..") / "outside.txt")


def test_resolve_under_allows_nested(tmp_path: Path):
    root = tmp_path / "data"
    root.mkdir()
    (root / "a").mkdir()
    p = resolve_under(root, Path("a") / "f.bin")
    assert p.parent == root / "a"


def test_pack_rejects_empty():
    with pytest.raises(ValueError):
        pack_binary_pm1(np.array([]).reshape(0, 8))


def test_gemm_rejects_bad_ndim():
    with pytest.raises(ValueError):
        binary_gemm_packed(np.ones(8), np.ones((4, 8)))
