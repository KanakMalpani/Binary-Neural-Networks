"""Path safety + packed validation fail-fast."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bnn.kernels.packed import binary_gemm_packed, pack_binary_pm1
from bnn.paths import (
    REPO_ROOT,
    PathSecurityError,
    data_path,
    repo_relative,
    resolve_under,
    results_path,
)


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


@pytest.mark.parametrize(
    "escape",
    [
        "../outside.txt",
        "a/../../outside.txt",       # escapes only after normalisation
        "./../outside.txt",
        "a/b/../../../outside.txt",
    ],
)
def test_resolve_under_blocks_normalised_escapes(tmp_path: Path, escape: str):
    """Traversal must be caught after resolution, not by a naive prefix check."""
    root = tmp_path / "data"
    (root / "a" / "b").mkdir(parents=True)
    with pytest.raises(PathSecurityError):
        resolve_under(root, escape)


def test_resolve_under_blocks_absolute_path_outside(tmp_path: Path):
    root = tmp_path / "data"
    root.mkdir()
    outside = tmp_path / "elsewhere" / "secret.txt"
    outside.parent.mkdir()
    outside.write_text("x", encoding="utf-8")
    with pytest.raises(PathSecurityError):
        resolve_under(root, outside)


def test_resolve_under_allows_absolute_path_inside(tmp_path: Path):
    root = tmp_path / "data"
    root.mkdir()
    inside = root / "ok.bin"
    assert resolve_under(root, inside) == inside.resolve()


def test_resolve_under_must_exist(tmp_path: Path):
    root = tmp_path / "data"
    root.mkdir()
    with pytest.raises(FileNotFoundError):
        resolve_under(root, "nope.bin", must_exist=True)
    (root / "yes.bin").write_text("x", encoding="utf-8")
    assert resolve_under(root, "yes.bin", must_exist=True).name == "yes.bin"


def test_data_and_results_path_stay_under_repo():
    assert data_path().parent == REPO_ROOT
    assert results_path().parent == REPO_ROOT
    assert results_path("a", "b.json").parts[-2:] == ("a", "b.json")
    with pytest.raises(PathSecurityError):
        results_path("..", "escape.json")


def test_repo_relative_is_portable_posix():
    """Committed JSON must not embed a machine-specific absolute path."""
    rel = repo_relative(REPO_ROOT / "results" / "benchmark.json")
    assert rel == "results/benchmark.json"
    assert "\\" not in rel
    assert not Path(rel).is_absolute()


def test_repo_relative_falls_back_to_name_outside_repo(tmp_path: Path):
    outside = tmp_path / "somewhere" / "thing.json"
    assert repo_relative(outside) == "thing.json"


def test_pack_rejects_empty():
    with pytest.raises(ValueError):
        pack_binary_pm1(np.array([]).reshape(0, 8))


def test_gemm_rejects_bad_ndim():
    with pytest.raises(ValueError):
        binary_gemm_packed(np.ones(8), np.ones((4, 8)))
