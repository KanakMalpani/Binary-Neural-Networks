"""Path safety + packed validation fail-fast + pickle / untrusted-pack policy."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from bnn.codec import encode_file, load_bnnpack
from bnn.export import load_checkpoint, save_checkpoint
from bnn.kernels.packed import binary_gemm_packed, pack_binary_pm1
from bnn.layers import BinaryLinear
from bnn.paths import (
    REPO_ROOT,
    PathSecurityError,
    data_path,
    is_under_repo_trusted_pack_root,
    repo_relative,
    resolve_under,
    results_path,
    warn_untrusted_pack,
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
        "a/../../outside.txt",
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


def test_checkpoint_roundtrip_weights_only(tmp_path: Path):
    """Trusted STE checkpoints round-trip under weights_only=True."""
    model = nn.Sequential(BinaryLinear(16, 8))
    path = tmp_path / "ste.pt"
    save_checkpoint(model, path, meta={"note": "trusted"})
    model2 = nn.Sequential(BinaryLinear(16, 8))
    meta = load_checkpoint(model2, path)
    assert meta.get("note") == "trusted"
    for (n1, p1), (n2, p2) in zip(
        model.named_parameters(), model2.named_parameters(), strict=True
    ):
        assert n1 == n2
        assert torch.equal(p1, p2)


def test_bnnpack_refuses_unsafe_pickle_fallback(tmp_path: Path):
    """W10.T03/T06 — corrupted / non-tensor packs must not fall back to pickle."""
    path = tmp_path / "evil.bnnpack"
    path.write_bytes(b"not-a-torch-pickle")
    with pytest.raises(ValueError) as ei:
        load_bnnpack(path)
    msg = str(ei.value).lower()
    assert "weights_only" in msg or "refusing" in msg or "unsafe" in msg


def test_warn_untrusted_pack_outside_lab_roots(tmp_path: Path, capsys):
    outsider = tmp_path / "downloaded.bnnpack"
    outsider.write_text("x", encoding="utf-8")
    assert is_under_repo_trusted_pack_root(outsider) is False
    assert warn_untrusted_pack(outsider) is True
    err = capsys.readouterr().err
    assert "untrusted" in err.lower()


def test_no_warn_for_results_pack(tmp_path: Path, monkeypatch, capsys):
    """Packs under repo results/ are lab-local — no soft warning."""
    fake_root = tmp_path / "repo"
    (fake_root / "results").mkdir(parents=True)
    pack = fake_root / "results" / "toy.bnnpack"
    pack.write_text("x", encoding="utf-8")
    monkeypatch.setattr("bnn.paths.REPO_ROOT", fake_root)
    assert is_under_repo_trusted_pack_root(pack) is True
    assert warn_untrusted_pack(pack) is False
    assert "untrusted" not in capsys.readouterr().err.lower()


def test_load_bnnpack_warns_outside_lab(tmp_path: Path, capsys):
    model = nn.Sequential(BinaryLinear(64, 32))
    pack = tmp_path / "outside.bnnpack"
    encode_file(model, pack, meta={"t": 1}, min_in_features=1)
    load_bnnpack(pack)
    err = capsys.readouterr().err
    assert "untrusted" in err.lower()
