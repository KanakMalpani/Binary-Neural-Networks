"""Checkpoint/pack export, wrap guardrails, and ImageNet folder checks."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from bnn.export import (
    load_checkpoint,
    load_packed_linears,
    pack_linear_weight,
    save_checkpoint,
    save_packed_linears,
)
from bnn.vision import check_imagenet_folder, describe_imagenet_folder_layout
from bnn.wrap.guardrails import (
    HARD_REFUSE_IN,
    HARD_REFUSE_OUT,
    check_linear_wrap_guardrails,
)


def _toy() -> nn.Module:
    return nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 8))


# --------------------------------------------------------------------------
# checkpoints
# --------------------------------------------------------------------------

def test_checkpoint_round_trip_restores_weights(tmp_path: Path):
    model = _toy()
    path = save_checkpoint(model, tmp_path / "m.pt", meta={"epoch": 3})
    assert path.is_file()

    restored = _toy()
    assert not torch.allclose(restored[0].weight, model[0].weight)
    meta = load_checkpoint(restored, path)
    assert meta == {"epoch": 3}
    assert torch.allclose(restored[0].weight, model[0].weight)


def test_checkpoint_creates_missing_parent_dirs(tmp_path: Path):
    path = save_checkpoint(_toy(), tmp_path / "a" / "b" / "m.pt")
    assert path.is_file()


def test_load_checkpoint_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_checkpoint(_toy(), tmp_path / "nope.pt")


def test_load_checkpoint_rejects_payload_without_state_dict(tmp_path: Path):
    bad = tmp_path / "bad.pt"
    torch.save({"not_a_state_dict": 1}, bad)
    with pytest.raises(ValueError, match="state_dict"):
        load_checkpoint(_toy(), bad)


def test_missing_meta_yields_empty_dict(tmp_path: Path):
    path = tmp_path / "nometa.pt"
    torch.save({"state_dict": _toy().state_dict()}, path)
    assert load_checkpoint(_toy(), path) == {}


# --------------------------------------------------------------------------
# packed export
# --------------------------------------------------------------------------

def test_pack_linear_weight_reports_exact_compression():
    w = torch.randn(64, 256)  # in_features multiple of 64 → exact 32x
    blob = pack_linear_weight(w)
    assert blob["in_features"] == 256
    assert blob["out_features"] == 64
    assert blob["n"] == 256
    assert blob["compression"] == pytest.approx(32.0)
    assert blob["packed_bytes"] * 32 == blob["fp32_bytes"]
    assert blob["alpha"] > 0


def test_pack_linear_weight_unaligned_is_below_32x():
    """Padding to a 64-bit word boundary costs real compression — report it."""
    blob = pack_linear_weight(torch.randn(8, 100))
    assert blob["compression"] < 32.0


def test_save_and_load_packed_linears_round_trip(tmp_path: Path):
    model = _toy()
    path = save_packed_linears(model, tmp_path / "packed.pt")
    payload = load_packed_linears(path)
    # nn.Sequential names Linears "0" and "2".
    assert set(payload) == {"0", "2"}
    assert payload["0"]["in_features"] == 64
    assert payload["0"]["bias"] is not None
    assert payload["0"]["packed"].dtype.name == "uint64"


def test_packed_export_records_bias_none(tmp_path: Path):
    model = nn.Sequential(nn.Linear(64, 8, bias=False))
    payload = load_packed_linears(save_packed_linears(model, tmp_path / "p.pt"))
    assert payload["0"]["bias"] is None


def test_load_packed_linears_rejects_non_dict(tmp_path: Path):
    path = tmp_path / "list.pt"
    torch.save([1, 2, 3], path)
    with pytest.raises(ValueError, match="not a dict"):
        load_packed_linears(path)


# --------------------------------------------------------------------------
# wrap guardrails
# --------------------------------------------------------------------------

def test_wide_binary_linear_is_accepted():
    verdict = check_linear_wrap_guardrails(nn.Linear(4096, 4096))
    assert verdict.ok
    assert verdict.code == "OK"


def test_pathologically_narrow_binary_is_refused():
    verdict = check_linear_wrap_guardrails(nn.Linear(HARD_REFUSE_IN - 1, 64))
    assert not verdict.ok
    assert verdict.code == "NARROW_BINARY"
    # Message must name the escape hatch, not just say no.
    assert "force" in verdict.message.lower()
    assert "ternary_weight_only" in verdict.message


def test_narrow_out_features_also_refused():
    verdict = check_linear_wrap_guardrails(nn.Linear(4096, HARD_REFUSE_OUT - 1))
    assert not verdict.ok


def test_force_overrides_refusal_but_records_it():
    verdict = check_linear_wrap_guardrails(nn.Linear(16, 4), force=True)
    assert verdict.ok
    assert verdict.code == "FORCED_NARROW"


def test_moderately_narrow_is_allowed_with_efficiency_warning():
    verdict = check_linear_wrap_guardrails(nn.Linear(64, 64))
    assert verdict.ok
    assert verdict.code == "SUBOPTIMAL_WIDTH"
    assert "wall-clock" in verdict.message


def test_non_binary_modes_skip_width_guardrails():
    verdict = check_linear_wrap_guardrails(nn.Linear(8, 4), mode="ternary_weight_only")
    assert verdict.ok
    assert verdict.code == "OK"


# --------------------------------------------------------------------------
# ImageNet folder stub
# --------------------------------------------------------------------------

def test_imagenet_layout_hint_mentions_non_goal(tmp_path: Path):
    text = describe_imagenet_folder_layout(tmp_path)
    assert "train/" in text and "val/" in text
    assert "NON-GOAL" in text


def test_check_imagenet_folder_missing(tmp_path: Path):
    report = check_imagenet_folder(tmp_path)
    assert report["ok"] is False
    assert report["train_classes"] == 0
    assert report["val_classes"] == 0


def test_check_imagenet_folder_valid(tmp_path: Path):
    for split in ("train", "val"):
        (tmp_path / split / "cat").mkdir(parents=True)
        (tmp_path / split / "dog").mkdir(parents=True)
    report = check_imagenet_folder(tmp_path)
    assert report["ok"] is True
    assert report["train_classes"] == 2
    assert report["val_classes"] == 2


def test_check_imagenet_folder_empty_class_dirs_not_ok(tmp_path: Path):
    (tmp_path / "train").mkdir()
    (tmp_path / "val").mkdir()
    assert check_imagenet_folder(tmp_path)["ok"] is False
