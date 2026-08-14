"""Hub canary cards + encode script (Wave P1). Offline; no Hub network."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def test_cards_say_canary_not_sota():
    texts = [
        _read("cards/README.md"),
        _read("cards/wrap-demo/README.md"),
        _read("cards/mnist-mlp/README.md"),
        _read("cards/codec/README.md"),
        _read("docs/HUB_BNNPACK.md"),
        _read("docs/tutorials/08_HF_OPTIMISER.md"),
    ]
    joined = "\n".join(texts).lower()
    assert "canary" in joined
    assert "not imagenet" in joined or "not-sota" in joined or "not sota" in joined
    assert "sign()" in "\n".join(texts)


def test_wrap_card_quotes_and_gate_and_ultra_refuse():
    wrap = _read("cards/wrap-demo/README.md")
    hub = _read("docs/HUB_BNNPACK.md")
    for text in (wrap, hub):
        assert "0.85" in text
        assert "1.5" in text
        assert "4096" in text
        assert "REFUSE" in text
        assert "0.70" in text or "~0.70" in text
        assert "golden_floors.json" in text
        assert "wrap_demo.json" in text


def test_mnist_card_quotes_floors():
    text = _read("cards/mnist-mlp/README.md")
    assert "95.0" in text
    assert "96.36" in text
    assert "97.67" in text
    assert "train_results.json" in text


def test_manifest_repo_ids():
    manifest = json.loads(_read("cards/manifest.json"))
    ids = {m["repo_id"] for m in manifest["models"]}
    assert ids == {
        "KanakMalpani/bnn-lab-wrap-demo",
        "KanakMalpani/bnn-lab-mnist-mlp-canary",
        "KanakMalpani/bnn-lab-codec-canary",
    }
    assert "collections/KanakMalpani/bnn-lab-bnnpack-canaries-" in manifest["collection_url"]


def test_encode_codec_and_mnist_offline(tmp_path: Path):
    script = ROOT / "scripts" / "encode_hf_canaries.py"
    out = tmp_path / "packs"
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--out-dir",
            str(out),
            "--only",
            "codec",
            "--only",
            "mnist-mlp",
            "--codec-dim",
            "128",
            "--mnist-hidden",
            "128",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    from bnn.codec import decode_file, packed_module_fp_err

    codec_pack = out / "codec.bnnpack"
    mnist_pack = out / "mnist-mlp.bnnpack"
    assert codec_pack.is_file() and mnist_pack.is_file()
    modules, meta = decode_file(codec_pack)
    assert meta.get("canary") is True
    assert packed_module_fp_err(next(iter(modules.values()))) == 0.0
    modules_m, meta_m = decode_file(mnist_pack)
    assert meta_m.get("canary") is True
    assert len(modules_m) >= 1
    for mod in modules_m.values():
        assert packed_module_fp_err(mod) == 0.0


def test_encode_wrap_small_shape(tmp_path: Path):
    """Hidden=128 keeps CI tiny; Hub wrap-demo uses committed 4096."""
    script = ROOT / "scripts" / "encode_hf_canaries.py"
    out = tmp_path / "packs"
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--out-dir",
            str(out),
            "--only",
            "wrap-demo",
            "--wrap-hidden",
            "128",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    from bnn.codec import decode_file, packed_module_fp_err

    modules, meta = decode_file(out / "wrap-demo.bnnpack")
    assert set(modules) == {"3", "5"}
    assert meta.get("hidden") == 128
    assert meta.get("qat") is False
    for mod in modules.values():
        assert packed_module_fp_err(mod) == 0.0
        assert abs((mod.in_features * mod.out_features * 4) / max(mod.packed_weight_bytes(), 1) - 32.0) < 1e-6
