"""ResNet-BiReal + ImageNet protocol smoke (no network / no SOTA gate)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn as nn

from bnn.ste import clip_weights_, set_approx_sign
from bnn.vision import (
    IMAGENET_DATASET_CONTRACT,
    check_imagenet_folder,
    make_proxy_imagenet,
    write_dataset_contract,
)
from bnn.vision.models import (
    ResNetBiReal,
    ResNetBiReal18,
    ResNetBiRealCIFAR,
    build_vision_model,
)

ROOT = Path(__file__).resolve().parents[1]


def test_resnet_bireal_cifar_forward():
    set_approx_sign(False)
    m = ResNetBiRealCIFAR(num_classes=10, width=8)
    y = m(torch.randn(2, 3, 32, 32))
    assert y.shape == (2, 10)
    assert torch.isfinite(y).all()


def test_resnet_bireal18_imagenet_stem_forward():
    set_approx_sign(False)
    m = ResNetBiReal18(num_classes=4, width=8)
    # Smaller than 224 keeps CI fast; stem still uses 7×7 + max-pool path.
    y = m(torch.randn(1, 3, 64, 64))
    assert y.shape == (1, 4)


def test_build_vision_model_resnet():
    m = build_vision_model("resnet_bireal_cifar", channels=16, width=8)
    assert isinstance(m, ResNetBiReal)
    m2 = build_vision_model("resnet_bireal18", width=8, num_classes=5)
    assert m2.head.out_features == 5
    assert m2.cifar is False


def test_resnet_bireal_one_train_step_within_sane_loss():
    """1-step STE smoke — proves trainability; floors remain image_cifar CNN."""
    set_approx_sign(True)
    try:
        g = torch.Generator().manual_seed(0)
        x = torch.randn(8, 3, 32, 32, generator=g)
        y = torch.randint(0, 10, (8,), generator=g)
        model = ResNetBiRealCIFAR(num_classes=10, width=8)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()
        model.train()
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        clip_weights_(model)
        assert torch.isfinite(loss)
        assert float(loss.detach()) < 50.0  # sanity, not a golden
    finally:
        set_approx_sign(False)


def test_imagenet_dataset_contract_schema():
    c = IMAGENET_DATASET_CONTRACT
    assert c["schema"] == "bnn_imagenet_folder_contract_v1"
    assert c["pass_gates"]["sota_top1"] is False
    assert c["pass_gates"]["invented_goldens"] is False
    assert c["proxy_minimum"]["num_classes"] >= 2


def test_make_proxy_imagenet_and_check(tmp_path: Path):
    report = make_proxy_imagenet(tmp_path / "proxy", n_classes=3, images_per_class=1)
    assert report["ok"] is True
    assert report["train_classes"] == 3
    assert report["train_images"] == 3
    assert report["meets_proxy_minimum"] is True
    again = check_imagenet_folder(tmp_path / "proxy", require_images=True)
    assert again["ok"] is True


def test_check_imagenet_require_images_fails_without_files(tmp_path: Path):
    for split in ("train", "val"):
        (tmp_path / split / "cat").mkdir(parents=True)
        (tmp_path / split / "dog").mkdir(parents=True)
    assert check_imagenet_folder(tmp_path, require_images=False)["ok"] is True
    assert check_imagenet_folder(tmp_path, require_images=True)["ok"] is False


def test_write_dataset_contract(tmp_path: Path):
    path = write_dataset_contract(tmp_path / "contract.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["schema"] == "bnn_imagenet_folder_contract_v1"


def test_imagenet_protocol_script_smoke(tmp_path: Path):
    out = tmp_path / "smoke.json"
    proxy = tmp_path / "proxy"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "imagenet_protocol.py"),
            "--mode",
            "smoke",
            "--root",
            str(proxy),
            "--out",
            str(out),
            "--n-classes",
            "3",
            "--image-size",
            "32",
            "--width",
            "8",
            "--batch-size",
            "2",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["sota_gate"] is False
    assert payload["smoke"]["logits_finite"] is True
    assert payload["folder"]["ok"] is True
