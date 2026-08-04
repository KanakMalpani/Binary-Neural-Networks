"""Vision package: CIFAR / ResNet-BiReal models + ImageNet folder protocol."""

from __future__ import annotations

from .imagenet_protocol import (
    IMAGENET_DATASET_CONTRACT,
    check_imagenet_folder,
    dataset_contract_summary,
    describe_imagenet_folder_layout,
    make_proxy_imagenet,
    write_dataset_contract,
)
from .models import (
    BiRealBasicBlock,
    BinaryCIFARCNN,
    FP32CIFARCNN,
    ResNetBiReal,
    ResNetBiReal18,
    ResNetBiRealCIFAR,
    TinyBinaryViT,
    build_vision_model,
)

__all__ = [
    "FP32CIFARCNN",
    "BinaryCIFARCNN",
    "BiRealBasicBlock",
    "ResNetBiReal",
    "ResNetBiRealCIFAR",
    "ResNetBiReal18",
    "TinyBinaryViT",
    "build_vision_model",
    "IMAGENET_DATASET_CONTRACT",
    "check_imagenet_folder",
    "describe_imagenet_folder_layout",
    "make_proxy_imagenet",
    "write_dataset_contract",
    "dataset_contract_summary",
]
