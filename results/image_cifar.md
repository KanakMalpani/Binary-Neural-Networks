# Image — CIFAR-10 Bi-Real

- Subset: 30000 | epochs: 8 | approx_sign: True
- FP32 CNN: **71.14%** (163.0 ms/batch)
- Binary Bi-Real: **61.14%** (361.8 ms/batch)
- Gap: **10.00 pp**
- Bi-Real binary within 10.00 pp of FP CNN twin. Packed Linear kernels apply to ViT FFN / MLP heads; Conv path is STE-sim (BinaryConv wrap available for size — see wrapper).
