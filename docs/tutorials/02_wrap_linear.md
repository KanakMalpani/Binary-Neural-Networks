# Tutorial 02 — Wrap existing Linears

## Goal

Replace middle `nn.Linear` layers with packed XNOR inference modules.

```bat
bnn wrap --mode binary_xnor --hidden 4096 --batch 32
```

Or in Python:

```python
from bnn import wrap_model
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(784, 512),  # treat carefully — often keep stem FP via skip list
    nn.ReLU(),
    nn.Linear(512, 512),
    nn.ReLU(),
    nn.Linear(512, 10),
)
model, report = wrap_model(model, policy="default", min_in_features=64)
print(report.replaced, report.compression)
```

## Honest expectation

Without QAT, cosine vs FP can be poor. Size compresses ~32×; speed needs native kernel + wide layers.
See `docs/12_WRAPPER_AND_EXISTING_MODELS.md`.
