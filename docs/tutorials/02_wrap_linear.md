# Tutorial 02 — Wrap existing Linears

## Goal

Replace FFN/MLP `nn.Linear` layers with packed XNOR or ternary weight-only modules.

```bat
:: Legacy wide-MLP microbench
bnn wrap --mode binary_xnor --hidden 4096 --batch 32

:: Ultra wrap (hybrid policy + calib + effectiveness gate)
bnn wrap --ultra --policy auto --qat-steps 40 --batch 32 --force
```

Or in Python:

```python
from bnn import wrap_model
from bnn.wrapper import CalibConfig, recommend_wrap_policy
import torch.nn as nn

class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(784, 256)
        self.ffn_fc1 = nn.Linear(256, 1024)
        self.ffn_fc2 = nn.Linear(1024, 256)
        self.lm_head = nn.Linear(256, 10)

    def forward(self, x):
        h = self.embed(x)
        return self.lm_head(self.ffn_fc2(self.ffn_fc1(h)))

model = Tiny()
print(recommend_wrap_policy(model.ffn_fc1))
model, report = wrap_model(
    model,
    policy="hybrid_ffn",          # skip embed/attn/lm_head
    mode="ternary_weight_only",   # accuracy-first when needed
    calib=CalibConfig(per_channel=True),
    min_in_features=64,
)
print(report.replaced, report.compression, report.policy_reason)
```

## Honest expectation

- **Binary aggressive PTQ** cosine can be ~0.28 — not drop-in.
- **Hybrid FFN + calib** improves binary cosine (~0.7 on demo); **ternary/auto** aims ≥0.85.
- Size compresses ~32× (binary) / ~16× (ternary pack); speed needs native kernel + wide layers.
- See `docs/12_WRAPPER_AND_EXISTING_MODELS.md` and `docs/33_ULTRA_WRAP_LAYER.md`.
