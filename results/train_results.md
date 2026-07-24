# Training results

| Model | Test acc % | Train s | Sim img/s | Binary-ish params |
|-------|------------|---------|-----------|-------------------|
| fp32_mlp | 97.67 | 8.4 | 206629 | 0 |
| binary_mlp | 96.36 | 14.2 | 107762 | 524288 |
| ternary_mlp | 97.16 | 13.5 | 126364 | 524288 |
| fp32_cnn | 96.61 | 115.5 | 6500 | 0 |
| binary_cnn | 94.79 | 238.7 | 2285 | 73728 |
