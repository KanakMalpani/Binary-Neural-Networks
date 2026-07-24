# Dataset cards (lab)

**Task:** W6.T04. Datasets are **not** committed under `data/` (gitignored).

## MNIST

| | |
|--|--|
| **Use** | STE MLP pedagogy (`bnn train`) |
| **Loader** | `bnn` MNIST helper (no torchvision required) |
| **Split** | Standard train/test |
| **License** | Yann LeCun MNIST terms (public research use) |
| **Card honesty** | Accuracy floors in `tests/golden_floors.json` — not SOTA chase |

## CIFAR-10

| | |
|--|--|
| **Use** | Bi-Real / image lane (`bnn train-image`, `train-cifar`) |
| **Loader** | HF datasets or local cache under `data/` (untracked) |
| **Card honesty** | Proxy schedule; ImageNet SOTA is a **non-goal** |

## Synthetic audio tones

| | |
|--|--|
| **Use** | Audio lane pedagogy (`bnn train-audio`) |
| **Generator** | `bnn.audio` synthetic tones |
| **Non-goal** | Production ASR / Whisper replacement |

## Seq reverse task

| | |
|--|--|
| **Use** | Encoder–decoder STE (`bnn train-seq2seq`) |
| **Data** | On-the-fly reverse sequences |
| **Docs** | `docs/36`, tutorial 06 |
