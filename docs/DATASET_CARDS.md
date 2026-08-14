# Dataset cards (lab)

**Task:** W6.T04 (MNIST / CIFAR / synth audio) + W6.T05 (seq reverse). Datasets are **not** committed under `data/` (gitignored).

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

W6.T05. Not a downloaded corpus — generated in-process. Do **not** commit sequences under `data/`.

| | |
|--|--|
| **Use** | Encoder–decoder STE pedagogy (`bnn train-seq2seq`) |
| **Generator** | On-the-fly reverse sequences (`bnn.seq.make_reverse_batch`) |
| **Split** | Synthetic batches each run; no train/test files on disk |
| **License** | N/A (generated; not a third-party dataset) |
| **Card honesty** | Smoke: eval token acc = **1.0** at 80 STE steps (`results/seq2seq_encoder_decoder.json`). Not a real NLP / translation benchmark. Attention and LayerNorm stay FP; FFN is binary or ternary. |
| **Docs** | [`36_ENCODER_DECODER_AND_NEXT.md`](36_ENCODER_DECODER_AND_NEXT.md), [tutorial 06](tutorials/06_encoder_decoder.md), [`api/seq.md`](api/seq.md) |
