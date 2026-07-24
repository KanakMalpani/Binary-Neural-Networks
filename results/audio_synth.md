# Audio — synthetic tone spectrograms

- Classes: 8 | train: 800 | epochs: 5
- FP32 CNN: **94.50%**
- Binary CNN: **96.00%**
- Gap: **-1.50 pp**
- Binary audio-CNN within -1.50 pp of FP on synthetic tone spectrograms. Classic BNN is NOT production ASR — use INT8 Whisper/ORT for real speech; this demo proves STE + packing pattern on audio features.
