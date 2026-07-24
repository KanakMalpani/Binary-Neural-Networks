"""Audio / speech feature demos for extreme low-bit inference pedagogy."""

from .data import get_audio_loaders, synthesize_tone
from .features import waveform_to_features
from .models import build_audio_model

__all__ = [
    "get_audio_loaders",
    "synthesize_tone",
    "waveform_to_features",
    "build_audio_model",
]
