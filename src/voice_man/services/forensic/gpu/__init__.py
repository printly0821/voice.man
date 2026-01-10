"""
GPU Audio Processing Module
SPEC-GPUAUDIO-001: GPU-accelerated audio feature extraction

This module provides GPU-accelerated audio processing capabilities
using torchcrepe for F0 extraction and nnAudio for spectrogram generation.
"""

from voice_man.services.forensic.gpu.backend import GPUAudioBackend
from voice_man.services.forensic.gpu.crepe_extractor import TorchCrepeExtractor

__all__ = [
    "GPUAudioBackend",
    "TorchCrepeExtractor",
]
