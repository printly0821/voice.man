"""
GPU Audio Backend
SPEC-GPUAUDIO-001 TAG-003: Unified GPU audio processing backend

This module provides a unified interface for GPU-accelerated audio processing,
including F0 extraction and spectrogram generation with automatic CPU fallback.
"""

from typing import Tuple, Optional
import logging

import numpy as np

from voice_man.services.forensic.gpu.crepe_extractor import TorchCrepeExtractor
from voice_man.services.forensic.gpu.nnaudio_processor import NNAudioProcessor

logger = logging.getLogger(__name__)


class GPUAudioBackend:
    """
    Unified GPU audio processing backend.

    This backend provides a consistent interface for GPU-accelerated
    audio processing with automatic fallback to CPU when GPU is unavailable.

    Attributes:
        use_gpu: Whether to attempt GPU acceleration
        device: Current device ("cuda" or "cpu")
        is_gpu_available: Whether GPU is actually available
    """

    def __init__(self, use_gpu: bool = True):
        """
        Initialize the GPU audio backend.

        Args:
            use_gpu: Whether to attempt GPU acceleration (default: True)
        """
        self._use_gpu = use_gpu
        self._torch = None
        self._crepe_extractor: Optional[TorchCrepeExtractor] = None
        self._nnaudio_processor: Optional[NNAudioProcessor] = None

    @property
    def torch(self):
        """Lazy load torch."""
        if self._torch is None:
            import torch

            self._torch = torch
        return self._torch

    @property
    def device(self) -> str:
        """
        Get the current computation device.

        Returns:
            Device string ("cuda" or "cpu")
        """
        if self._use_gpu and self.torch.cuda.is_available():
            return "cuda"
        return "cpu"

    @property
    def is_gpu_available(self) -> bool:
        """Check if GPU is available for computation."""
        return self.torch.cuda.is_available()

    @property
    def crepe_extractor(self) -> TorchCrepeExtractor:
        """
        Get the CREPE F0 extractor (lazy initialization).

        Returns:
            TorchCrepeExtractor instance

        Note:
            Uses "tiny" model by default for optimal performance (70x+ realtime).
            Change to "full" for higher accuracy at cost of speed (27x realtime).
        """
        if self._crepe_extractor is None:
            self._crepe_extractor = TorchCrepeExtractor(
                model="tiny",  # Optimized for speed (70x realtime vs 27x with full)
                device=self.device,
            )
        return self._crepe_extractor

    def extract_f0(
        self,
        audio: np.ndarray,
        sr: int,
        fmin: float = 50.0,
        fmax: float = 550.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract F0 from audio using GPU-accelerated CREPE.

        Args:
            audio: Audio signal as numpy array
            sr: Sample rate in Hz
            fmin: Minimum F0 in Hz
            fmax: Maximum F0 in Hz

        Returns:
            Tuple of (f0, confidence):
                - f0: F0 values in Hz (NaN for unvoiced frames)
                - confidence: Confidence scores (0-1)
        """
        return self.crepe_extractor.extract_f0(audio, sr, fmin, fmax)

    def extract_f0_batch(
        self,
        audio_windows: np.ndarray,
        sr: int,
        fmin: float = 50.0,
        fmax: float = 550.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract F0 from multiple audio windows in batch.

        Args:
            audio_windows: Audio windows as numpy array (num_windows, window_samples)
            sr: Sample rate in Hz
            fmin: Minimum F0 in Hz
            fmax: Maximum F0 in Hz

        Returns:
            Tuple of (f0_values, confidence_values):
                - f0_values: Mean F0 for each window (num_windows,)
                - confidence_values: Mean confidence for each window (num_windows,)
        """
        return self.crepe_extractor.extract_f0_batch(audio_windows, sr, fmin, fmax)

    @property
    def nnaudio_processor(self) -> NNAudioProcessor:
        """
        Get the NNAudio processor (lazy initialization).

        Returns:
            NNAudioProcessor instance configured for this backend
        """
        if self._nnaudio_processor is None:
            self._nnaudio_processor = NNAudioProcessor(
                device=self.device,
            )
        return self._nnaudio_processor

    def compute_stft(
        self,
        audio: np.ndarray,
        sr: int,
        n_fft: int = 2048,
        hop_length: int = 512,
        return_complex: bool = False,
    ) -> np.ndarray:
        """
        Compute STFT of audio signal using GPU acceleration.

        Args:
            audio: Audio signal as numpy array
            sr: Sample rate in Hz
            n_fft: FFT window size
            hop_length: Hop length in samples
            return_complex: If True, return complex STFT

        Returns:
            STFT result as numpy array (freq_bins, time_frames)
        """
        # Create processor with specified parameters if different from default
        if (
            self._nnaudio_processor is None
            or self._nnaudio_processor.n_fft != n_fft
            or self._nnaudio_processor.hop_length != hop_length
            or self._nnaudio_processor.sr != sr
        ):
            self._nnaudio_processor = NNAudioProcessor(
                n_fft=n_fft,
                hop_length=hop_length,
                sr=sr,
                device=self.device,
            )

        return self.nnaudio_processor.compute_stft(audio, return_complex=return_complex)

    def compute_mel_spectrogram(
        self,
        audio: np.ndarray,
        sr: int,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        power: float = 2.0,
        normalize: bool = False,
    ) -> np.ndarray:
        """
        Compute Mel spectrogram of audio signal using GPU acceleration.

        Args:
            audio: Audio signal as numpy array
            sr: Sample rate in Hz
            n_fft: FFT window size
            hop_length: Hop length in samples
            n_mels: Number of Mel frequency bins
            power: Exponent for the magnitude
            normalize: If True, normalize the spectrogram

        Returns:
            Mel spectrogram as numpy array (n_mels, time_frames)
        """
        # Create processor with specified parameters if different from default
        if (
            self._nnaudio_processor is None
            or self._nnaudio_processor.n_fft != n_fft
            or self._nnaudio_processor.hop_length != hop_length
            or self._nnaudio_processor.n_mels != n_mels
            or self._nnaudio_processor.sr != sr
        ):
            self._nnaudio_processor = NNAudioProcessor(
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
                sr=sr,
                device=self.device,
            )

        return self.nnaudio_processor.compute_mel_spectrogram(
            audio, power=power, normalize=normalize
        )
