"""
TorchCrepe F0 Extractor
SPEC-GPUAUDIO-001 TAG-002: GPU-accelerated F0 extraction using torchcrepe

This module provides GPU-accelerated fundamental frequency (F0) extraction
using the CREPE (Convolutional Representation for Pitch Estimation) model.
"""

from typing import Tuple, Optional, Literal
import logging

import numpy as np

logger = logging.getLogger(__name__)

ModelType = Literal["full", "tiny"]


class TorchCrepeExtractor:
    """
    GPU-accelerated F0 extractor using torchcrepe.

    This extractor provides batch processing capability for efficient
    F0 extraction from multiple audio windows simultaneously.

    Attributes:
        model: CREPE model variant ("full" or "tiny")
        device: Device for computation ("cuda" or "cpu")
        batch_size: Batch size for GPU processing
    """

    DEFAULT_BATCH_SIZE = 2048
    DEFAULT_HOP_LENGTH = 512
    MIN_AUDIO_SAMPLES = 512

    def __init__(
        self,
        model: ModelType = "full",
        device: Optional[str] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        """
        Initialize the TorchCrepe F0 extractor.

        Args:
            model: CREPE model variant ("full" for accuracy, "tiny" for speed)
            device: Device for computation ("cuda", "cpu", or None for auto-detect)
            batch_size: Batch size for GPU processing
        """
        self.model = model
        self.batch_size = batch_size
        self._torch = None
        self._torchcrepe = None
        self._device = None
        self._device_override = device

    @property
    def torch(self):
        """Lazy load torch."""
        if self._torch is None:
            import torch

            self._torch = torch
        return self._torch

    @property
    def torchcrepe(self):
        """Lazy load torchcrepe."""
        if self._torchcrepe is None:
            import torchcrepe

            self._torchcrepe = torchcrepe
        return self._torchcrepe

    @property
    def device(self) -> str:
        """
        Get the computation device.

        Returns:
            Device string ("cuda" or "cpu")
        """
        if self._device is None:
            if self._device_override is not None:
                self._device = self._device_override
            elif self.torch.cuda.is_available():
                self._device = "cuda"
            else:
                self._device = "cpu"
        return self._device

    @property
    def is_gpu_available(self) -> bool:
        """Check if GPU is available for computation."""
        return self.torch.cuda.is_available()

    def _pad_audio(self, audio: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Pad audio to minimum required length.

        Args:
            audio: Audio signal as numpy array

        Returns:
            Tuple of (padded_audio, original_length)
        """
        original_length = len(audio)
        if original_length < self.MIN_AUDIO_SAMPLES:
            padding = self.MIN_AUDIO_SAMPLES - original_length
            audio = np.pad(audio, (0, padding), mode="constant", constant_values=0)
        return audio, original_length

    def extract_f0(
        self,
        audio: np.ndarray,
        sr: int,
        fmin: float = 50.0,
        fmax: float = 550.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract F0 from a single audio signal.

        Args:
            audio: Audio signal as numpy array (samples,)
            sr: Sample rate in Hz
            fmin: Minimum F0 in Hz
            fmax: Maximum F0 in Hz

        Returns:
            Tuple of (f0, confidence):
                - f0: F0 values in Hz (NaN for unvoiced frames)
                - confidence: Confidence scores (0-1)

        Raises:
            ValueError: If audio is empty
        """
        # Handle empty audio
        if len(audio) == 0:
            raise ValueError("Audio array is empty")

        # Pad short audio
        audio, original_length = self._pad_audio(audio)

        # Convert to tensor
        audio_tensor = self.torch.tensor(
            audio, dtype=self.torch.float32, device=self.device
        ).unsqueeze(0)  # Add batch dimension

        # Extract F0 using torchcrepe
        f0, periodicity = self.torchcrepe.predict(
            audio_tensor,
            sr,
            hop_length=self.DEFAULT_HOP_LENGTH,
            fmin=fmin,
            fmax=fmax,
            model=self.model,
            batch_size=self.batch_size,
            device=self.device,
            return_periodicity=True,
        )

        # Convert to numpy
        f0 = f0.squeeze(0).cpu().numpy()
        confidence = periodicity.squeeze(0).cpu().numpy()

        # Apply confidence threshold to mark unvoiced frames as NaN
        f0[confidence < 0.5] = np.nan

        return f0, confidence

    def extract_f0_batch(
        self,
        audio_windows: np.ndarray,
        sr: int,
        fmin: float = 50.0,
        fmax: float = 550.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract F0 from multiple audio windows in batch.

        This method processes multiple audio windows simultaneously
        for efficient GPU utilization.

        Args:
            audio_windows: Audio windows as numpy array (num_windows, window_samples)
            sr: Sample rate in Hz
            fmin: Minimum F0 in Hz
            fmax: Maximum F0 in Hz

        Returns:
            Tuple of (f0_values, confidence_values):
                - f0_values: Mean F0 for each window (num_windows,)
                - confidence_values: Mean confidence for each window (num_windows,)

        Raises:
            ValueError: If audio_windows is empty or has wrong dimensions
        """
        if len(audio_windows) == 0:
            raise ValueError("audio_windows is empty")

        if audio_windows.ndim != 2:
            raise ValueError(
                f"audio_windows must be 2D (num_windows, samples), got {audio_windows.ndim}D"
            )

        num_windows = audio_windows.shape[0]
        f0_values = np.zeros(num_windows)
        confidence_values = np.zeros(num_windows)

        # Process each window
        # Note: torchcrepe.predict handles batching internally
        for i in range(num_windows):
            window = audio_windows[i]

            # Skip empty or very short windows
            if len(window) < self.MIN_AUDIO_SAMPLES:
                # Pad short audio
                window, _ = self._pad_audio(window)

            try:
                f0, confidence = self.extract_f0(window, sr, fmin, fmax)

                # Get mean F0 for voiced frames
                valid_f0 = f0[~np.isnan(f0)]
                if len(valid_f0) > 0:
                    f0_values[i] = np.mean(valid_f0)
                    confidence_values[i] = np.mean(confidence[~np.isnan(f0)])
                else:
                    f0_values[i] = np.nan
                    confidence_values[i] = 0.0

            except Exception as e:
                logger.warning(f"Failed to extract F0 for window {i}: {e}")
                f0_values[i] = np.nan
                confidence_values[i] = 0.0

        return f0_values, confidence_values
