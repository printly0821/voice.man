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

    def _ensure_2d(self, arr: np.ndarray, expected_batch_size: int) -> np.ndarray:
        """
        Ensure array is 2D with shape (batch_size, frames).

        This method robustly handles various output shapes from torchcrepe
        without using squeeze(), which can fail on unexpected dimensions.

        Args:
            arr: Array to reshape (can be 1D, 2D, or 3D)
            expected_batch_size: Expected number of samples in batch

        Returns:
            2D array with shape (batch_size, frames)
        """
        if arr.ndim == 2:
            # Already 2D, verify batch dimension
            if arr.shape[0] == expected_batch_size:
                return arr
            elif arr.shape[1] == expected_batch_size:
                # Transposed shape, fix it
                return arr.T
            else:
                # Unexpected shape, reshape based on total elements
                return arr.reshape(expected_batch_size, -1)
        elif arr.ndim == 1:
            # 1D: single sample with single frame, or flattened
            if arr.size == expected_batch_size:
                # One frame per sample
                return arr.reshape(expected_batch_size, 1)
            else:
                # Single sample with multiple frames
                return arr.reshape(1, -1)
        elif arr.ndim == 3:
            # 3D: reshape to 2D by flattening extra dimensions
            # Common shapes: (batch, 1, frames) or (1, batch, frames)
            total_elements = arr.size
            num_frames = total_elements // expected_batch_size
            return arr.reshape(expected_batch_size, num_frames)
        else:
            # Unexpected dimensionality, attempt flat reshape
            return arr.reshape(expected_batch_size, -1)

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

        # Convert to numpy - use reshape instead of squeeze for robustness
        f0_np = f0.cpu().numpy()
        confidence_np = periodicity.cpu().numpy()

        # Ensure 1D output by removing batch dimension safely
        if f0_np.ndim > 1:
            f0_np = f0_np.reshape(-1)
            confidence_np = confidence_np.reshape(-1)

        # Apply confidence threshold to mark unvoiced frames as NaN
        f0_np[confidence_np < 0.5] = np.nan

        return f0_np, confidence_np

    def extract_f0_batch(
        self,
        audio_windows: np.ndarray,
        sr: int,
        fmin: float = 50.0,
        fmax: float = 550.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract F0 from multiple audio windows efficiently.

        This method concatenates all windows, extracts F0 once using torchcrepe,
        then splits the results back into windows. This is much more efficient
        than calling torchcrepe for each window separately.

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
        window_samples = audio_windows.shape[1]

        # Pad windows if needed
        if window_samples < self.MIN_AUDIO_SAMPLES:
            padding = self.MIN_AUDIO_SAMPLES - window_samples
            audio_windows = np.pad(
                audio_windows,
                ((0, 0), (0, padding)),
                mode="constant",
                constant_values=0,
            )
            window_samples = audio_windows.shape[1]

        # Concatenate all windows into a single audio stream
        # This allows torchcrepe to process everything in one GPU call
        concatenated_audio = audio_windows.reshape(-1)  # Flatten to 1D

        # Convert to tensor with shape (1, total_samples) as torchcrepe expects
        audio_tensor = self.torch.tensor(
            concatenated_audio, dtype=self.torch.float32, device=self.device
        ).unsqueeze(0)  # Add batch dimension

        try:
            # Extract F0 for entire concatenated audio in one call
            f0_frames, periodicity_frames = self.torchcrepe.predict(
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

            # Convert to numpy and flatten
            f0_np = f0_frames.cpu().numpy().reshape(-1)
            periodicity_np = periodicity_frames.cpu().numpy().reshape(-1)

            # Apply confidence threshold
            f0_np[periodicity_np < 0.5] = np.nan

            # Calculate frames per window
            frames_per_window = int(np.ceil(window_samples / self.DEFAULT_HOP_LENGTH))
            total_frames = len(f0_np)

            # Split F0 values back into windows and calculate mean for each
            f0_values = np.zeros(num_windows)
            confidence_values = np.zeros(num_windows)

            for i in range(num_windows):
                frame_start = i * frames_per_window
                frame_end = min(frame_start + frames_per_window, total_frames)

                if frame_end > frame_start:
                    window_f0 = f0_np[frame_start:frame_end]
                    window_conf = periodicity_np[frame_start:frame_end]

                    valid_f0 = window_f0[~np.isnan(window_f0)]
                    if len(valid_f0) > 0:
                        f0_values[i] = np.mean(valid_f0)
                        confidence_values[i] = np.mean(window_conf[~np.isnan(window_f0)])
                    else:
                        f0_values[i] = np.nan
                        confidence_values[i] = 0.0
                else:
                    f0_values[i] = np.nan
                    confidence_values[i] = 0.0

            return f0_values, confidence_values

        except Exception as e:
            logger.warning(f"Batch F0 extraction failed, falling back to sequential: {e}")
            # Fallback to sequential processing
            f0_values = np.zeros(num_windows)
            confidence_values = np.zeros(num_windows)

            for i in range(num_windows):
                try:
                    f0, confidence = self.extract_f0(audio_windows[i], sr, fmin, fmax)
                    valid_f0 = f0[~np.isnan(f0)]
                    if len(valid_f0) > 0:
                        f0_values[i] = np.mean(valid_f0)
                        confidence_values[i] = np.mean(confidence[~np.isnan(f0)])
                    else:
                        f0_values[i] = np.nan
                        confidence_values[i] = 0.0
                except Exception:
                    f0_values[i] = np.nan
                    confidence_values[i] = 0.0

            return f0_values, confidence_values
