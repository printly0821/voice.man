"""
Smart Device Selector for WhisperX Pipeline

Provides intelligent device selection (GPU/CPU) based on:
- File size (small files → CPU for efficiency)
- GPU memory availability (GPU > 70% → CPU fallback)
- System constraints

Memory Optimization (2026-01):
- Smart CPU Fallback: GPU > 70% or small files → CPU
- Dynamic model selection based on file size
"""

import logging
from typing import Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class SmartDeviceSelector:
    """
    Smart device selector for WhisperX pipeline.

    Automatically chooses between GPU and CPU based on:
    - File size (small files processed faster on CPU)
    - GPU memory usage (fallback to CPU when GPU is busy)
    - GPU availability

    Features:
    - should_use_cpu(): Decide whether to use CPU
    - get_optimal_model(): Select model based on file size
    - get_device_config(): Get complete device configuration
    """

    # Thresholds for device selection
    SMALL_FILE_THRESHOLD_MB = 20.0  # Files < 20MB use CPU
    MEDIUM_FILE_THRESHOLD_MB = 50.0  # Files 20-50MB use base model
    GPU_MEMORY_THRESHOLD_PERCENT = 70.0  # Fallback to CPU if GPU > 70%

    # Model selection based on file size
    MODEL_MAPPING = {
        "tiny": (0, 10),      # 0-10MB: tiny model
        "base": (10, 50),     # 10-50MB: base model
        "distil-large-v3": (50, float("inf")),  # >50MB: distil-large-v3
    }

    def __init__(
        self,
        small_file_threshold_mb: float = SMALL_FILE_THRESHOLD_MB,
        gpu_memory_threshold_percent: float = GPU_MEMORY_THRESHOLD_PERCENT,
    ):
        """
        Initialize SmartDeviceSelector.

        Args:
            small_file_threshold_mb: File size threshold for CPU (MB)
            gpu_memory_threshold_percent: GPU memory threshold for CPU fallback (%)
        """
        self.small_file_threshold_mb = small_file_threshold_mb
        self.gpu_memory_threshold_percent = gpu_memory_threshold_percent

        # Check GPU availability
        self._gpu_available = self._check_gpu_available()

        logger.info(
            f"SmartDeviceSelector initialized: "
            f"gpu_available={self._gpu_available}, "
            f"small_file_threshold={small_file_threshold_mb}MB, "
            f"gpu_threshold={gpu_memory_threshold_percent}%"
        )

    def _check_gpu_available(self) -> bool:
        """Check if GPU is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
        except Exception as e:
            logger.warning(f"Error checking GPU availability: {e}")
            return False

    def _get_gpu_memory_percent(self) -> Optional[float]:
        """
        Get GPU memory usage percentage.

        Returns:
            Memory usage percentage, or None if unavailable
        """
        if not self._gpu_available:
            return None

        try:
            import torch

            allocated = torch.cuda.memory_allocated(0)
            reserved = torch.cuda.memory_reserved(0)
            total = torch.cuda.get_device_properties(0).total_memory

            # Use reserved memory (more conservative)
            usage_percent = (reserved / total) * 100 if total > 0 else 0
            return usage_percent

        except Exception as e:
            logger.warning(f"Failed to get GPU memory: {e}")
            return None

    def should_use_cpu(
        self,
        file_size_mb: float,
        gpu_memory_percent: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """
        Decide whether to use CPU based on file size and GPU memory.

        Args:
            file_size_mb: File size in megabytes
            gpu_memory_percent: Optional GPU memory percentage (auto-detected if None)

        Returns:
            Tuple of (should_use_cpu, reason)

        Decision Logic:
        - Use CPU if GPU is not available
        - Use CPU if file size < small_file_threshold_mb (20MB default)
        - Use CPU if GPU memory > gpu_memory_threshold_percent (70% default)
        - Use GPU otherwise
        """
        if not self._gpu_available:
            return True, "GPU not available, using CPU"

        # Check GPU memory if not provided
        if gpu_memory_percent is None:
            gpu_memory_percent = self._get_gpu_memory_percent()

        # Check GPU memory threshold
        if gpu_memory_percent is not None and gpu_memory_percent > self.gpu_memory_threshold_percent:
            return (
                True,
                f"GPU memory ({gpu_memory_percent:.1f}%) exceeds threshold "
                f"({self.gpu_memory_threshold_percent}%), using CPU"
            )

        # Check file size threshold
        if file_size_mb < self.small_file_threshold_mb:
            return (
                True,
                f"File size ({file_size_mb:.1f}MB) below threshold "
                f"({self.small_file_threshold_mb}MB), using CPU for efficiency"
            )

        return False, f"Using GPU (file_size={file_size_mb:.1f}MB, gpu_memory={gpu_memory_percent:.1f}%)"

    def get_optimal_model(self, file_size_mb: float) -> str:
        """
        Select optimal model based on file size.

        Args:
            file_size_mb: File size in megabytes

        Returns:
            Model size identifier

        Model Selection Logic:
        - tiny: 0-10MB files
        - base: 10-50MB files
        - distil-large-v3: >50MB files (default for quality)
        """
        for model, (min_mb, max_mb) in self.MODEL_MAPPING.items():
            if min_mb <= file_size_mb < max_mb:
                logger.info(f"Selected model '{model}' for file size {file_size_mb:.1f}MB")
                return model

        # Default to distil-large-v3 for large files
        logger.info(f"Using default model 'distil-large-v3' for file size {file_size_mb:.1f}MB")
        return "distil-large-v3"

    def get_device_config(
        self,
        file_path: str,
        gpu_memory_percent: Optional[float] = None,
    ) -> dict:
        """
        Get complete device configuration for a file.

        Args:
            file_path: Path to audio file
            gpu_memory_percent: Optional GPU memory percentage

        Returns:
            Dictionary with device configuration:
            - device: "cuda" or "cpu"
            - model_size: Optimal model size
            - compute_type: "int8" for memory efficiency
            - reason: Explanation of device choice
        """
        # Get file size
        file_size_mb = Path(file_path).stat().st_size / (1024 * 1024)

        # Determine device
        should_use_cpu, reason = self.should_use_cpu(file_size_mb, gpu_memory_percent)
        device = "cpu" if should_use_cpu else "cuda"

        # Get optimal model
        model_size = self.get_optimal_model(file_size_mb)

        # Use int8 for memory efficiency
        compute_type = "int8"

        config = {
            "device": device,
            "model_size": model_size,
            "compute_type": compute_type,
            "reason": reason,
            "file_size_mb": file_size_mb,
        }

        logger.info(
            f"Device config: device={device}, model={model_size}, "
            f"compute_type={compute_type}, file_size={file_size_mb:.1f}MB"
        )

        return config

    def get_recommended_device(self) -> str:
        """
        Get recommended device without file context.

        Returns:
            "cuda" if GPU is available and has sufficient memory, "cpu" otherwise
        """
        if not self._gpu_available:
            return "cpu"

        gpu_memory = self._get_gpu_memory_percent()
        if gpu_memory is not None and gpu_memory > self.gpu_memory_threshold_percent:
            return "cpu"

        return "cuda"
