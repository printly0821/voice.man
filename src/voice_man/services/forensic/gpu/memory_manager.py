"""
GPU Memory Manager
SPEC-GPUAUDIO-001: Dynamic GPU memory management for OOM prevention

This module provides GPU memory monitoring and dynamic batch size adjustment
to prevent out-of-memory errors during audio processing.
"""

from typing import Optional
import logging

logger = logging.getLogger(__name__)


class GPUMemoryManager:
    """
    GPU memory manager for dynamic batch size adjustment.

    This is a skeleton implementation for Phase 2.
    Current phase focuses on F0 extraction only.
    """

    DEFAULT_MIN_BATCH_SIZE = 128
    DEFAULT_MAX_BATCH_SIZE = 4096
    DEFAULT_MEMORY_THRESHOLD_GB = 2.0

    def __init__(
        self,
        min_batch_size: int = DEFAULT_MIN_BATCH_SIZE,
        max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
        memory_threshold_gb: float = DEFAULT_MEMORY_THRESHOLD_GB,
    ):
        """
        Initialize the GPU memory manager.

        Args:
            min_batch_size: Minimum batch size for GPU processing
            max_batch_size: Maximum batch size for GPU processing
            memory_threshold_gb: GPU memory threshold for batch size reduction
        """
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.memory_threshold_gb = memory_threshold_gb
        self._current_batch_size = max_batch_size

    @property
    def current_batch_size(self) -> int:
        """Get current recommended batch size."""
        return self._current_batch_size

    def get_available_memory_gb(self) -> float:
        """
        Get available GPU memory in GB.

        Returns:
            Available GPU memory in gigabytes, or 0.0 if no GPU available.
        """
        try:
            import torch

            if torch.cuda.is_available():
                free_memory = torch.cuda.get_device_properties(0).total_memory
                allocated = torch.cuda.memory_allocated(0)
                return (free_memory - allocated) / (1024**3)
        except Exception as e:
            logger.warning(f"Failed to get GPU memory: {e}")
        return 0.0

    def adjust_batch_size(self) -> int:
        """
        Adjust batch size based on available GPU memory.

        Returns:
            Adjusted batch size.
        """
        available_gb = self.get_available_memory_gb()

        if available_gb < self.memory_threshold_gb:
            # Reduce batch size by 50%
            self._current_batch_size = max(self.min_batch_size, self._current_batch_size // 2)
            logger.info(
                f"Reduced batch size to {self._current_batch_size} "
                f"due to low GPU memory ({available_gb:.2f} GB)"
            )

        return self._current_batch_size

    def reset_batch_size(self) -> None:
        """Reset batch size to maximum."""
        self._current_batch_size = self.max_batch_size
