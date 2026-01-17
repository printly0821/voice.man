"""
Batch Size Optimizer for Voice Man Pipeline

Provides dynamic batch size calculation based on:
- Available system RAM
- GPU memory (if available)
- File size characteristics

Memory Optimization (2026-01):
- Dynamic batch size based on RAM (16GB → 1, 32GB → 2-4, 64GB+ → 4-8)
- GPU memory-aware batch sizing
"""

import logging
from typing import Optional, Tuple
import psutil

logger = logging.getLogger(__name__)


class BatchSizeOptimizer:
    """
    Dynamic batch size optimizer based on available memory.

    Automatically calculates optimal batch size considering:
    - System RAM availability
    - GPU memory availability (if GPU present)
    - Memory per file estimation

    Features:
    - get_optimal_batch_size(): Calculate optimal batch size
    - get_memory_info(): Get current memory status
    - should_reduce_batch_size(): Check if batch should be reduced
    """

    # RAM-based batch size thresholds (in GB)
    RAM_THRESHOLDS = {
        "min_ram_gb": 8,      # Minimum RAM for batch processing
        "small_ram_gb": 16,   # Small RAM: batch_size = 1
        "medium_ram_gb": 32,  # Medium RAM: batch_size = 2-4
        "large_ram_gb": 64,   # Large RAM: batch_size = 4-8
    }

    # GPU memory thresholds
    GPU_MEMORY_THRESHOLD_PERCENT = 70.0  # Fallback to CPU if GPU > 70%

    # Memory per file estimation (in MB)
    DEFAULT_MEMORY_PER_FILE_MB = 512  # Conservative estimate

    def __init__(
        self,
        min_batch_size: int = 1,
        max_batch_size: int = 8,
        memory_per_file_mb: int = DEFAULT_MEMORY_PER_FILE_MB,
        safety_margin: float = 1.3,
    ):
        """
        Initialize BatchSizeOptimizer.

        Args:
            min_batch_size: Minimum batch size (default: 1)
            max_batch_size: Maximum batch size (default: 8)
            memory_per_file_mb: Estimated memory per file in MB
            safety_margin: Safety margin multiplier (default: 1.3 = 30% buffer)
        """
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.memory_per_file_mb = memory_per_file_mb
        self.safety_margin = safety_margin

        # Check GPU availability
        self._gpu_available = self._check_gpu_available()

        logger.info(
            f"BatchSizeOptimizer initialized: "
            f"min_batch={min_batch_size}, max_batch={max_batch_size}, "
            f"memory_per_file={memory_per_file_mb}MB, "
            f"gpu_available={self._gpu_available}"
        )

    def _check_gpu_available(self) -> bool:
        """Check if GPU is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
        except Exception:
            return False

    def _get_gpu_memory_percent(self) -> Optional[float]:
        """Get GPU memory usage percentage."""
        if not self._gpu_available:
            return None

        try:
            import torch
            reserved = torch.cuda.memory_reserved(0)
            total = torch.cuda.get_device_properties(0).total_memory
            return (reserved / total) * 100 if total > 0 else None
        except Exception:
            return None

    def get_memory_info(self) -> dict:
        """
        Get current memory information.

        Returns:
            Dictionary with memory stats:
            - total_ram_gb: Total system RAM in GB
            - available_ram_gb: Available RAM in GB
            - used_ram_percent: Used RAM percentage
            - gpu_memory_percent: GPU memory usage (if available)
            - recommended_batch_size: Calculated batch size
        """
        # System memory
        mem = psutil.virtual_memory()
        total_ram_gb = mem.total / (1024**3)
        available_ram_gb = mem.available / (1024**3)
        used_ram_percent = mem.percent

        info = {
            "total_ram_gb": round(total_ram_gb, 2),
            "available_ram_gb": round(available_ram_gb, 2),
            "used_ram_percent": round(used_ram_percent, 2),
        }

        # GPU memory
        gpu_memory_percent = self._get_gpu_memory_percent()
        if gpu_memory_percent is not None:
            info["gpu_memory_percent"] = round(gpu_memory_percent, 2)

        # Recommended batch size
        info["recommended_batch_size"] = self.get_optimal_batch_size()

        return info

    def get_optimal_batch_size(
        self,
        file_size_mb: Optional[float] = None,
    ) -> int:
        """
        Calculate optimal batch size based on available memory.

        Args:
            file_size_mb: Optional file size for more accurate calculation

        Returns:
            Optimal batch size within min/max bounds

        Logic:
        - 16GB RAM: batch_size = 1
        - 32GB RAM: batch_size = 2-4
        - 64GB+ RAM: batch_size = 4-8
        """
        # Get available memory
        mem = psutil.virtual_memory()
        available_ram_gb = mem.available / (1024**3)

        # Base calculation on total RAM
        total_ram_gb = mem.total / (1024**3)

        if total_ram_gb < self.RAM_THRESHOLDS["small_ram_gb"]:
            # Small RAM (< 16GB): batch_size = 1
            batch_size = 1
            reason = f"Small RAM ({total_ram_gb:.1f}GB), using batch_size=1"

        elif total_ram_gb < self.RAM_THRESHOLDS["medium_ram_gb"]:
            # Medium RAM (16-32GB): batch_size = 2
            batch_size = 2
            reason = f"Medium RAM ({total_ram_gb:.1f}GB), using batch_size=2"

        elif total_ram_gb < self.RAM_THRESHOLDS["large_ram_gb"]:
            # Large RAM (32-64GB): batch_size = 4
            batch_size = 4
            reason = f"Large RAM ({total_ram_gb:.1f}GB), using batch_size=4"

        else:
            # Very large RAM (64GB+): batch_size = 8
            batch_size = 8
            reason = f"Very large RAM ({total_ram_gb:.1f}GB), using batch_size=8"

        # Adjust based on available memory and file size
        if file_size_mb:
            memory_per_file = file_size_mb * self.safety_margin
            max_from_memory = int((available_ram_gb * 1024) / memory_per_file)
            batch_size = min(batch_size, max_from_memory)
            reason += f", adjusted to {batch_size} based on file size ({file_size_mb:.1f}MB)"

        # Ensure within bounds
        batch_size = max(self.min_batch_size, min(batch_size, self.max_batch_size))

        logger.info(f"{reason}, final_batch_size={batch_size}")

        return batch_size

    def should_reduce_batch_size(
        self,
        current_batch_size: int,
        memory_threshold_percent: float = 85.0,
    ) -> Tuple[bool, int]:
        """
        Check if batch size should be reduced based on memory pressure.

        Args:
            current_batch_size: Current batch size
            memory_threshold_percent: Memory threshold for reduction (default: 85%)

        Returns:
            Tuple of (should_reduce, recommended_batch_size)
        """
        mem = psutil.virtual_memory()

        if mem.percent > memory_threshold_percent:
            # Reduce batch size by half
            recommended = max(current_batch_size // 2, self.min_batch_size)
            logger.warning(
                f"Memory pressure ({mem.percent:.1f}% > {memory_threshold_percent}%), "
                f"reducing batch size: {current_batch_size} -> {recommended}"
            )
            return True, recommended

        # Also check GPU memory
        gpu_percent = self._get_gpu_memory_percent()
        if gpu_percent is not None and gpu_percent > self.GPU_MEMORY_THRESHOLD_PERCENT:
            # Reduce batch size by half
            recommended = max(current_batch_size // 2, self.min_batch_size)
            logger.warning(
                f"GPU memory pressure ({gpu_percent:.1f}% > {self.GPU_MEMORY_THRESHOLD_PERCENT}%), "
                f"reducing batch size: {current_batch_size} -> {recommended}"
            )
            return True, recommended

        return False, current_batch_size

    def can_process_batch(
        self,
        batch_size: int,
        file_size_mb: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """
        Check if batch can be processed safely.

        Args:
            batch_size: Proposed batch size
            file_size_mb: Optional file size for accurate calculation

        Returns:
            Tuple of (can_process, message)
        """
        mem = psutil.virtual_memory()
        available_ram_mb = mem.available / (1024**2)

        # Estimate memory needed
        memory_per_file = file_size_mb if file_size_mb else self.memory_per_file_mb
        estimated_memory_mb = batch_size * memory_per_file * self.safety_margin

        if estimated_memory_mb > available_ram_mb:
            return (
                False,
                f"Insufficient memory: need {estimated_memory_mb:.0f}MB, "
                f"available {available_ram_mb:.0f}MB. Reduce batch size."
            )

        predicted_percent = ((mem.used + estimated_memory_mb) / mem.total) * 100
        if predicted_percent > 90:
            return (
                True,
                f"WARNING: High memory usage predicted ({predicted_percent:.1f}%). "
                f"Proceed with caution."
            )

        return (
            True,
            f"Memory OK: predicted {predicted_percent:.1f}% usage. "
            f"Safe to process batch of {batch_size} files."
        )
