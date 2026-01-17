"""
GPU monitoring service for parallel audio processing.

Provides GPU availability detection, memory monitoring, and device selection
with automatic fallback to CPU based on EARS requirements (SPEC-PARALLEL-001).

EARS Requirements Implemented:
- E1: GPU availability check at batch start with CPU fallback
- E2: Auto-reduce batch size by 50% on memory shortage
- U2: GPU memory real-time monitoring (80% warning, 95% auto-adjust)
- S1: CPU fallback when GPU memory is critical
"""

import logging
from typing import Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GPUMemoryThresholds:
    """Thresholds for GPU memory monitoring."""

    warning_percent: float = 80.0  # U2: 80% warning threshold
    critical_percent: float = 95.0  # U2: 95% auto-adjust threshold
    fallback_percent: float = 98.0  # S1: CPU fallback threshold


class GPUMonitorService:
    """
    GPU monitoring service with memory tracking and device selection.

    Monitors GPU availability and memory usage, provides recommendations
    for batch size adjustment, and handles CPU fallback when needed.
    """

    def __init__(
        self,
        device_index: int = 0,
        min_batch_size: int = 2,
        thresholds: Optional[GPUMemoryThresholds] = None,
    ):
        """
        Initialize GPU monitor service.

        Args:
            device_index: CUDA device index to monitor (default: 0)
            min_batch_size: Minimum batch size to maintain (default: 2)
            thresholds: Memory threshold configuration
        """
        self.device_index = device_index
        self.min_batch_size = min_batch_size
        self.thresholds = thresholds or GPUMemoryThresholds()
        self._nvml_initialized = False

    def _ensure_nvml_init(self) -> bool:
        """
        Initialize NVML if not already initialized.

        Returns:
            True if NVML is available and initialized, False otherwise
        """
        if self._nvml_initialized:
            return True

        if not self.is_gpu_available():
            return False

        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml_initialized = True
            logger.debug("NVML initialized successfully")
            return True
        except Exception as e:
            logger.warning(f"Failed to initialize NVML: {e}")
            return False

    def is_gpu_available(self) -> bool:
        """
        Check if GPU (CUDA) is available.

        Returns:
            True if CUDA is available, False otherwise

        Implements:
            E1: GPU availability check at batch start
        """
        try:
            import torch

            available = torch.cuda.is_available()
            logger.debug(f"GPU availability check: {available}")
            return available
        except ImportError:
            logger.warning("PyTorch not installed, GPU unavailable")
            return False
        except Exception as e:
            logger.warning(f"Error checking GPU availability: {e}")
            return False

    def get_recommended_device(self) -> str:
        """
        Get recommended device based on GPU availability and memory status.

        Returns:
            "cuda" if GPU is available and has sufficient memory, "cpu" otherwise

        Implements:
            E1: CPU fallback when GPU unavailable
            S1: CPU fallback when GPU memory is critical
        """
        if not self.is_gpu_available():
            logger.info("GPU unavailable, recommending CPU")
            return "cpu"

        # Check memory status for critical condition
        memory_status = self.check_memory_status()
        if memory_status.get("critical", False):
            usage = memory_status.get("usage_percentage", 0)
            if usage >= self.thresholds.fallback_percent:
                logger.warning(f"GPU memory critical ({usage:.1f}%), falling back to CPU")
                return "cpu"

        return "cuda"

    def _get_gpu_memory_stats_torch(self) -> Dict:
        """
        Get GPU memory statistics using PyTorch native functions.

        This is used as a fallback when NVML is not available or fails.
        PyTorch's memory functions work on unified memory GPUs like GB10.

        Returns:
            Dictionary with memory statistics:
            - total_mb: Total GPU memory in MB
            - used_mb: Used GPU memory in MB (allocated)
            - reserved_mb: Reserved GPU memory in MB
            - free_mb: Free GPU memory in MB
            - usage_percentage: Memory usage percentage
            - available: True if GPU is available
            - source: "torch" to indicate PyTorch native
        """
        if not self.is_gpu_available():
            return {
                "total_mb": 0,
                "used_mb": 0,
                "reserved_mb": 0,
                "free_mb": 0,
                "usage_percentage": 0.0,
                "available": False,
                "source": "torch",
            }

        try:
            import torch

            # Get memory info (free, total) in bytes
            free_bytes, total_bytes = torch.cuda.mem_get_info(self.device_index)

            # Get allocated and reserved memory in bytes
            allocated_bytes = torch.cuda.memory_allocated(self.device_index)
            reserved_bytes = torch.cuda.memory_reserved(self.device_index)

            # Convert to MB
            total_mb = total_bytes / (1024 * 1024)
            allocated_mb = allocated_bytes / (1024 * 1024)
            reserved_mb = reserved_bytes / (1024 * 1024)
            free_mb = free_bytes / (1024 * 1024)

            # Use reserved memory for usage percentage (more conservative)
            # On unified memory GPUs, reserved represents actual GPU memory commitment
            usage_percentage = (reserved_bytes / total_bytes) * 100 if total_bytes > 0 else 0.0

            stats = {
                "total_mb": total_mb,
                "used_mb": allocated_mb,
                "reserved_mb": reserved_mb,
                "free_mb": free_mb,
                "usage_percentage": usage_percentage,
                "available": True,
                "source": "torch",
            }

            logger.debug(
                f"GPU memory stats (PyTorch): {allocated_mb:.0f}MB allocated, "
                f"{reserved_mb:.0f}MB reserved / {total_mb:.0f}MB ({usage_percentage:.1f}%)"
            )
            return stats

        except Exception as e:
            logger.warning(f"Failed to get GPU memory stats via PyTorch: {e}")
            return {
                "total_mb": 0,
                "used_mb": 0,
                "reserved_mb": 0,
                "free_mb": 0,
                "usage_percentage": 0.0,
                "available": False,
                "source": "torch",
            }

    def get_gpu_memory_stats(self) -> Dict:
        """
        Get GPU memory statistics.

        Tries NVML first for accurate metrics, falls back to PyTorch native
        functions if NVML is unavailable or fails (e.g., on unified memory
        GPUs like NVIDIA GB10).

        Returns:
            Dictionary with memory statistics:
            - total_mb: Total GPU memory in MB
            - used_mb: Used GPU memory in MB
            - reserved_mb: Reserved GPU memory in MB (PyTorch only)
            - free_mb: Free GPU memory in MB
            - usage_percentage: Memory usage percentage
            - source: "nvml" or "torch" indicating data source

        Implements:
            U2: GPU memory real-time monitoring
        """
        # Try NVML first (more accurate for discrete GPUs)
        if self._ensure_nvml_init():
            try:
                import pynvml

                handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

                total_mb = mem_info.total / (1024 * 1024)
                used_mb = mem_info.used / (1024 * 1024)
                free_mb = (mem_info.total - mem_info.used) / (1024 * 1024)
                usage_percentage = (mem_info.used / mem_info.total) * 100

                stats = {
                    "total_mb": total_mb,
                    "used_mb": used_mb,
                    "reserved_mb": used_mb,  # Same as used for NVML
                    "free_mb": free_mb,
                    "usage_percentage": usage_percentage,
                    "available": True,
                    "source": "nvml",
                }

                logger.debug(
                    f"GPU memory stats (NVML): {used_mb:.0f}MB / {total_mb:.0f}MB ({usage_percentage:.1f}%)"
                )
                return stats

            except Exception as e:
                logger.warning(f"NVML memory query failed: {e}, falling back to PyTorch")
                # Fall through to PyTorch fallback

        # Fallback to PyTorch native functions (works on GB10 and unified memory GPUs)
        return self._get_gpu_memory_stats_torch()

    def check_memory_status(self) -> Dict:
        """
        Check GPU memory status and return warnings/recommendations.

        Returns:
            Dictionary with status information:
            - warning: True if memory usage exceeds warning threshold (80%)
            - critical: True if memory usage exceeds critical threshold (95%)
            - auto_adjust_recommended: True if batch size should be reduced
            - usage_percentage: Current memory usage percentage
            - message: Human-readable status message

        Implements:
            U2: 80% warning, 95% auto-adjust thresholds
        """
        stats = self.get_gpu_memory_stats()

        if not stats.get("available", False):
            return {
                "warning": False,
                "critical": False,
                "auto_adjust_recommended": False,
                "usage_percentage": 0.0,
                "message": "GPU not available",
            }

        usage = stats["usage_percentage"]
        warning = usage >= self.thresholds.warning_percent
        critical = usage >= self.thresholds.critical_percent

        if critical:
            message = f"Critical: GPU memory at {usage:.1f}% (threshold: {self.thresholds.critical_percent}%)"
        elif warning:
            message = f"Warning: GPU memory at {usage:.1f}% exceeds 80% threshold"
        else:
            message = f"GPU memory usage normal: {usage:.1f}%"

        status = {
            "warning": warning,
            "critical": critical,
            "auto_adjust_recommended": critical,
            "usage_percentage": usage,
            "message": message,
        }

        if warning:
            logger.warning(message)
        else:
            logger.debug(message)

        return status

    def get_recommended_batch_size(self, current_batch_size: int) -> int:
        """
        Get recommended batch size based on GPU memory status.

        Args:
            current_batch_size: Current batch size

        Returns:
            Recommended batch size (reduced by 50% if memory critical)

        Implements:
            E2: Auto-reduce batch size by 50% on memory shortage
        """
        status = self.check_memory_status()

        if status.get("auto_adjust_recommended", False):
            # Reduce batch size by 50%
            new_size = max(current_batch_size // 2, self.min_batch_size)
            logger.info(f"Recommending batch size reduction: {current_batch_size} -> {new_size}")
            return new_size

        return current_batch_size

    def clear_gpu_cache(self) -> None:
        """
        Clear GPU memory cache.

        Calls torch.cuda.empty_cache() to free unused GPU memory.
        No-op if GPU is not available.
        """
        if not self.is_gpu_available():
            logger.debug("GPU not available, skipping cache clear")
            return

        try:
            import torch

            torch.cuda.empty_cache()
            logger.info("GPU cache cleared")
        except Exception as e:
            logger.warning(f"Failed to clear GPU cache: {e}")

    def get_device_info(self) -> Dict:
        """
        Get detailed GPU device information.

        Returns:
            Dictionary with device information
        """
        if not self.is_gpu_available():
            return {"available": False, "name": "N/A", "compute_capability": "N/A"}

        try:
            import torch

            device_name = torch.cuda.get_device_name(self.device_index)
            capability = torch.cuda.get_device_capability(self.device_index)

            return {
                "available": True,
                "name": device_name,
                "compute_capability": f"{capability[0]}.{capability[1]}",
                "device_index": self.device_index,
            }
        except Exception as e:
            logger.warning(f"Failed to get device info: {e}")
            return {"available": False, "name": "Unknown", "compute_capability": "N/A"}

    def __del__(self):
        """Cleanup NVML on destruction."""
        if self._nvml_initialized:
            try:
                import pynvml

                pynvml.nvmlShutdown()
                logger.debug("NVML shutdown complete")
            except Exception:
                pass
