"""
Memory management service for audio file processing.

Provides garbage collection, memory monitoring, and resource cleanup.
Extended for GPU memory monitoring as per SPEC-PARALLEL-001.
Optimized for forensic analysis workloads as per SPEC-PERFOPT-001.

EARS Requirements Implemented:
- S2: GC trigger at 80% system memory usage
- U2: GPU memory real-time monitoring
- N2: Unlimited memory allocation prohibited (80% system, 95% GPU limits)
- SPEC-PERFOPT-001: 30GB forensic memory threshold for heavy ML model workloads
"""

import gc
import logging
import psutil
from typing import Optional, Dict
from dataclasses import dataclass, field
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

# SPEC-PERFOPT-001: Forensic analysis memory threshold (30GB)
# This higher threshold is optimized for forensic audio analysis workloads
# that require loading multiple ML models (WhisperX, SER, etc.) simultaneously.
FORENSIC_MEMORY_THRESHOLD_MB = 30000


@dataclass
class MemoryStats:
    """Memory usage statistics."""

    initial_mb: float = field(default_factory=lambda: MemoryStats._get_current_mb())
    current_mb: float = field(default_factory=lambda: MemoryStats._get_current_mb())
    peak_mb: float = field(default_factory=lambda: MemoryStats._get_current_mb())
    delta_mb: float = 0.0

    @staticmethod
    def _get_current_mb() -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024

    def update(self) -> None:
        """Update current memory usage and recalculate statistics."""
        self.current_mb = self._get_current_mb()
        self.peak_mb = max(self.peak_mb, self.current_mb)
        self.delta_mb = self.current_mb - self.initial_mb

    def to_dict(self) -> dict:
        """Convert stats to dictionary."""
        return {
            "initial_mb": round(self.initial_mb, 2),
            "current_mb": round(self.current_mb, 2),
            "peak_mb": round(self.peak_mb, 2),
            "delta_mb": round(self.delta_mb, 2),
        }


class MemoryManager:
    """Manage memory during batch processing.

    Extended for GPU memory monitoring as per SPEC-PARALLEL-001.
    """

    def __init__(
        self,
        threshold_mb: float = FORENSIC_MEMORY_THRESHOLD_MB,
        enable_gc: bool = True,
        enable_gpu_monitoring: bool = False,
        system_memory_threshold: float = 80.0,  # S2: 80% system memory threshold
        gpu_memory_threshold: float = 95.0,  # N2: 95% GPU memory limit
    ) -> None:
        """Initialize memory manager.

        SPEC-PERFOPT-001: Default threshold is 30GB (30000MB) for forensic
        audio analysis workloads that require loading multiple ML models
        (WhisperX, SER, etc.) simultaneously.

        Args:
            threshold_mb: Memory threshold in MB for automatic garbage collection.
                          Default is 30000MB (30GB) for forensic workloads.
            enable_gc: Whether to enable automatic garbage collection
            enable_gpu_monitoring: Whether to monitor GPU memory
            system_memory_threshold: System memory percentage threshold for GC (S2)
            gpu_memory_threshold: GPU memory percentage limit (N2)
        """
        self.threshold_mb = threshold_mb
        self.enable_gc = enable_gc
        self.enable_gpu_monitoring = enable_gpu_monitoring
        self.system_memory_threshold = system_memory_threshold
        self.gpu_memory_threshold = gpu_memory_threshold
        self.stats = MemoryStats()
        self._collections_count = 0
        self._gpu_available = self._check_gpu_available()

    def _check_gpu_available(self) -> bool:
        """Check if GPU is available for monitoring."""
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False
        except Exception:
            return False

    def get_current_usage_mb(self) -> float:
        """Get current memory usage in MB.

        Returns:
            Current memory usage in megabytes
        """
        return MemoryStats._get_current_mb()

    def force_garbage_collection(self) -> int:
        """Force garbage collection and return number of objects collected.

        Also clears GPU cache if GPU monitoring is enabled.

        Returns:
            Number of objects collected
        """
        logger.debug("Forcing garbage collection")
        collected = gc.collect()

        generation_0 = len(gc.get_objects(0))
        generation_1 = len(gc.get_objects(1))
        generation_2 = len(gc.get_objects(2))

        logger.debug(
            f"GC collected {collected} objects. "
            f"Remaining: Gen0={generation_0}, Gen1={generation_1}, Gen2={generation_2}"
        )

        # Clear GPU cache if enabled
        if self.enable_gpu_monitoring and self._gpu_available:
            self._clear_gpu_cache()

        self._collections_count += 1
        return collected

    def _clear_gpu_cache(self) -> None:
        """Clear GPU memory cache."""
        try:
            import torch

            torch.cuda.empty_cache()
            logger.debug("GPU cache cleared")
        except Exception as e:
            logger.debug(f"Could not clear GPU cache: {e}")

    def should_collect(self) -> bool:
        """Check if garbage collection should be performed.

        Returns:
            True if memory usage exceeds threshold

        Implements:
            S2: GC trigger at 80% system memory usage
            N2: Memory allocation limits enforcement
        """
        if not self.enable_gc:
            return False

        # Check process memory threshold
        current_usage = self.get_current_usage_mb()
        if current_usage > self.threshold_mb:
            return True

        # S2: Check system memory percentage threshold
        try:
            vmem = psutil.virtual_memory()
            if vmem.percent >= self.system_memory_threshold:
                logger.info(
                    f"System memory at {vmem.percent:.1f}% exceeds "
                    f"{self.system_memory_threshold}% threshold"
                )
                return True
        except Exception as e:
            logger.debug(f"Could not check system memory: {e}")

        return False

    def track_memory_usage(self) -> MemoryStats:
        """Track and update memory usage statistics.

        Returns:
            Updated MemoryStats object
        """
        self.stats.update()

        # Auto-collect if threshold exceeded
        if self.should_collect():
            logger.info(
                f"Memory usage {self.stats.current_mb:.2f}MB exceeds threshold "
                f"{self.threshold_mb:.2f}MB. Forcing garbage collection."
            )
            self.force_garbage_collection()
            self.stats.update()

        return self.stats

    def get_memory_summary(self) -> dict:
        """Get comprehensive memory usage summary.

        Returns:
            Dictionary with memory statistics including GPU if enabled

        Implements:
            U2: GPU memory real-time monitoring
        """
        self.stats.update()
        summary = self.stats.to_dict()
        summary["threshold_mb"] = self.threshold_mb
        summary["gc_enabled"] = self.enable_gc
        summary["collections_count"] = self._collections_count
        summary["usage_percentage"] = (
            (self.stats.current_mb / self.threshold_mb * 100) if self.threshold_mb > 0 else 0
        )

        # Add system memory info
        try:
            vmem = psutil.virtual_memory()
            summary["system_memory_percent"] = vmem.percent
            summary["system_memory_total_mb"] = vmem.total / (1024 * 1024)
            summary["system_memory_available_mb"] = vmem.available / (1024 * 1024)
        except Exception:
            pass

        # Add GPU memory info if monitoring enabled (U2)
        if self.enable_gpu_monitoring and self._gpu_available:
            gpu_stats = self._get_gpu_memory_stats()
            if gpu_stats:
                summary["gpu_memory_mb"] = gpu_stats.get("used_mb", 0)
                summary["gpu_memory_percentage"] = gpu_stats.get("usage_percentage", 0)
                summary["gpu_memory_total_mb"] = gpu_stats.get("total_mb", 0)

        return summary

    def _get_gpu_memory_stats(self) -> Optional[Dict]:
        """Get GPU memory statistics.

        Returns:
            Dictionary with GPU memory info or None if unavailable
        """
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

            total_mb = mem_info.total / (1024 * 1024)
            used_mb = mem_info.used / (1024 * 1024)
            usage_percentage = (mem_info.used / mem_info.total) * 100

            return {
                "total_mb": total_mb,
                "used_mb": used_mb,
                "free_mb": (mem_info.total - mem_info.used) / (1024 * 1024),
                "usage_percentage": usage_percentage,
            }
        except Exception as e:
            logger.debug(f"Could not get GPU memory stats: {e}")
            return None

    def reset_tracking(self) -> None:
        """Reset memory tracking to current baseline."""
        self.stats = MemoryStats()
        logger.debug("Memory tracking reset")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup."""
        if self.enable_gc:
            logger.info("Context manager exit: forcing garbage collection")
            self.force_garbage_collection()
        return False
