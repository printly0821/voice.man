"""
Memory management service for audio file processing.

Provides garbage collection, memory monitoring, and resource cleanup.
"""

import gc
import logging
import psutil
from typing import Optional
from dataclasses import dataclass, field
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


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
    """Manage memory during batch processing."""

    def __init__(self, threshold_mb: float = 100, enable_gc: bool = True) -> None:
        """Initialize memory manager.

        Args:
            threshold_mb: Memory threshold in MB for automatic garbage collection
            enable_gc: Whether to enable automatic garbage collection
        """
        self.threshold_mb = threshold_mb
        self.enable_gc = enable_gc
        self.stats = MemoryStats()
        self._collections_count = 0

    def get_current_usage_mb(self) -> float:
        """Get current memory usage in MB.

        Returns:
            Current memory usage in megabytes
        """
        return MemoryStats._get_current_mb()

    def force_garbage_collection(self) -> int:
        """Force garbage collection and return number of objects collected.

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

        self._collections_count += 1
        return collected

    def should_collect(self) -> bool:
        """Check if garbage collection should be performed.

        Returns:
            True if memory usage exceeds threshold
        """
        if not self.enable_gc:
            return False

        current_usage = self.get_current_usage_mb()
        return current_usage > self.threshold_mb

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
            Dictionary with memory statistics
        """
        self.stats.update()
        summary = self.stats.to_dict()
        summary["threshold_mb"] = self.threshold_mb
        summary["gc_enabled"] = self.enable_gc
        summary["collections_count"] = self._collections_count
        summary["usage_percentage"] = (
            (self.stats.current_mb / self.threshold_mb * 100) if self.threshold_mb > 0 else 0
        )
        return summary

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
