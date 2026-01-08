"""
Unit tests for memory management service.

Tests garbage collection, memory monitoring, and resource cleanup.
"""

import gc
import pytest
from pathlib import Path

from voice_man.services.memory_service import MemoryManager, MemoryStats


class TestMemoryStats:
    """Test memory statistics tracking."""

    def test_initial_stats(self):
        """Test initial memory statistics."""
        stats = MemoryStats()
        assert stats.initial_mb > 0
        assert stats.current_mb > 0
        assert stats.peak_mb == stats.current_mb
        assert stats.delta_mb == 0.0

    def test_stats_update(self):
        """Test statistics update."""
        stats = MemoryStats()
        initial = stats.current_mb

        # Allocate some memory
        data = [i for i in range(1000000)]

        stats.update()

        assert stats.current_mb >= initial
        assert stats.peak_mb >= stats.current_mb
        assert stats.delta_mb == stats.current_mb - stats.initial_mb

        del data
        gc.collect()


class TestMemoryManager:
    """Test memory manager."""

    @pytest.fixture
    def manager(self):
        """Create a memory manager instance."""
        return MemoryManager(threshold_mb=100, enable_gc=True)

    def test_initialization(self, manager):
        """Test manager initialization."""
        assert manager.threshold_mb == 100
        assert manager.enable_gc is True
        assert manager.stats.initial_mb > 0

    def test_get_current_usage_mb(self, manager):
        """Test getting current memory usage."""
        usage = manager.get_current_usage_mb()
        assert usage > 0
        assert usage < 10000  # Reasonable upper bound (10GB)

    def test_force_garbage_collection(self, manager):
        """Test forced garbage collection."""
        # Create some objects
        data = [i for i in range(100000)]
        objects_before = len(gc.get_objects())

        # Force collection
        collected = manager.force_garbage_collection()

        # Clean up
        del data
        gc.collect()

        assert collected >= 0

    def test_should_collect_below_threshold(self, manager):
        """Test collection decision below threshold."""
        manager.threshold_mb = 10000  # Very high threshold
        assert manager.should_collect() is False

    def test_should_collect_above_threshold(self):
        """Test collection decision above threshold."""
        manager = MemoryManager(threshold_mb=0.001)  # Very low threshold
        assert manager.should_collect() is True

    def test_track_memory_usage(self, manager):
        """Test memory usage tracking."""
        initial_usage = manager.get_current_usage_mb()

        # Allocate memory
        data = [i for i in range(1000000)]

        # Track and check if usage increased
        manager.track_memory_usage()
        assert manager.stats.delta_mb >= 0

        # Clean up
        del data
        manager.force_garbage_collection()

    def test_get_memory_summary(self, manager):
        """Test memory summary generation."""
        summary = manager.get_memory_summary()

        assert "initial_mb" in summary
        assert "current_mb" in summary
        assert "peak_mb" in summary
        assert "delta_mb" in summary
        assert "threshold_mb" in summary
        assert summary["threshold_mb"] == 100

    def test_reset_tracking(self, manager):
        """Test resetting memory tracking."""
        # Allocate some memory
        data = [i for i in range(1000000)]
        manager.track_memory_usage()

        # Reset
        manager.reset_tracking()

        # Check that stats are reset
        assert manager.stats.delta_mb == 0.0
        assert manager.stats.peak_mb == manager.stats.current_mb

        # Clean up
        del data
        gc.collect()

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test memory manager as context manager."""
        async with MemoryManager(enable_gc=True) as manager:
            usage_before = manager.get_current_usage_mb()

            # Allocate memory
            data = [i for i in range(1000000)]

            usage_during = manager.get_current_usage_mb()
            assert usage_during >= usage_before

            # Clean up happens on exit
            del data

        # After context, garbage collection should have run
        assert manager is not None
