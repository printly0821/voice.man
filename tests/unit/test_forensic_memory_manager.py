"""
ForensicMemoryManager Unit Tests

SPEC-PERFOPT-001 Phase 2: Stage-based GPU memory allocation for forensic pipeline.
TDD RED Phase - Tests written FIRST before implementation.

Stage Memory Allocations:
    - STT: 16GB
    - Alignment: 4GB
    - Diarization: 8GB
    - SER: 10GB
    - Scoring: 2GB
"""

import pytest
import threading
import time
from unittest.mock import patch, MagicMock, PropertyMock


class TestForensicMemoryManagerImport:
    """Test that ForensicMemoryManager can be imported."""

    def test_import_forensic_memory_manager(self):
        """
        GIVEN: SPEC-PERFOPT-001 Phase 2 implementation
        WHEN: Importing ForensicMemoryManager
        THEN: The import should succeed without errors
        """
        from voice_man.services.forensic.memory_manager import ForensicMemoryManager

        assert ForensicMemoryManager is not None


class TestForensicMemoryManagerInitialization:
    """Test ForensicMemoryManager initialization."""

    def test_initialization_with_default_allocations(self):
        """
        GIVEN: ForensicMemoryManager class
        WHEN: Initialized with default parameters
        THEN: Should have correct stage allocations in MB
            - STT: 16384 MB (16GB)
            - Alignment: 4096 MB (4GB)
            - Diarization: 8192 MB (8GB)
            - SER: 10240 MB (10GB)
            - Scoring: 2048 MB (2GB)
        """
        from voice_man.services.forensic.memory_manager import ForensicMemoryManager

        manager = ForensicMemoryManager()

        assert manager.stage_allocations["stt"] == 16384
        assert manager.stage_allocations["alignment"] == 4096
        assert manager.stage_allocations["diarization"] == 8192
        assert manager.stage_allocations["ser"] == 10240
        assert manager.stage_allocations["scoring"] == 2048

    def test_initialization_with_custom_allocations(self):
        """
        GIVEN: ForensicMemoryManager class
        WHEN: Initialized with custom allocations
        THEN: Should use custom values
        """
        from voice_man.services.forensic.memory_manager import ForensicMemoryManager

        custom_allocations = {
            "stt": 8192,
            "alignment": 2048,
            "diarization": 4096,
            "ser": 5120,
            "scoring": 1024,
        }

        manager = ForensicMemoryManager(stage_allocations=custom_allocations)

        assert manager.stage_allocations["stt"] == 8192
        assert manager.stage_allocations["ser"] == 5120

    def test_has_thread_lock(self):
        """
        GIVEN: ForensicMemoryManager class
        WHEN: Initialized
        THEN: Should have a threading.Lock for thread-safety
        """
        from voice_man.services.forensic.memory_manager import ForensicMemoryManager

        manager = ForensicMemoryManager()

        assert hasattr(manager, "_lock")
        assert isinstance(manager._lock, type(threading.Lock()))


class TestMemoryAllocation:
    """Test memory allocation methods."""

    def test_allocate_stage_memory(self):
        """
        GIVEN: ForensicMemoryManager instance
        WHEN: allocate() is called for 'stt' stage
        THEN: Should allocate 16384 MB and mark stage as allocated
        """
        from voice_man.services.forensic.memory_manager import ForensicMemoryManager

        manager = ForensicMemoryManager()

        result = manager.allocate("stt")

        assert result is True
        assert manager.is_allocated("stt") is True

    def test_allocate_with_custom_amount(self):
        """
        GIVEN: ForensicMemoryManager instance
        WHEN: allocate() is called with custom amount_mb
        THEN: Should use custom amount instead of default
        """
        from voice_man.services.forensic.memory_manager import ForensicMemoryManager

        manager = ForensicMemoryManager()

        result = manager.allocate("stt", amount_mb=8000)

        assert result is True
        stats = manager.get_memory_stats()
        assert stats["allocated_stages"]["stt"]["amount_mb"] == 8000

    def test_allocate_unknown_stage_raises_error(self):
        """
        GIVEN: ForensicMemoryManager instance
        WHEN: allocate() is called with unknown stage
        THEN: Should raise ValueError
        """
        from voice_man.services.forensic.memory_manager import ForensicMemoryManager

        manager = ForensicMemoryManager()

        with pytest.raises(ValueError, match="Unknown stage"):
            manager.allocate("unknown_stage")

    def test_double_allocation_returns_false(self):
        """
        GIVEN: ForensicMemoryManager with 'stt' already allocated
        WHEN: allocate() is called again for 'stt'
        THEN: Should return False (already allocated)
        """
        from voice_man.services.forensic.memory_manager import ForensicMemoryManager

        manager = ForensicMemoryManager()
        manager.allocate("stt")

        result = manager.allocate("stt")

        assert result is False

    def test_allocate_all_stages(self):
        """
        GIVEN: ForensicMemoryManager instance
        WHEN: allocate() is called for all stages
        THEN: All stages should be allocated
        """
        from voice_man.services.forensic.memory_manager import ForensicMemoryManager

        manager = ForensicMemoryManager()

        stages = ["stt", "alignment", "diarization", "ser", "scoring"]
        for stage in stages:
            result = manager.allocate(stage)
            assert result is True, f"Failed to allocate {stage}"

        for stage in stages:
            assert manager.is_allocated(stage) is True


class TestMemoryRelease:
    """Test memory release methods."""

    def test_release_allocated_memory(self):
        """
        GIVEN: ForensicMemoryManager with 'stt' allocated
        WHEN: release() is called for 'stt'
        THEN: Should release memory and mark stage as not allocated
        """
        from voice_man.services.forensic.memory_manager import ForensicMemoryManager

        manager = ForensicMemoryManager()
        manager.allocate("stt")

        result = manager.release("stt")

        assert result is True
        assert manager.is_allocated("stt") is False

    def test_release_not_allocated_returns_false(self):
        """
        GIVEN: ForensicMemoryManager without 'stt' allocated
        WHEN: release() is called for 'stt'
        THEN: Should return False
        """
        from voice_man.services.forensic.memory_manager import ForensicMemoryManager

        manager = ForensicMemoryManager()

        result = manager.release("stt")

        assert result is False

    def test_release_unknown_stage_raises_error(self):
        """
        GIVEN: ForensicMemoryManager instance
        WHEN: release() is called with unknown stage
        THEN: Should raise ValueError
        """
        from voice_man.services.forensic.memory_manager import ForensicMemoryManager

        manager = ForensicMemoryManager()

        with pytest.raises(ValueError, match="Unknown stage"):
            manager.release("unknown_stage")

    def test_release_all_stages(self):
        """
        GIVEN: ForensicMemoryManager with all stages allocated
        WHEN: release_all() is called
        THEN: All stages should be released
        """
        from voice_man.services.forensic.memory_manager import ForensicMemoryManager

        manager = ForensicMemoryManager()
        stages = ["stt", "alignment", "diarization", "ser", "scoring"]
        for stage in stages:
            manager.allocate(stage)

        manager.release_all()

        for stage in stages:
            assert manager.is_allocated(stage) is False


class TestMemoryStats:
    """Test memory statistics methods."""

    def test_get_memory_stats_initial(self):
        """
        GIVEN: Fresh ForensicMemoryManager instance
        WHEN: get_memory_stats() is called
        THEN: Should return stats with no allocations
        """
        from voice_man.services.forensic.memory_manager import ForensicMemoryManager

        manager = ForensicMemoryManager()

        stats = manager.get_memory_stats()

        assert "total_allocated_mb" in stats
        assert stats["total_allocated_mb"] == 0
        assert "allocated_stages" in stats
        assert len(stats["allocated_stages"]) == 0

    def test_get_memory_stats_after_allocations(self):
        """
        GIVEN: ForensicMemoryManager with 'stt' and 'ser' allocated
        WHEN: get_memory_stats() is called
        THEN: Should return correct total and per-stage allocations
        """
        from voice_man.services.forensic.memory_manager import ForensicMemoryManager

        manager = ForensicMemoryManager()
        manager.allocate("stt")  # 16384 MB
        manager.allocate("ser")  # 10240 MB

        stats = manager.get_memory_stats()

        assert stats["total_allocated_mb"] == 16384 + 10240
        assert "stt" in stats["allocated_stages"]
        assert "ser" in stats["allocated_stages"]

    def test_get_memory_stats_includes_gpu_info(self):
        """
        GIVEN: ForensicMemoryManager instance
        WHEN: get_memory_stats() is called
        THEN: Should include GPU memory information if available
        """
        from voice_man.services.forensic.memory_manager import ForensicMemoryManager

        manager = ForensicMemoryManager()

        stats = manager.get_memory_stats()

        assert "gpu_available" in stats
        # If GPU available, should have additional fields
        if stats["gpu_available"]:
            assert "gpu_total_mb" in stats
            assert "gpu_used_mb" in stats
            assert "gpu_free_mb" in stats


class TestThreadSafety:
    """Test thread-safety of memory operations."""

    def test_concurrent_allocations(self):
        """
        GIVEN: ForensicMemoryManager instance
        WHEN: Multiple threads try to allocate different stages
        THEN: All allocations should complete without race conditions
        """
        from voice_man.services.forensic.memory_manager import ForensicMemoryManager

        manager = ForensicMemoryManager()
        results = {}
        threads = []

        def allocate_stage(stage):
            results[stage] = manager.allocate(stage)

        stages = ["stt", "alignment", "diarization", "ser", "scoring"]
        for stage in stages:
            t = threading.Thread(target=allocate_stage, args=(stage,))
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All allocations should succeed
        assert all(results.values())

    def test_concurrent_allocation_same_stage(self):
        """
        GIVEN: ForensicMemoryManager instance
        WHEN: Multiple threads try to allocate the same stage
        THEN: Only one should succeed
        """
        from voice_man.services.forensic.memory_manager import ForensicMemoryManager

        manager = ForensicMemoryManager()
        results = []
        threads = []

        def allocate_stt():
            results.append(manager.allocate("stt"))

        for _ in range(5):
            t = threading.Thread(target=allocate_stt)
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Exactly one should succeed
        assert results.count(True) == 1
        assert results.count(False) == 4


class TestGPUIntegration:
    """Test GPU memory integration via pynvml."""

    def test_gpu_memory_query_with_pynvml(self):
        """
        GIVEN: ForensicMemoryManager with pynvml available
        WHEN: GPU memory is queried
        THEN: Should return actual GPU memory stats
        """
        from voice_man.services.forensic.memory_manager import ForensicMemoryManager

        manager = ForensicMemoryManager()

        stats = manager.get_memory_stats()

        # Should have GPU availability flag
        assert "gpu_available" in stats

    def test_fallback_when_pynvml_unavailable(self):
        """
        GIVEN: ForensicMemoryManager when pynvml is not available
        WHEN: GPU memory is queried
        THEN: Should gracefully fallback with gpu_available=False
        """
        from voice_man.services.forensic.memory_manager import ForensicMemoryManager

        with patch.dict("sys.modules", {"pynvml": None}):
            manager = ForensicMemoryManager()
            stats = manager.get_memory_stats()

            # Should still work but indicate GPU unavailable
            assert "gpu_available" in stats


class TestContextManager:
    """Test context manager support."""

    def test_context_manager_allocate_release(self):
        """
        GIVEN: ForensicMemoryManager instance
        WHEN: Used as context manager for a stage
        THEN: Should automatically allocate on enter and release on exit
        """
        from voice_man.services.forensic.memory_manager import ForensicMemoryManager

        manager = ForensicMemoryManager()

        with manager.stage_context("stt") as allocated:
            assert allocated is True
            assert manager.is_allocated("stt") is True

        assert manager.is_allocated("stt") is False

    def test_context_manager_exception_handling(self):
        """
        GIVEN: ForensicMemoryManager used as context manager
        WHEN: Exception occurs inside the context
        THEN: Memory should still be released
        """
        from voice_man.services.forensic.memory_manager import ForensicMemoryManager

        manager = ForensicMemoryManager()

        try:
            with manager.stage_context("stt"):
                assert manager.is_allocated("stt") is True
                raise ValueError("Test exception")
        except ValueError:
            pass

        assert manager.is_allocated("stt") is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
