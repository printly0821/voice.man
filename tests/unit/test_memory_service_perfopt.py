"""
Unit tests for Memory Service Performance Optimization (SPEC-PERFOPT-001).

Tests for:
- TASK-001: Memory threshold change from 100MB to 30GB (30000MB)
- FORENSIC_MEMORY_THRESHOLD_MB constant

TDD RED Phase: These tests should FAIL before implementation.
"""

import pytest
from unittest.mock import patch, MagicMock


class TestMemoryThresholdOptimization:
    """Test memory threshold optimization for forensic analysis workloads."""

    def test_forensic_memory_threshold_constant_exists(self):
        """Test that FORENSIC_MEMORY_THRESHOLD_MB constant is defined."""
        from voice_man.services.memory_service import FORENSIC_MEMORY_THRESHOLD_MB

        assert FORENSIC_MEMORY_THRESHOLD_MB == 30000, (
            "FORENSIC_MEMORY_THRESHOLD_MB should be 30000 (30GB)"
        )

    def test_memory_manager_default_threshold_is_30gb(self):
        """Test that MemoryManager defaults to 30GB threshold for forensic workloads."""
        from voice_man.services.memory_service import MemoryManager

        manager = MemoryManager()

        # Default threshold should be 30GB (30000MB) for forensic analysis
        assert manager.threshold_mb == 30000, (
            f"Default threshold should be 30000MB (30GB), got {manager.threshold_mb}MB"
        )

    def test_memory_manager_accepts_custom_threshold(self):
        """Test that MemoryManager still accepts custom threshold values."""
        from voice_man.services.memory_service import MemoryManager

        custom_threshold = 16000  # 16GB
        manager = MemoryManager(threshold_mb=custom_threshold)

        assert manager.threshold_mb == custom_threshold, (
            f"Custom threshold {custom_threshold} not applied"
        )

    def test_memory_manager_docstring_reflects_new_threshold(self):
        """Test that MemoryManager docstring documents 30GB threshold."""
        from voice_man.services.memory_service import MemoryManager

        # Check that the docstring or __init__ mentions 30GB or forensic threshold
        init_doc = MemoryManager.__init__.__doc__

        assert init_doc is not None, "MemoryManager.__init__ should have a docstring"
        assert "30" in init_doc or "forensic" in init_doc.lower() or "30000" in init_doc, (
            "Docstring should document the 30GB forensic memory threshold"
        )

    def test_should_collect_uses_correct_threshold(self):
        """Test that should_collect uses the new 30GB threshold correctly."""
        from voice_man.services.memory_service import MemoryManager

        manager = MemoryManager()

        # With a 30GB threshold, should_collect should not trigger on typical usage
        # Patch the current usage to be below threshold
        with patch.object(manager, "get_current_usage_mb", return_value=1000):
            # 1GB usage should not trigger collection with 30GB threshold
            assert manager.should_collect() is False

    def test_should_collect_triggers_above_threshold(self):
        """Test that should_collect triggers when usage exceeds 30GB threshold."""
        from voice_man.services.memory_service import MemoryManager

        manager = MemoryManager()

        # Mock high memory usage above threshold
        with patch.object(manager, "get_current_usage_mb", return_value=35000):
            # Also need to patch psutil to avoid system memory check triggering
            with patch("psutil.virtual_memory") as mock_vmem:
                mock_vmem.return_value = MagicMock(percent=50.0)
                # 35GB usage should trigger collection with 30GB threshold
                assert manager.should_collect() is True

    def test_get_memory_summary_shows_correct_threshold(self):
        """Test that memory summary shows the 30GB threshold."""
        from voice_man.services.memory_service import MemoryManager

        manager = MemoryManager()
        summary = manager.get_memory_summary()

        assert summary["threshold_mb"] == 30000, (
            f"Memory summary should show 30000MB threshold, got {summary['threshold_mb']}"
        )


class TestMemoryThresholdBackwardsCompatibility:
    """Test backwards compatibility when using custom thresholds."""

    def test_explicit_100mb_threshold_still_works(self):
        """Test that explicitly setting 100MB threshold still works."""
        from voice_man.services.memory_service import MemoryManager

        manager = MemoryManager(threshold_mb=100)
        assert manager.threshold_mb == 100

    def test_explicit_threshold_overrides_default(self):
        """Test that explicit threshold overrides the 30GB default."""
        from voice_man.services.memory_service import MemoryManager

        manager = MemoryManager(threshold_mb=8000)
        assert manager.threshold_mb == 8000

    def test_usage_percentage_calculated_correctly(self):
        """Test that usage percentage is calculated with correct threshold."""
        from voice_man.services.memory_service import MemoryManager

        manager = MemoryManager()  # 30GB default

        with patch.object(manager.stats, "_get_current_mb", return_value=3000):
            manager.stats.update()
            summary = manager.get_memory_summary()

            # 3GB / 30GB = 10%
            expected_percentage = (3000 / 30000) * 100
            assert abs(summary["usage_percentage"] - expected_percentage) < 0.1, (
                f"Expected ~{expected_percentage}%, got {summary['usage_percentage']}%"
            )
