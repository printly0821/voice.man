#!/usr/bin/env python3
"""
Test suite for DynamicBatchAdjuster class.

Tests memory pressure-based batch size adjustment functionality.
"""

import sys
import pytest
from dataclasses import dataclass
from typing import Dict, Any

# Add src to path
sys.path.insert(0, "src")
sys.path.insert(0, "scripts")


@dataclass
class MockMemoryMonitor:
    """Mock MemoryMonitor for testing."""

    ram_total_mb: float = 122568.9
    ram_used_mb: float = 10000.0

    def get_memory_stats(self) -> Dict[str, Any]:
        return {
            "ram_total_mb": self.ram_total_mb,
            "ram_used_mb": self.ram_used_mb,
        }


class TestDynamicBatchAdjuster:
    """Test suite for DynamicBatchAdjuster."""

    def setup_method(self):
        """Setup test fixtures."""
        # Import here to avoid import issues
        from scripts.run_safe_forensic_batch import DynamicBatchAdjuster

        self.DynamicBatchAdjuster = DynamicBatchAdjuster
        self.initial_batch_size = 10

    def test_low_memory_pressure(self):
        """Test batch size at LOW memory pressure (< 50%)."""
        monitor = MockMemoryMonitor(ram_used_mb=30000)  # ~25%
        adjuster = self.DynamicBatchAdjuster(
            memory_monitor=monitor, initial_batch_size=self.initial_batch_size
        )

        assert adjuster.get_memory_pressure_level() == "LOW"
        assert adjuster.calculate_batch_size("LOW") == 10

    def test_medium_memory_pressure(self):
        """Test batch size at MEDIUM memory pressure (50-70%)."""
        monitor = MockMemoryMonitor(ram_used_mb=70000)  # ~57%
        adjuster = self.DynamicBatchAdjuster(
            memory_monitor=monitor, initial_batch_size=self.initial_batch_size
        )

        assert adjuster.get_memory_pressure_level() == "MEDIUM"
        assert adjuster.calculate_batch_size("MEDIUM") == 5

    def test_high_memory_pressure(self):
        """Test batch size at HIGH memory pressure (70-90%)."""
        monitor = MockMemoryMonitor(ram_used_mb=95000)  # ~77%
        adjuster = self.DynamicBatchAdjuster(
            memory_monitor=monitor, initial_batch_size=self.initial_batch_size
        )

        assert adjuster.get_memory_pressure_level() == "HIGH"
        assert adjuster.calculate_batch_size("HIGH") == 3

    def test_critical_memory_pressure(self):
        """Test batch size at CRITICAL memory pressure (> 90%)."""
        monitor = MockMemoryMonitor(ram_used_mb=115000)  # ~94%
        adjuster = self.DynamicBatchAdjuster(
            memory_monitor=monitor, initial_batch_size=self.initial_batch_size
        )

        assert adjuster.get_memory_pressure_level() == "CRITICAL"
        assert adjuster.calculate_batch_size("CRITICAL") == 1

    def test_adjust_batch_size_decrease(self):
        """Test batch size adjustment when memory pressure increases."""
        monitor = MockMemoryMonitor(ram_used_mb=30000)  # Start LOW
        adjuster = self.DynamicBatchAdjuster(
            memory_monitor=monitor, initial_batch_size=self.initial_batch_size
        )

        # Initial LOW pressure
        assert adjuster.adjust_batch_size() == 10

        # Simulate memory pressure increase
        monitor.ram_used_mb = 95000  # HIGH
        assert adjuster.adjust_batch_size() == 3

    def test_min_batch_size_constraint(self):
        """Test that batch size never goes below minimum."""
        monitor = MockMemoryMonitor(ram_used_mb=120000)  # CRITICAL
        adjuster = self.DynamicBatchAdjuster(
            memory_monitor=monitor, initial_batch_size=2, min_batch_size=1
        )

        assert adjuster.calculate_batch_size("CRITICAL") == 1

    def test_max_batch_size_constraint(self):
        """Test that batch size never exceeds maximum."""
        monitor = MockMemoryMonitor(ram_used_mb=10000)  # LOW
        adjuster = self.DynamicBatchAdjuster(
            memory_monitor=monitor, initial_batch_size=20, max_batch_size=10
        )

        assert adjuster.calculate_batch_size("LOW") == 10

    def test_get_ram_percent(self):
        """Test RAM percentage calculation."""
        monitor = MockMemoryMonitor(ram_used_mb=61284)  # 50%
        adjuster = self.DynamicBatchAdjuster(
            memory_monitor=monitor, initial_batch_size=self.initial_batch_size
        )

        assert adjuster.get_ram_percent() == pytest.approx(50.0, rel=0.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
