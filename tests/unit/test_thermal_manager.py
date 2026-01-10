"""
ThermalManager Unit Tests for Forensic Pipeline

SPEC-PERFOPT-001 Phase 2: GPU thermal management for forensic analysis.
TDD RED Phase - Tests written FIRST before implementation.

Thresholds:
    - THROTTLE_START: 80C
    - THROTTLE_STOP: 70C
    - CRITICAL: 85C

Features:
    - pynvml GPU temperature monitoring
    - Background monitoring thread with hysteresis
    - Callback registration for throttling events
"""

import pytest
import threading
import time
from unittest.mock import patch, MagicMock, PropertyMock


class TestThermalManagerImport:
    """Test that ThermalManager can be imported."""

    def test_import_thermal_manager(self):
        """
        GIVEN: SPEC-PERFOPT-001 Phase 2 implementation
        WHEN: Importing ThermalManager from forensic module
        THEN: The import should succeed without errors
        """
        from voice_man.services.forensic.thermal_manager import ThermalManager

        assert ThermalManager is not None

    def test_import_thermal_thresholds(self):
        """
        GIVEN: SPEC-PERFOPT-001 Phase 2 implementation
        WHEN: Importing thermal threshold constants
        THEN: Should have THROTTLE_START, THROTTLE_STOP, CRITICAL constants
        """
        from voice_man.services.forensic.thermal_manager import (
            THROTTLE_START_TEMP,
            THROTTLE_STOP_TEMP,
            CRITICAL_TEMP,
        )

        assert THROTTLE_START_TEMP == 80
        assert THROTTLE_STOP_TEMP == 70
        assert CRITICAL_TEMP == 85


class TestThermalManagerInitialization:
    """Test ThermalManager initialization."""

    def test_initialization_with_defaults(self):
        """
        GIVEN: ThermalManager class
        WHEN: Initialized without parameters
        THEN: Should use default thresholds (80, 70, 85)
        """
        from voice_man.services.forensic.thermal_manager import ThermalManager

        manager = ThermalManager()

        assert manager.throttle_start_temp == 80
        assert manager.throttle_stop_temp == 70
        assert manager.critical_temp == 85

    def test_initialization_with_custom_thresholds(self):
        """
        GIVEN: ThermalManager class
        WHEN: Initialized with custom thresholds
        THEN: Should use custom values
        """
        from voice_man.services.forensic.thermal_manager import ThermalManager

        manager = ThermalManager(
            throttle_start_temp=75,
            throttle_stop_temp=65,
            critical_temp=80,
        )

        assert manager.throttle_start_temp == 75
        assert manager.throttle_stop_temp == 65
        assert manager.critical_temp == 80

    def test_invalid_thresholds_raises_error(self):
        """
        GIVEN: ThermalManager class
        WHEN: Initialized with invalid thresholds (stop >= start)
        THEN: Should raise ValueError
        """
        from voice_man.services.forensic.thermal_manager import ThermalManager

        with pytest.raises(ValueError):
            ThermalManager(throttle_start_temp=70, throttle_stop_temp=80)

    def test_has_monitoring_state(self):
        """
        GIVEN: ThermalManager instance
        WHEN: Initialized
        THEN: Should have monitoring state attributes
        """
        from voice_man.services.forensic.thermal_manager import ThermalManager

        manager = ThermalManager()

        assert hasattr(manager, "is_monitoring")
        assert manager.is_monitoring is False


class TestTemperatureMonitoring:
    """Test temperature monitoring functionality."""

    def test_get_current_temperature(self):
        """
        GIVEN: ThermalManager instance
        WHEN: get_current_temperature() is called
        THEN: Should return integer temperature value
        """
        from voice_man.services.forensic.thermal_manager import ThermalManager

        manager = ThermalManager()

        temp = manager.get_current_temperature()

        assert isinstance(temp, int)
        assert 0 <= temp <= 150  # Reasonable range

    def test_get_temperature_without_gpu(self):
        """
        GIVEN: ThermalManager without GPU handle
        WHEN: get_current_temperature() is called
        THEN: Should return default temperature (50)
        """
        from voice_man.services.forensic.thermal_manager import ThermalManager

        manager = ThermalManager()
        manager._gpu_handle = None

        temp = manager.get_current_temperature()

        assert temp == 50  # Default value


class TestThrottlingState:
    """Test throttling state management."""

    def test_initial_throttle_state(self):
        """
        GIVEN: Fresh ThermalManager instance
        WHEN: Checking throttle state
        THEN: Should not be throttling
        """
        from voice_man.services.forensic.thermal_manager import ThermalManager

        manager = ThermalManager()

        assert manager.is_throttling is False

    def test_throttle_starts_at_threshold(self):
        """
        GIVEN: ThermalManager instance
        WHEN: Temperature reaches THROTTLE_START (80C)
        THEN: Should enter throttling state
        """
        from voice_man.services.forensic.thermal_manager import ThermalManager

        manager = ThermalManager()

        manager.check_temperature(current_temp=80)

        assert manager.is_throttling is True

    def test_throttle_stops_below_threshold(self):
        """
        GIVEN: ThermalManager in throttling state
        WHEN: Temperature drops below THROTTLE_STOP (70C)
        THEN: Should exit throttling state (hysteresis)
        """
        from voice_man.services.forensic.thermal_manager import ThermalManager

        manager = ThermalManager()
        manager.check_temperature(current_temp=85)  # Start throttling
        assert manager.is_throttling is True

        manager.check_temperature(current_temp=69)  # Below stop threshold

        assert manager.is_throttling is False

    def test_hysteresis_behavior(self):
        """
        GIVEN: ThermalManager in throttling state
        WHEN: Temperature is between STOP (70C) and START (80C)
        THEN: Should maintain current state (hysteresis)
        """
        from voice_man.services.forensic.thermal_manager import ThermalManager

        manager = ThermalManager()

        # Start throttling
        manager.check_temperature(current_temp=82)
        assert manager.is_throttling is True

        # Temperature drops but still above stop threshold
        manager.check_temperature(current_temp=75)
        assert manager.is_throttling is True  # Should maintain throttling

    def test_critical_temperature_detection(self):
        """
        GIVEN: ThermalManager instance
        WHEN: Temperature reaches CRITICAL (85C)
        THEN: Should set is_critical flag
        """
        from voice_man.services.forensic.thermal_manager import ThermalManager

        manager = ThermalManager()

        manager.check_temperature(current_temp=86)

        assert manager.is_critical is True


class TestCallbackRegistration:
    """Test callback registration for throttling events."""

    def test_register_throttle_callback(self):
        """
        GIVEN: ThermalManager instance
        WHEN: Registering a throttle callback
        THEN: Callback should be stored
        """
        from voice_man.services.forensic.thermal_manager import ThermalManager

        manager = ThermalManager()

        callback_called = []

        def on_throttle(is_throttling: bool):
            callback_called.append(is_throttling)

        manager.register_throttle_callback(on_throttle)

        assert len(manager._throttle_callbacks) == 1

    def test_callback_invoked_on_throttle_start(self):
        """
        GIVEN: ThermalManager with registered callback
        WHEN: Throttling starts
        THEN: Callback should be invoked with True
        """
        from voice_man.services.forensic.thermal_manager import ThermalManager

        manager = ThermalManager()
        callback_results = []

        def on_throttle(is_throttling: bool):
            callback_results.append(is_throttling)

        manager.register_throttle_callback(on_throttle)
        manager.check_temperature(current_temp=82)

        assert True in callback_results

    def test_callback_invoked_on_throttle_stop(self):
        """
        GIVEN: ThermalManager throttling with registered callback
        WHEN: Throttling stops
        THEN: Callback should be invoked with False
        """
        from voice_man.services.forensic.thermal_manager import ThermalManager

        manager = ThermalManager()
        callback_results = []

        def on_throttle(is_throttling: bool):
            callback_results.append(is_throttling)

        manager.register_throttle_callback(on_throttle)
        manager.check_temperature(current_temp=82)  # Start throttling
        manager.check_temperature(current_temp=65)  # Stop throttling

        assert callback_results[-1] is False

    def test_register_critical_callback(self):
        """
        GIVEN: ThermalManager instance
        WHEN: Registering a critical temperature callback
        THEN: Callback should be invoked when critical temp reached
        """
        from voice_man.services.forensic.thermal_manager import ThermalManager

        manager = ThermalManager()
        critical_triggered = []

        def on_critical():
            critical_triggered.append(True)

        manager.register_critical_callback(on_critical)
        manager.check_temperature(current_temp=86)

        assert len(critical_triggered) == 1

    def test_unregister_callback(self):
        """
        GIVEN: ThermalManager with registered callback
        WHEN: Unregistering the callback
        THEN: Callback should be removed
        """
        from voice_man.services.forensic.thermal_manager import ThermalManager

        manager = ThermalManager()

        def on_throttle(is_throttling: bool):
            pass

        manager.register_throttle_callback(on_throttle)
        manager.unregister_throttle_callback(on_throttle)

        assert len(manager._throttle_callbacks) == 0


class TestBackgroundMonitoring:
    """Test background monitoring thread."""

    def test_start_monitoring(self):
        """
        GIVEN: ThermalManager instance
        WHEN: start_monitoring() is called
        THEN: Should start background monitoring thread
        """
        from voice_man.services.forensic.thermal_manager import ThermalManager

        manager = ThermalManager()

        manager.start_monitoring(interval_seconds=0.1)

        try:
            assert manager.is_monitoring is True
            time.sleep(0.2)  # Let it run briefly
        finally:
            manager.stop_monitoring()

    def test_stop_monitoring(self):
        """
        GIVEN: ThermalManager with active monitoring
        WHEN: stop_monitoring() is called
        THEN: Should stop the monitoring thread
        """
        from voice_man.services.forensic.thermal_manager import ThermalManager

        manager = ThermalManager()
        manager.start_monitoring(interval_seconds=0.1)

        manager.stop_monitoring()

        assert manager.is_monitoring is False

    def test_monitoring_invokes_temperature_check(self):
        """
        GIVEN: ThermalManager with active monitoring
        WHEN: Monitoring runs for a period
        THEN: Should check temperature multiple times
        """
        from voice_man.services.forensic.thermal_manager import ThermalManager

        manager = ThermalManager()
        check_count = [0]

        original_check = manager.check_temperature

        def counting_check(current_temp=None):
            check_count[0] += 1
            return original_check(current_temp)

        manager.check_temperature = counting_check

        manager.start_monitoring(interval_seconds=0.05)
        time.sleep(0.2)
        manager.stop_monitoring()

        assert check_count[0] >= 2


class TestThermalStats:
    """Test thermal statistics."""

    def test_get_thermal_stats(self):
        """
        GIVEN: ThermalManager instance
        WHEN: get_thermal_stats() is called
        THEN: Should return dictionary with thermal info
        """
        from voice_man.services.forensic.thermal_manager import ThermalManager

        manager = ThermalManager()

        stats = manager.get_thermal_stats()

        assert "current_temp" in stats
        assert "is_throttling" in stats
        assert "is_critical" in stats
        assert "throttle_start_temp" in stats
        assert "throttle_stop_temp" in stats
        assert "critical_temp" in stats

    def test_stats_include_history(self):
        """
        GIVEN: ThermalManager with temperature history
        WHEN: get_thermal_stats() is called
        THEN: Should include max and average temperatures
        """
        from voice_man.services.forensic.thermal_manager import ThermalManager

        manager = ThermalManager()
        manager.check_temperature(current_temp=70)
        manager.check_temperature(current_temp=75)
        manager.check_temperature(current_temp=72)

        stats = manager.get_thermal_stats()

        assert "max_temp" in stats
        assert "avg_temp" in stats
        assert stats["max_temp"] == 75


class TestContextManager:
    """Test context manager support for monitoring."""

    def test_context_manager_starts_stops_monitoring(self):
        """
        GIVEN: ThermalManager instance
        WHEN: Used as context manager
        THEN: Should start monitoring on enter and stop on exit
        """
        from voice_man.services.forensic.thermal_manager import ThermalManager

        manager = ThermalManager()

        with manager.monitoring_context(interval_seconds=0.1):
            assert manager.is_monitoring is True
            time.sleep(0.1)

        assert manager.is_monitoring is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
