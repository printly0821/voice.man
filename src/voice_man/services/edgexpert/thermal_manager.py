"""
ThermalManager - MiniPC thermal management for MSI EdgeXpert.

This module provides GPU temperature monitoring and dynamic throttling
to maintain temperature below 85°C in miniPC form factor.

Features:
    - Real-time GPU temperature monitoring
    - Dynamic batch size adjustment based on temperature
    - Thermal policy management
    - Temperature history tracking
    - Cooldown mechanisms

Reference: SPEC-EDGEXPERT-001 Phase 2
"""

import logging
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Try to import NVML for GPU monitoring
try:
    import pynvml

    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    logger.warning("pynvml not available, thermal monitoring will be limited")


@dataclass
class ThermalPolicy:
    """Thermal policy configuration."""

    max_temp: int = 85
    warning_temp: int = 80
    target_temp: int = 70

    def __post_init__(self):
        """Validate thermal policy parameters."""
        if self.warning_temp >= self.max_temp:
            raise ValueError(
                f"warning_temp ({self.warning_temp}) must be < max_temp ({self.max_temp})"
            )
        if self.target_temp >= self.warning_temp:
            raise ValueError(
                f"target_temp ({self.target_temp}) must be < warning_temp ({self.warning_temp})"
            )


class ThermalHistory:
    """Temperature history tracking."""

    def __init__(self, max_samples: int = 1000):
        """Initialize temperature history.

        Args:
            max_samples: Maximum number of temperature samples to store
        """
        self.max_samples = max_samples
        self.temperatures: List[int] = []
        self.timestamps: List[float] = []

    def record(self, temp: int) -> None:
        """Record a temperature sample."""
        self.temperatures.append(temp)
        self.timestamps.append(time.time())

        # Trim to max_samples
        if len(self.temperatures) > self.max_samples:
            self.temperatures = self.temperatures[-self.max_samples :]
            self.timestamps = self.timestamps[-self.max_samples :]

    def get_average(self) -> float:
        """Get average temperature."""
        if not self.temperatures:
            return 0.0
        return sum(self.temperatures) / len(self.temperatures)

    def get_max(self) -> int:
        """Get maximum temperature."""
        return max(self.temperatures) if self.temperatures else 0

    def get_min(self) -> int:
        """Get minimum temperature."""
        return min(self.temperatures) if self.temperatures else 0

    def get_all(self) -> List[int]:
        """Get all temperature samples."""
        return self.temperatures.copy()


class CooldownManager:
    """Cooldown mode management."""

    def __init__(self):
        """Initialize cooldown manager."""
        self.in_cooldown = False
        self.cooldown_start_time: Optional[float] = None
        self.min_cooldown_duration = 30.0  # seconds

    def enter_cooldown(self) -> None:
        """Enter cooldown mode."""
        self.in_cooldown = True
        self.cooldown_start_time = time.time()
        logger.warning("Entering cooldown mode due to overheating")

    def exit_cooldown(self) -> None:
        """Exit cooldown mode."""
        self.in_cooldown = False
        self.cooldown_start_time = None
        logger.info("Exiting cooldown mode")

    def is_in_cooldown(self) -> bool:
        """Check if currently in cooldown mode."""
        return self.in_cooldown

    def can_exit_cooldown(self, current_temp: int, target_temp: int) -> bool:
        """Check if cooldown can be exited."""
        if not self.in_cooldown:
            return True

        # Must be in cooldown for minimum duration
        if self.cooldown_start_time is None:
            return False

        elapsed = time.time() - self.cooldown_start_time
        if elapsed < self.min_cooldown_duration:
            return False

        # Temperature must be below target
        if current_temp > target_temp:
            return False

        return True


class ThermalManager:
    """
    MiniPC thermal management system.

    Maintains GPU temperature below 85°C through:
        - Continuous temperature monitoring
        - Dynamic batch size adjustment
        - Thermal throttling
        - Cooldown mechanisms

    Attributes:
        max_temp: Maximum allowed temperature (default: 85°C)
        warning_temp: Warning temperature threshold (default: 80°C)
        target_temp: Target temperature (default: 70°C)
    """

    def __init__(
        self,
        max_temp: int = 85,
        warning_temp: int = 80,
        target_temp: int = 70,
    ):
        """Initialize thermal manager.

        Args:
            max_temp: Maximum allowed temperature (°C)
            warning_temp: Warning temperature threshold (°C)
            target_temp: Target operating temperature (°C)
        """
        self.policy = ThermalPolicy(
            max_temp=max_temp,
            warning_temp=warning_temp,
            target_temp=target_temp,
        )

        self.gpu_handle: Optional[Any] = None
        self.history = ThermalHistory()
        self.cooldown = CooldownManager()

        self.throttle_count = 0
        self.throttle_start_time: Optional[float] = None

        # Initialize NVML
        self._init_nvml()

        logger.info(
            f"ThermalManager initialized: "
            f"max={max_temp}°C, warning={warning_temp}°C, target={target_temp}°C"
        )

    def _init_nvml(self) -> None:
        """Initialize NVML for GPU monitoring."""
        if not PYNVML_AVAILABLE:
            logger.warning("NVML not available, GPU monitoring disabled")
            return

        try:
            import pynvml

            pynvml.nvmlInit()
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            logger.info("NVML initialized successfully")
        except Exception as e:
            logger.warning(f"NVML initialization failed: {e}")
            self.gpu_handle = None

    @property
    def max_temp(self) -> int:
        """Get maximum temperature."""
        return self.policy.max_temp

    @property
    def warning_temp(self) -> int:
        """Get warning temperature."""
        return self.policy.warning_temp

    @property
    def target_temp(self) -> int:
        """Get target temperature."""
        return self.policy.target_temp

    def get_current_temperature(self) -> int:
        """
        Get current GPU temperature.

        Returns:
            Current temperature in °C, or 50 if unavailable
        """
        if self.gpu_handle is None:
            return 50  # Default value

        try:
            import pynvml

            temp = pynvml.nvmlDeviceGetTemperature(self.gpu_handle, pynvml.NVML_TEMPERATURE_GPU)
            return temp
        except Exception as e:
            logger.warning(f"Failed to read temperature: {e}")
            return 50  # Default value on error

    def record_temperature(self) -> int:
        """
        Record current temperature to history.

        Returns:
            Current temperature
        """
        temp = self.get_current_temperature()
        self.history.record(temp)
        return temp

    def adjust_batch_size(
        self,
        base_batch_size: int,
        temp: Optional[int] = None,
        current_batch_size: Optional[int] = None,
    ) -> int:
        """
        Adjust batch size based on temperature.

        Args:
            base_batch_size: Original batch size
            temp: Current temperature (None to auto-detect)
            current_batch_size: Current batch size for gradual recovery

        Returns:
            Adjusted batch size (0 if critical temperature)
        """
        if temp is None:
            temp = self.get_current_temperature()

        # Critical temperature: stop processing
        if temp >= self.policy.max_temp:
            logger.critical(
                f"Temperature {temp}°C >= {self.policy.max_temp}°C, stopping processing"
            )
            self.throttle_count += 1
            self.throttle_start_time = time.time()
            self.cooldown.enter_cooldown()
            return 0

        # Warning temperature: reduce batch size by 50%
        if temp >= self.policy.warning_temp:
            new_size = base_batch_size // 2
            logger.warning(
                f"Temperature {temp}°C >= {self.policy.warning_temp}°C, "
                f"reducing batch to {new_size}"
            )
            self.throttle_count += 1
            if self.throttle_start_time is None:
                self.throttle_start_time = time.time()
            return new_size

        # Target temperature exceeded: reduce batch size by 25%
        if temp >= self.policy.target_temp:
            new_size = int(base_batch_size * 0.75)
            logger.info(
                f"Temperature {temp}°C >= {self.policy.target_temp}°C, reducing batch to {new_size}"
            )
            return new_size

        # Normal temperature: maintain full batch
        # Exit cooldown if applicable
        if self.cooldown.can_exit_cooldown(temp, self.policy.target_temp):
            self.cooldown.exit_cooldown()

        return base_batch_size

    def adjust_fan_speed(self, target_temp: int, current_temp: int) -> None:
        """
        Adjust fan speed based on temperature.

        Args:
            target_temp: Target temperature
            current_temp: Current temperature
        """
        # Fan speed control would be implemented here
        # This is a placeholder for actual fan control logic
        pass

    def is_in_cooldown(self) -> bool:
        """Check if currently in cooldown mode."""
        return self.cooldown.is_in_cooldown()

    def enter_cooldown(self) -> None:
        """Manually enter cooldown mode."""
        self.cooldown.enter_cooldown()

    def check_cooldown_recovery(self) -> bool:
        """Check if cooldown can be exited."""
        current_temp = self.get_current_temperature()
        return self.cooldown.can_exit_cooldown(current_temp, self.policy.target_temp)

    def get_temperature_history(self) -> List[int]:
        """Get temperature history."""
        return self.history.get_all()

    def get_thermal_stats(self) -> Dict[str, Any]:
        """Get thermal statistics."""
        return {
            "current_temp": self.get_current_temperature(),
            "max_temp_observed": self.history.get_max(),
            "min_temp_observed": self.history.get_min(),
            "avg_temp": self.history.get_average(),
            "throttle_count": self.throttle_count,
            "in_cooldown": self.is_in_cooldown(),
            "policy": {
                "max_temp": self.policy.max_temp,
                "warning_temp": self.policy.warning_temp,
                "target_temp": self.policy.target_temp,
            },
        }

    def calculate_throttle_percentage(self, original_batch: int, throttled_batch: int) -> float:
        """Calculate throttle percentage."""
        if original_batch == 0:
            return 0.0
        return ((original_batch - throttled_batch) / original_batch) * 100.0

    def calculate_average_temperature(self, temps: List[int]) -> float:
        """Calculate average temperature."""
        if not temps:
            return 0.0
        return sum(temps) / len(temps)

    def get_throttle_duration(self) -> float:
        """Get duration of current throttle period."""
        if self.throttle_start_time is None:
            return 0.0
        return time.time() - self.throttle_start_time
