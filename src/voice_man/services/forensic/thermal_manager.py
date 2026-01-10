"""
ThermalManager - GPU thermal management for forensic pipeline.

SPEC-PERFOPT-001 Phase 2: Temperature monitoring and throttling for GPU workloads.

Thresholds:
    - THROTTLE_START: 80C - Begin throttling operations
    - THROTTLE_STOP: 70C - Resume normal operations (hysteresis)
    - CRITICAL: 85C - Critical temperature, halt operations

Features:
    - pynvml GPU temperature monitoring
    - Background monitoring thread with configurable interval
    - Hysteresis-based throttling to prevent oscillation
    - Callback registration for throttling events
    - Context manager for automatic monitoring lifecycle
"""

import gc
import logging
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, List, Optional

logger = logging.getLogger(__name__)

# Try to import pynvml for GPU temperature monitoring
try:
    import pynvml

    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    logger.warning("pynvml not available, GPU temperature monitoring will be limited")

# Default temperature thresholds (Celsius)
THROTTLE_START_TEMP = 80  # Start throttling
THROTTLE_STOP_TEMP = 70  # Stop throttling (hysteresis)
CRITICAL_TEMP = 85  # Critical - halt operations

# Default temperature when GPU unavailable
DEFAULT_TEMP = 50


class ThermalManager:
    """
    GPU thermal manager for forensic pipeline.

    Monitors GPU temperature and triggers throttling callbacks when
    temperature exceeds thresholds. Uses hysteresis to prevent
    rapid throttle on/off oscillation.

    Example:
        manager = ThermalManager()

        def on_throttle(is_throttling: bool):
            if is_throttling:
                # Reduce batch size
                pass
            else:
                # Resume normal batch size
                pass

        manager.register_throttle_callback(on_throttle)
        manager.start_monitoring(interval_seconds=1.0)

        # ... processing ...

        manager.stop_monitoring()
    """

    def __init__(
        self,
        throttle_start_temp: int = THROTTLE_START_TEMP,
        throttle_stop_temp: int = THROTTLE_STOP_TEMP,
        critical_temp: int = CRITICAL_TEMP,
        device_index: int = 0,
    ):
        """
        Initialize ThermalManager.

        Args:
            throttle_start_temp: Temperature to start throttling (default: 80C)
            throttle_stop_temp: Temperature to stop throttling (default: 70C)
            critical_temp: Critical temperature threshold (default: 85C)
            device_index: GPU device index for monitoring.

        Raises:
            ValueError: If throttle_stop_temp >= throttle_start_temp
        """
        if throttle_stop_temp >= throttle_start_temp:
            raise ValueError(
                f"throttle_stop_temp ({throttle_stop_temp}) must be < "
                f"throttle_start_temp ({throttle_start_temp})"
            )

        self.throttle_start_temp = throttle_start_temp
        self.throttle_stop_temp = throttle_stop_temp
        self.critical_temp = critical_temp
        self.device_index = device_index

        # State
        self._is_throttling = False
        self._is_critical = False
        self._is_monitoring = False

        # Callbacks
        self._throttle_callbacks: List[Callable[[bool], None]] = []
        self._critical_callbacks: List[Callable[[], None]] = []

        # Temperature history
        self._temp_history: List[int] = []
        self._max_history_size = 1000

        # Monitoring thread
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        # GPU handle
        self._gpu_handle: Optional[Any] = None
        self._init_pynvml()

        logger.info(
            f"ThermalManager initialized: start={throttle_start_temp}C, "
            f"stop={throttle_stop_temp}C, critical={critical_temp}C"
        )

    def _init_pynvml(self) -> None:
        """Initialize pynvml for GPU temperature monitoring."""
        if not PYNVML_AVAILABLE:
            return

        try:
            pynvml.nvmlInit()
            self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            logger.debug(f"pynvml initialized for device {self.device_index}")
        except Exception as e:
            logger.warning(f"Failed to initialize pynvml: {e}")
            self._gpu_handle = None

    @property
    def is_throttling(self) -> bool:
        """Check if currently throttling."""
        return self._is_throttling

    @property
    def is_critical(self) -> bool:
        """Check if critical temperature reached."""
        return self._is_critical

    @property
    def is_monitoring(self) -> bool:
        """Check if background monitoring is active."""
        return self._is_monitoring

    def get_current_temperature(self) -> int:
        """
        Get current GPU temperature.

        Returns:
            Temperature in Celsius, or DEFAULT_TEMP if unavailable.
        """
        if not PYNVML_AVAILABLE or self._gpu_handle is None:
            return DEFAULT_TEMP

        try:
            temp = pynvml.nvmlDeviceGetTemperature(self._gpu_handle, pynvml.NVML_TEMPERATURE_GPU)
            return temp
        except Exception as e:
            logger.warning(f"Failed to read GPU temperature: {e}")
            return DEFAULT_TEMP

    def check_temperature(self, current_temp: Optional[int] = None) -> None:
        """
        Check temperature and update throttling state.

        Args:
            current_temp: Temperature to check. If None, reads from GPU.
        """
        if current_temp is None:
            current_temp = self.get_current_temperature()

        # Record history
        self._record_temperature(current_temp)

        previous_throttling = self._is_throttling
        previous_critical = self._is_critical

        # Check critical temperature
        if current_temp >= self.critical_temp:
            self._is_critical = True
            if not self._is_throttling:
                self._is_throttling = True
            if not previous_critical:
                self._invoke_critical_callbacks()
                logger.critical(f"Critical temperature reached: {current_temp}C")
        else:
            self._is_critical = False

        # Hysteresis-based throttling
        if not self._is_critical:
            if current_temp >= self.throttle_start_temp:
                self._is_throttling = True
            elif current_temp < self.throttle_stop_temp:
                self._is_throttling = False
            # Between stop and start thresholds: maintain current state

        # Invoke callbacks if state changed
        if self._is_throttling != previous_throttling:
            self._invoke_throttle_callbacks(self._is_throttling)
            if self._is_throttling:
                logger.warning(f"Throttling started at {current_temp}C")
            else:
                logger.info(f"Throttling stopped at {current_temp}C")

    def _record_temperature(self, temp: int) -> None:
        """Record temperature to history."""
        self._temp_history.append(temp)
        if len(self._temp_history) > self._max_history_size:
            self._temp_history = self._temp_history[-self._max_history_size :]

    def register_throttle_callback(self, callback: Callable[[bool], None]) -> None:
        """
        Register callback for throttle state changes.

        Args:
            callback: Function called with is_throttling boolean.
        """
        with self._lock:
            self._throttle_callbacks.append(callback)
        logger.debug("Throttle callback registered")

    def unregister_throttle_callback(self, callback: Callable[[bool], None]) -> None:
        """
        Unregister a throttle callback.

        Args:
            callback: Callback function to remove.
        """
        with self._lock:
            if callback in self._throttle_callbacks:
                self._throttle_callbacks.remove(callback)
        logger.debug("Throttle callback unregistered")

    def register_critical_callback(self, callback: Callable[[], None]) -> None:
        """
        Register callback for critical temperature events.

        Args:
            callback: Function called when critical temperature reached.
        """
        with self._lock:
            self._critical_callbacks.append(callback)
        logger.debug("Critical callback registered")

    def _invoke_throttle_callbacks(self, is_throttling: bool) -> None:
        """Invoke all registered throttle callbacks."""
        with self._lock:
            callbacks = self._throttle_callbacks.copy()

        for callback in callbacks:
            try:
                callback(is_throttling)
            except Exception as e:
                logger.error(f"Error in throttle callback: {e}")

    def _invoke_critical_callbacks(self) -> None:
        """Invoke all registered critical callbacks."""
        with self._lock:
            callbacks = self._critical_callbacks.copy()

        for callback in callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Error in critical callback: {e}")

    def start_monitoring(self, interval_seconds: float = 1.0) -> None:
        """
        Start background temperature monitoring.

        Args:
            interval_seconds: Monitoring interval in seconds.
        """
        if self._is_monitoring:
            logger.warning("Monitoring already active")
            return

        self._stop_event.clear()
        self._is_monitoring = True

        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_seconds,),
            daemon=True,
            name="ThermalMonitor",
        )
        self._monitor_thread.start()
        logger.info(f"Temperature monitoring started (interval: {interval_seconds}s)")

    def stop_monitoring(self) -> None:
        """Stop background temperature monitoring."""
        if not self._is_monitoring:
            return

        self._stop_event.set()
        self._is_monitoring = False

        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2.0)

        self._monitor_thread = None
        logger.info("Temperature monitoring stopped")

    def _monitoring_loop(self, interval_seconds: float) -> None:
        """Background monitoring loop."""
        while not self._stop_event.is_set():
            try:
                self.check_temperature()
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")

            self._stop_event.wait(interval_seconds)

    def get_thermal_stats(self) -> dict:
        """
        Get current thermal statistics.

        Returns:
            Dictionary with thermal information.
        """
        return {
            "current_temp": self.get_current_temperature(),
            "is_throttling": self._is_throttling,
            "is_critical": self._is_critical,
            "throttle_start_temp": self.throttle_start_temp,
            "throttle_stop_temp": self.throttle_stop_temp,
            "critical_temp": self.critical_temp,
            "max_temp": max(self._temp_history) if self._temp_history else 0,
            "avg_temp": (
                sum(self._temp_history) / len(self._temp_history) if self._temp_history else 0.0
            ),
            "is_monitoring": self._is_monitoring,
        }

    @contextmanager
    def monitoring_context(self, interval_seconds: float = 1.0):
        """
        Context manager for monitoring lifecycle.

        Args:
            interval_seconds: Monitoring interval.

        Example:
            with manager.monitoring_context(interval_seconds=0.5):
                # processing with temperature monitoring
                pass
        """
        self.start_monitoring(interval_seconds)
        try:
            yield self
        finally:
            self.stop_monitoring()

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.stop_monitoring()
            if PYNVML_AVAILABLE:
                try:
                    pynvml.nvmlShutdown()
                except Exception:
                    pass
        except Exception:
            pass
