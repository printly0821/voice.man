"""
Comprehensive Memory Manager Module

Provides advanced memory management for voice analysis pipeline with:
- Pre-allocation memory checks
- Per-file memory tracking
- Mid-processing monitoring with watchdog
- Complete model cleanup for all services
- Memory pressure prediction

Integrates with existing checkpoint system for fault tolerance.
"""

import gc
import logging
import threading
import time
import psutil
from abc import ABC, abstractmethod
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union

logger = logging.getLogger(__name__)


class MemoryPressureLevel(Enum):
    """Memory pressure severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class FileMemoryStats:
    """
    Memory statistics for a single file processing operation.

    Attributes:
        file_path: Path to the processed file
        file_size_mb: File size in megabytes
        start_memory_mb: Memory usage at start (MB)
        end_memory_mb: Memory usage at end (MB)
        peak_memory_mb: Peak memory usage during processing (MB)
        delta_mb: Memory delta (end - start) in MB
        processing_time_sec: Processing time in seconds
        success: Whether processing succeeded
        timestamp: When the file was processed
        stage: Processing stage (stt, ser, forensic, etc.)
    """

    file_path: str
    file_size_mb: float
    start_memory_mb: float
    end_memory_mb: float
    peak_memory_mb: float
    delta_mb: float
    processing_time_sec: float
    success: bool
    timestamp: datetime
    stage: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "file_path": self.file_path,
            "file_size_mb": round(self.file_size_mb, 2),
            "start_memory_mb": round(self.start_memory_mb, 2),
            "end_memory_mb": round(self.end_memory_mb, 2),
            "peak_memory_mb": round(self.peak_memory_mb, 2),
            "delta_mb": round(self.delta_mb, 2),
            "processing_time_sec": round(self.processing_time_sec, 2),
            "success": self.success,
            "timestamp": self.timestamp.isoformat(),
            "stage": self.stage,
        }


@dataclass
class MemoryPressureStatus:
    """
    Current memory pressure status.

    Attributes:
        level: Memory pressure level
        system_memory_percent: System memory usage percentage
        gpu_memory_percent: GPU memory usage percentage (if available)
        available_mb: Available memory in MB
        predicted_oom: Whether OOM is predicted
        recommended_action: Recommended action to take
        timestamp: When the status was captured
    """

    level: MemoryPressureLevel
    system_memory_percent: float
    gpu_memory_percent: Optional[float]
    available_mb: float
    predicted_oom: bool
    recommended_action: str
    timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "level": self.level.value,
            "system_memory_percent": round(self.system_memory_percent, 2),
            "gpu_memory_percent": round(self.gpu_memory_percent, 2)
            if self.gpu_memory_percent
            else None,
            "available_mb": round(self.available_mb, 2),
            "predicted_oom": self.predicted_oom,
            "recommended_action": self.recommended_action,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class PredictionResult:
    """
    Memory pressure prediction result.

    Attributes:
        file_size_mb: File size used for prediction
        predicted_memory_mb: Predicted memory usage in MB
        pressure_level: Predicted pressure level
        confidence: Prediction confidence (0-1)
        safe_to_process: Whether it's safe to process this file
        reason: Explanation of the prediction
    """

    file_size_mb: float
    predicted_memory_mb: float
    pressure_level: MemoryPressureLevel
    confidence: float
    safe_to_process: bool
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "file_size_mb": round(self.file_size_mb, 2),
            "predicted_memory_mb": round(self.predicted_memory_mb, 2),
            "pressure_level": self.pressure_level.value,
            "confidence": round(self.confidence, 3),
            "safe_to_process": self.safe_to_process,
            "reason": self.reason,
        }


class ServiceCleanupProtocol(Protocol):
    """
    Protocol for service cleanup interfaces.

    All services that need cleanup should implement this protocol.
    """

    def cleanup(self) -> None:
        """Release all resources and clear memory."""
        ...

    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        ...


class MemoryPredictor:
    """
    Memory usage predictor based on historical data.

    Uses linear regression and statistical analysis to predict
    memory usage for file processing.
    """

    def __init__(self, history_size: int = 100):
        """
        Initialize predictor.

        Args:
            history_size: Maximum number of historical records to keep
        """
        self.history_size = history_size
        self._history: deque = deque(maxlen=history_size)
        self._lock = threading.Lock()

    def add_record(self, file_size_mb: float, memory_used_mb: float) -> None:
        """
        Add a historical record.

        Args:
            file_size_mb: File size in MB
            memory_used_mb: Actual memory used in MB
        """
        with self._lock:
            self._history.append(
                {
                    "file_size_mb": file_size_mb,
                    "memory_used_mb": memory_used_mb,
                    "timestamp": datetime.now(timezone.utc),
                }
            )

    def predict(self, file_size_mb: float) -> PredictionResult:
        """
        Predict memory usage for a file.

        Args:
            file_size_mb: File size in MB

        Returns:
            PredictionResult with prediction details
        """
        with self._lock:
            if len(self._history) < 3:
                # Not enough data, use conservative estimate
                predicted_mb = file_size_mb * 5  # 5x file size as base estimate
                return PredictionResult(
                    file_size_mb=file_size_mb,
                    predicted_memory_mb=predicted_mb,
                    pressure_level=MemoryPressureLevel.LOW,
                    confidence=0.3,
                    safe_to_process=True,
                    reason="Insufficient historical data, using conservative 5x estimate",
                )

            # Calculate linear regression
            file_sizes = [record["file_size_mb"] for record in self._history]
            memory_usages = [record["memory_used_mb"] for record in self._history]

            # Simple linear regression: y = mx + b
            n = len(file_sizes)
            sum_x = sum(file_sizes)
            sum_y = sum(memory_usages)
            sum_xy = sum(fs * mu for fs, mu in zip(file_sizes, memory_usages))
            sum_x2 = sum(fs**2 for fs in file_sizes)

            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
            intercept = (sum_y - slope * sum_x) / n

            # Predict memory usage
            predicted_mb = slope * file_size_mb + intercept
            predicted_mb = max(predicted_mb, 0)  # Ensure non-negative

            # Calculate R-squared for confidence
            mean_y = sum_y / n
            ss_tot = sum((mu - mean_y) ** 2 for mu in memory_usages)
            ss_res = sum(
                (mu - (slope * fs + intercept)) ** 2 for fs, mu in zip(file_sizes, memory_usages)
            )
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            confidence = max(0, min(1, r_squared))

            # Determine pressure level
            current_memory = self._get_current_memory_mb()
            total_memory = psutil.virtual_memory().total / (1024 * 1024)
            predicted_percent = ((current_memory + predicted_mb) / total_memory) * 100

            if predicted_percent > 95:
                pressure_level = MemoryPressureLevel.CRITICAL
                safe_to_process = False
                reason = (
                    f"Predicted memory usage ({predicted_mb:.1f}MB) would exceed 95% system memory"
                )
            elif predicted_percent > 85:
                pressure_level = MemoryPressureLevel.HIGH
                safe_to_process = True
                reason = (
                    f"High memory pressure predicted: {predicted_percent:.1f}% of system memory"
                )
            elif predicted_percent > 70:
                pressure_level = MemoryPressureLevel.MEDIUM
                safe_to_process = True
                reason = (
                    f"Medium memory pressure predicted: {predicted_percent:.1f}% of system memory"
                )
            else:
                pressure_level = MemoryPressureLevel.LOW
                safe_to_process = True
                reason = f"Low memory pressure predicted: {predicted_percent:.1f}% of system memory"

            return PredictionResult(
                file_size_mb=file_size_mb,
                predicted_memory_mb=predicted_mb,
                pressure_level=pressure_level,
                confidence=confidence,
                safe_to_process=safe_to_process,
                reason=reason,
            )

    def _get_current_memory_mb(self) -> float:
        """Get current process memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)


class MemoryManager:
    """
    Comprehensive memory manager for voice analysis pipeline.

    Features:
    - Pre-allocation memory verification
    - Per-file memory tracking
    - Mid-processing monitoring with watchdog
    - Complete model cleanup for all services
    - Memory pressure prediction

    Integrates with checkpoint system for fault tolerance.
    """

    # Default thresholds
    DEFAULT_SYSTEM_MEMORY_THRESHOLD = 85.0  # Percentage
    DEFAULT_GPU_MEMORY_THRESHOLD = 90.0  # Percentage
    DEFAULT_CRITICAL_THRESHOLD = 95.0  # Percentage

    # Monitoring intervals
    DEFAULT_WATCHDOG_INTERVAL_SEC = 5.0
    DEFAULT_MONITORING_INTERVAL_SEC = 1.0

    def __init__(
        self,
        system_memory_threshold: float = DEFAULT_SYSTEM_MEMORY_THRESHOLD,
        gpu_memory_threshold: float = DEFAULT_GPU_MEMORY_THRESHOLD,
        critical_threshold: float = DEFAULT_CRITICAL_THRESHOLD,
        watchdog_interval_sec: float = DEFAULT_WATCHDOG_INTERVAL_SEC,
        monitoring_interval_sec: float = DEFAULT_MONITORING_INTERVAL_SEC,
        enable_gpu_monitoring: bool = True,
    ):
        """
        Initialize MemoryManager.

        Args:
            system_memory_threshold: System memory threshold percentage
            gpu_memory_threshold: GPU memory threshold percentage
            critical_threshold: Critical memory threshold percentage
            watchdog_interval_sec: Watchdog check interval in seconds
            monitoring_interval_sec: Memory monitoring interval in seconds
            enable_gpu_monitoring: Whether to enable GPU memory monitoring
        """
        self.system_memory_threshold = system_memory_threshold
        self.gpu_memory_threshold = gpu_memory_threshold
        self.critical_threshold = critical_threshold
        self.watchdog_interval_sec = watchdog_interval_sec
        self.monitoring_interval_sec = monitoring_interval_sec
        self.enable_gpu_monitoring = enable_gpu_monitoring

        # Initialize predictor
        self.predictor = MemoryPredictor()

        # Tracking state
        self._current_file_stats: Optional[FileMemoryStats] = None
        self._file_history: List[FileMemoryStats] = []
        self._lock = threading.Lock()

        # Monitoring state
        self._monitoring_active = False
        self._watchdog_active = False
        self._monitoring_thread: Optional[threading.Thread] = None
        self._watchdog_thread: Optional[threading.Thread] = None

        # Service registry for cleanup
        self._services: Dict[str, ServiceCleanupProtocol] = {}

        # Check GPU availability
        self._gpu_available = self._check_gpu_available()

        logger.info(
            f"MemoryManager initialized: "
            f"system_threshold={system_memory_threshold}%, "
            f"gpu_threshold={gpu_memory_threshold}%, "
            f"critical_threshold={critical_threshold}%, "
            f"gpu_available={self._gpu_available}"
        )

    def _check_gpu_available(self) -> bool:
        """Check if GPU is available for monitoring."""
        if not self.enable_gpu_monitoring:
            return False

        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            pass
        except Exception:
            pass

        try:
            import pynvml

            pynvml.nvmlInit()
            pynvml.nvmlShutdown()
            return True
        except ImportError:
            pass
        except Exception:
            pass

        return False

    def register_service(self, name: str, service: ServiceCleanupProtocol) -> None:
        """
        Register a service for memory cleanup.

        Args:
            name: Service name
            service: Service implementing cleanup protocol
        """
        with self._lock:
            self._services[name] = service
            logger.debug(f"Registered service for cleanup: {name}")

    def unregister_service(self, name: str) -> None:
        """
        Unregister a service from cleanup.

        Args:
            name: Service name
        """
        with self._lock:
            if name in self._services:
                del self._services[name]
                logger.debug(f"Unregistered service: {name}")

    def check_pre_allocation(
        self,
        batch_size: int,
        estimated_per_file_mb: float,
        safety_margin: float = 1.2,
    ) -> Tuple[bool, str]:
        """
        Check if sufficient memory is available before starting batch processing.

        Args:
            batch_size: Number of files to process
            estimated_per_file_mb: Estimated memory per file in MB
            safety_margin: Safety margin multiplier (default 1.2 = 20% buffer)

        Returns:
            Tuple of (is_safe, message)
        """
        current_memory_mb = self._get_process_memory_mb()
        total_memory_mb = psutil.virtual_memory().total / (1024 * 1024)
        available_memory_mb = psutil.virtual_memory().available / (1024 * 1024)

        # Estimate total memory needed with safety margin
        estimated_total_mb = batch_size * estimated_per_file_mb * safety_margin
        predicted_usage_mb = current_memory_mb + estimated_total_mb
        predicted_percent = (predicted_usage_mb / total_memory_mb) * 100

        logger.info(
            f"Pre-allocation check: "
            f"current={current_memory_mb:.1f}MB, "
            f"available={available_memory_mb:.1f}MB, "
            f"estimated_needed={estimated_total_mb:.1f}MB, "
            f"predicted_percent={predicted_percent:.1f}%"
        )

        if predicted_percent > self.critical_threshold:
            return False, (
                f"Insufficient memory: predicted {predicted_percent:.1f}% usage "
                f"exceeds critical threshold {self.critical_threshold}%. "
                f"Reduce batch size or free memory."
            )

        if predicted_percent > self.system_memory_threshold:
            return True, (
                f"Memory will be high: predicted {predicted_percent:.1f}% usage. "
                f"Consider reducing batch size for optimal performance."
            )

        return True, (
            f"Sufficient memory: predicted {predicted_percent:.1f}% usage. "
            f"Safe to proceed with batch processing."
        )

    def track_file_start(self, file_path: str, stage: str = "unknown") -> float:
        """
        Record memory usage before processing a file.

        Args:
            file_path: Path to the file being processed
            stage: Processing stage identifier

        Returns:
            Starting memory usage in MB
        """
        file_size_mb = self._get_file_size_mb(file_path)
        start_memory_mb = self._get_process_memory_mb()

        with self._lock:
            self._current_file_stats = {
                "file_path": file_path,
                "file_size_mb": file_size_mb,
                "start_memory_mb": start_memory_mb,
                "start_time": time.time(),
                "stage": stage,
            }

        logger.debug(
            f"File start tracking: {file_path} "
            f"(size={file_size_mb:.2f}MB, memory={start_memory_mb:.2f}MB, stage={stage})"
        )

        return start_memory_mb

    def track_file_end(
        self,
        file_path: str,
        success: bool = True,
    ) -> FileMemoryStats:
        """
        Record memory usage after processing a file.

        Args:
            file_path: Path to the processed file
            success: Whether processing succeeded

        Returns:
            FileMemoryStats with complete tracking information
        """
        with self._lock:
            if self._current_file_stats is None:
                logger.warning(f"No start tracking found for {file_path}, creating placeholder")
                start_memory_mb = self._get_process_memory_mb()
                file_size_mb = self._get_file_size_mb(file_path)
                start_time = time.time()
                stage = "unknown"
            else:
                start_memory_mb = self._current_file_stats["start_memory_mb"]
                file_size_mb = self._current_file_stats["file_size_mb"]
                start_time = self._current_file_stats["start_time"]
                stage = self._current_file_stats["stage"]

        end_memory_mb = self._get_process_memory_mb()
        processing_time = time.time() - start_time
        delta_mb = end_memory_mb - start_memory_mb
        peak_memory_mb = max(start_memory_mb, end_memory_mb)

        stats = FileMemoryStats(
            file_path=file_path,
            file_size_mb=file_size_mb,
            start_memory_mb=start_memory_mb,
            end_memory_mb=end_memory_mb,
            peak_memory_mb=peak_memory_mb,
            delta_mb=delta_mb,
            processing_time_sec=processing_time,
            success=success,
            timestamp=datetime.now(timezone.utc),
            stage=stage,
        )

        with self._lock:
            self._file_history.append(stats)
            self._current_file_stats = None

            # Add to predictor for future predictions
            if success:
                self.predictor.add_record(file_size_mb, delta_mb)

        logger.debug(
            f"File end tracking: {file_path} "
            f"(delta={delta_mb:+.2f}MB, time={processing_time:.2f}s, success={success})"
        )

        return stats

    def check_during_processing(self) -> MemoryPressureStatus:
        """
        Check memory pressure during processing.

        Returns:
            MemoryPressureStatus with current pressure information
        """
        system_memory = psutil.virtual_memory()
        system_percent = system_memory.percent
        available_mb = system_memory.available / (1024 * 1024)

        # Get GPU memory if available
        gpu_percent = None
        if self._gpu_available:
            gpu_percent = self._get_gpu_memory_percent()

        # Determine pressure level
        if system_percent >= self.critical_threshold:
            level = MemoryPressureLevel.CRITICAL
            predicted_oom = True
            action = "IMMEDIATE ACTION REQUIRED: Stop processing and run cleanup"
        elif system_percent >= self.system_memory_threshold:
            level = MemoryPressureLevel.HIGH
            predicted_oom = False
            action = "Consider pausing batch and running garbage collection"
        elif system_percent >= (self.system_memory_threshold - 10):
            level = MemoryPressureLevel.MEDIUM
            predicted_oom = False
            action = "Monitor closely, prepare for cleanup if needed"
        else:
            level = MemoryPressureLevel.LOW
            predicted_oom = False
            action = "Normal operation, continue processing"

        status = MemoryPressureStatus(
            level=level,
            system_memory_percent=system_percent,
            gpu_memory_percent=gpu_percent,
            available_mb=available_mb,
            predicted_oom=predicted_oom,
            recommended_action=action,
            timestamp=datetime.now(timezone.utc),
        )

        return status

    def predict_memory_pressure(self, file_size_mb: float) -> PredictionResult:
        """
        Predict memory pressure for processing a file.

        Args:
            file_size_mb: File size in MB

        Returns:
            PredictionResult with prediction details
        """
        return self.predictor.predict(file_size_mb)

    def complete_cleanup(self) -> Dict[str, Any]:
        """
        Perform complete cleanup of all registered services.

        This includes:
        - All registered services (STT, SER, forensic, crime, NLP, etc.)
        - GPU memory cache clearing
        - Python garbage collection
        - System memory cache clearing

        Returns:
            Dictionary with cleanup statistics
        """
        cleanup_stats = {
            "services_cleaned": 0,
            "services_failed": 0,
            "memory_before_mb": self._get_process_memory_mb(),
            "gpu_cleared": False,
            "gc_collected": 0,
        }

        logger.info("Starting complete cleanup...")

        # Cleanup all registered services
        with self._lock:
            services_to_cleanup = list(self._services.items())

        for name, service in services_to_cleanup:
            try:
                logger.debug(f"Cleaning up service: {name}")
                service.cleanup()
                cleanup_stats["services_cleaned"] += 1
            except Exception as e:
                logger.error(f"Failed to cleanup service {name}: {e}")
                cleanup_stats["services_failed"] += 1

        # Clear GPU cache
        if self._gpu_available:
            try:
                self._clear_gpu_cache()
                cleanup_stats["gpu_cleared"] = True
            except Exception as e:
                logger.error(f"Failed to clear GPU cache: {e}")

        # Force garbage collection
        collected = gc.collect()
        cleanup_stats["gc_collected"] = collected

        # Final memory measurement
        cleanup_stats["memory_after_mb"] = self._get_process_memory_mb()
        cleanup_stats["memory_freed_mb"] = (
            cleanup_stats["memory_before_mb"] - cleanup_stats["memory_after_mb"]
        )

        logger.info(
            f"Cleanup complete: "
            f"{cleanup_stats['services_cleaned']} services cleaned, "
            f"{cleanup_stats['gc_collected']} objects collected, "
            f"{cleanup_stats['memory_freed_mb']:.2f}MB freed"
        )

        return cleanup_stats

    def start_monitoring(self) -> None:
        """Start background memory monitoring thread."""
        with self._lock:
            if self._monitoring_active:
                logger.warning("Monitoring already active")
                return

            self._monitoring_active = True
            self._monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                daemon=True,
                name="MemoryMonitor",
            )
            self._monitoring_thread.start()

        logger.info("Memory monitoring started")

    def stop_monitoring(self) -> None:
        """Stop background memory monitoring thread."""
        with self._lock:
            self._monitoring_active = False

        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5.0)
            self._monitoring_thread = None

        logger.info("Memory monitoring stopped")

    def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while self._monitoring_active:
            try:
                status = self.check_during_processing()

                # Log if pressure is elevated
                if status.level in (MemoryPressureLevel.HIGH, MemoryPressureLevel.CRITICAL):
                    logger.warning(
                        f"High memory pressure detected: {status.system_memory_percent:.1f}% - "
                        f"{status.recommended_action}"
                    )

                # Auto-cleanup on critical pressure
                if status.level == MemoryPressureLevel.CRITICAL:
                    logger.critical("Critical memory pressure, triggering emergency cleanup")
                    self.complete_cleanup()

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")

            time.sleep(self.monitoring_interval_sec)

    def _get_process_memory_mb(self) -> float:
        """Get current process memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except Exception as e:
            logger.error(f"Failed to get process memory: {e}")
            return 0.0

    def _get_file_size_mb(self, file_path: str) -> float:
        """Get file size in MB."""
        try:
            return Path(file_path).stat().st_size / (1024 * 1024)
        except Exception as e:
            logger.warning(f"Failed to get file size for {file_path}: {e}")
            return 0.0

    def _get_gpu_memory_percent(self) -> Optional[float]:
        """Get GPU memory usage percentage."""
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            pynvml.nvmlShutdown()

            return (mem_info.used / mem_info.total) * 100
        except Exception:
            pass

        try:
            import torch

            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0)
                total = torch.cuda.get_device_properties(0).total_memory
                return (allocated / total) * 100
        except Exception:
            pass

        return None

    def _clear_gpu_cache(self) -> None:
        """Clear GPU memory cache."""
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()
        except Exception as e:
            logger.error(f"Failed to clear GPU cache: {e}")

    def get_file_history(self) -> List[FileMemoryStats]:
        """
        Get history of file processing statistics.

        Returns:
            List of FileMemoryStats
        """
        with self._lock:
            return list(self._file_history)

    def get_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive memory manager summary.

        Returns:
            Dictionary with memory manager statistics
        """
        current_memory_mb = self._get_process_memory_mb()
        system_memory = psutil.virtual_memory()

        summary = {
            "current_memory_mb": round(current_memory_mb, 2),
            "system_memory_percent": round(system_memory.percent, 2),
            "system_memory_available_mb": round(system_memory.available / (1024 * 1024), 2),
            "thresholds": {
                "system": self.system_memory_threshold,
                "gpu": self.gpu_memory_threshold,
                "critical": self.critical_threshold,
            },
            "gpu_available": self._gpu_available,
            "monitoring_active": self._monitoring_active,
            "registered_services": list(self._services.keys()),
            "file_history_count": len(self._file_history),
            "predictor_history_size": len(self.predictor._history),
        }

        if self._gpu_available:
            gpu_percent = self._get_gpu_memory_percent()
            if gpu_percent is not None:
                summary["gpu_memory_percent"] = round(gpu_percent, 2)

        return summary

    @contextmanager
    def file_context(
        self,
        file_path: str,
        stage: str = "unknown",
        auto_cleanup: bool = True,
    ):
        """
        Context manager for file processing with automatic memory tracking.

        Args:
            file_path: Path to the file being processed
            stage: Processing stage identifier
            auto_cleanup: Whether to run cleanup after processing

        Yields:
            FileMemoryStats object (will be updated after processing)

        Example:
            with manager.file_context("audio.m4a", stage="stt") as stats:
                # Process file here
                result = process_audio("audio.m4a")
                # stats will be updated automatically on exit
        """
        self.track_file_start(file_path, stage)
        success = False

        try:
            yield
            success = True
        finally:
            _stats = self.track_file_end(file_path, success)

            if auto_cleanup:
                # Run cleanup if memory pressure is high
                pressure = self.check_during_processing()
                if pressure.level in (MemoryPressureLevel.HIGH, MemoryPressureLevel.CRITICAL):
                    logger.info(f"Running auto-cleanup after processing {file_path}")
                    self.complete_cleanup()

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.stop_monitoring()
        except Exception:
            pass
