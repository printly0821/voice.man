"""
ForensicMemoryManager - Stage-based GPU memory allocation for forensic pipeline.

SPEC-PERFOPT-001 Phase 2: GPU memory management for forensic analysis stages.
SPEC-PIPELINE-001 Phase 1: GPU memory management enhancements including:
    - Actual GPU memory reservation (not just state tracking)
    - Pre-allocation validation
    - Memory pressure monitoring (90% threshold)
    - Emergency cleanup on OOM

Stage Memory Allocations (default):
    - STT: 16GB (16384 MB)
    - Alignment: 4GB (4096 MB)
    - Diarization: 8GB (8192 MB)
    - SER: 10GB (10240 MB)
    - Scoring: 2GB (2048 MB)

Features:
    - Thread-safe memory allocation/release via threading.Lock
    - GPU memory integration via pynvml
    - Context manager support for automatic cleanup
    - Memory statistics tracking
    - Actual GPU memory reservation with PyTorch tensors
    - Memory pressure monitoring with automatic cleanup
"""

import gc
import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Try to import pynvml for GPU memory monitoring
try:
    import pynvml

    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    logger.warning("pynvml not available, GPU memory monitoring will be limited")

# Try to import torch for GPU memory reservation
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("torch not available, GPU memory reservation will be disabled")


# Default stage memory allocations in MB
DEFAULT_STAGE_ALLOCATIONS: Dict[str, int] = {
    "stt": 16384,  # 16GB
    "alignment": 4096,  # 4GB
    "diarization": 8192,  # 8GB
    "ser": 10240,  # 10GB
    "scoring": 2048,  # 2GB
}


class MemoryPressureLevel(Enum):
    """Memory pressure severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class StageAllocation:
    """Represents a memory allocation for a stage."""

    stage: str
    amount_mb: int
    allocated_at: float = 0.0


@dataclass
class GPUMemoryReservation:
    """
    Represents an actual GPU memory reservation.

    Attributes:
        stage: Stage name
        reserved_mb: Memory reserved in MB
        tensor: PyTorch tensor holding the memory
        reserved_at: Reservation timestamp
    """

    stage: str
    reserved_mb: int
    tensor: Optional[Any] = None
    reserved_at: datetime = None

    def __post_init__(self):
        if self.reserved_at is None:
            self.reserved_at = datetime.now(timezone.utc)


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


class ForensicMemoryManager:
    """
    Stage-based GPU memory manager for forensic pipeline.

    Provides thread-safe memory allocation and release for different
    stages of the forensic audio analysis pipeline.

    SPEC-PIPELINE-001 Enhanced Features:
    - Actual GPU memory reservation (not just state tracking)
    - Pre-allocation validation
    - Memory pressure monitoring (90% threshold)
    - Emergency cleanup on OOM

    Attributes:
        stage_allocations: Default memory allocations for each stage (in MB)
        gpu_memory_threshold: GPU memory threshold percentage (default: 90%)
        critical_threshold: Critical memory threshold percentage (default: 95%)

    Example:
        manager = ForensicMemoryManager()
        manager.allocate("stt")
        # ... perform STT processing ...
        manager.release("stt")

        # Or with context manager:
        with manager.stage_context("ser") as allocated:
            if allocated:
                # ... perform SER processing ...

        # NEW: Actual GPU reservation
        reservation = manager.reserve_gpu_memory("stt", 16384)
        # ... perform processing ...
        manager.release_gpu_memory("stt")
    """

    # Default thresholds
    DEFAULT_GPU_MEMORY_THRESHOLD = 90.0  # Percentage
    DEFAULT_CRITICAL_THRESHOLD = 95.0  # Percentage

    def __init__(
        self,
        stage_allocations: Optional[Dict[str, int]] = None,
        device_index: int = 0,
        gpu_memory_threshold: float = DEFAULT_GPU_MEMORY_THRESHOLD,
        critical_threshold: float = DEFAULT_CRITICAL_THRESHOLD,
    ):
        """
        Initialize ForensicMemoryManager.

        Args:
            stage_allocations: Custom stage allocations in MB. If None, uses defaults.
            device_index: GPU device index for pynvml queries.
            gpu_memory_threshold: GPU memory threshold percentage (default: 90%)
            critical_threshold: Critical memory threshold percentage (default: 95%)
        """
        self.stage_allocations = (
            stage_allocations.copy() if stage_allocations else DEFAULT_STAGE_ALLOCATIONS.copy()
        )
        self.device_index = device_index
        self.gpu_memory_threshold = gpu_memory_threshold
        self.critical_threshold = critical_threshold

        # Thread-safety lock
        self._lock = threading.Lock()

        # Track allocated stages (state tracking)
        self._allocated_stages: Dict[str, StageAllocation] = {}

        # Track GPU reservations (actual memory reservation)
        self._gpu_reservations: Dict[str, GPUMemoryReservation] = {}

        # Initialize pynvml if available
        self._gpu_handle: Optional[Any] = None
        self._init_pynvml()

        logger.info(
            f"ForensicMemoryManager initialized with stages: {list(self.stage_allocations.keys())}, "
            f"gpu_threshold={gpu_memory_threshold}%, critical_threshold={critical_threshold}%"
        )

    def _init_pynvml(self) -> None:
        """Initialize pynvml for GPU memory monitoring."""
        if not PYNVML_AVAILABLE:
            return

        try:
            pynvml.nvmlInit()
            self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            logger.debug(f"pynvml initialized for device {self.device_index}")
        except Exception as e:
            logger.warning(f"Failed to initialize pynvml: {e}")
            self._gpu_handle = None

    def allocate(self, stage: str, amount_mb: Optional[int] = None) -> bool:
        """
        Allocate memory for a stage.

        Args:
            stage: Stage name (stt, alignment, diarization, ser, scoring)
            amount_mb: Optional custom amount in MB. Uses default if None.

        Returns:
            True if allocation successful, False if already allocated.

        Raises:
            ValueError: If stage is unknown.
        """
        if stage not in self.stage_allocations:
            raise ValueError(
                f"Unknown stage: {stage}. Valid stages: {list(self.stage_allocations.keys())}"
            )

        allocation_amount = amount_mb if amount_mb is not None else self.stage_allocations[stage]

        with self._lock:
            if stage in self._allocated_stages:
                logger.warning(f"Stage '{stage}' already allocated")
                return False

            import time

            self._allocated_stages[stage] = StageAllocation(
                stage=stage,
                amount_mb=allocation_amount,
                allocated_at=time.time(),
            )

            logger.info(f"Allocated {allocation_amount} MB for stage '{stage}'")
            return True

    def release(self, stage: str) -> bool:
        """
        Release memory for a stage.

        Args:
            stage: Stage name to release.

        Returns:
            True if release successful, False if not allocated.

        Raises:
            ValueError: If stage is unknown.
        """
        if stage not in self.stage_allocations:
            raise ValueError(
                f"Unknown stage: {stage}. Valid stages: {list(self.stage_allocations.keys())}"
            )

        with self._lock:
            if stage not in self._allocated_stages:
                logger.debug(f"Stage '{stage}' not allocated, nothing to release")
                return False

            allocation = self._allocated_stages.pop(stage)
            logger.info(f"Released {allocation.amount_mb} MB from stage '{stage}'")

        # Force garbage collection and clear CUDA cache
        self._cleanup_gpu_memory()

        return True

    def release_all(self) -> None:
        """Release all allocated stages."""
        with self._lock:
            stages_to_release = list(self._allocated_stages.keys())

        for stage in stages_to_release:
            self.release(stage)

        logger.info("All stages released")

    def is_allocated(self, stage: str) -> bool:
        """
        Check if a stage is currently allocated.

        Args:
            stage: Stage name to check.

        Returns:
            True if allocated, False otherwise.
        """
        with self._lock:
            return stage in self._allocated_stages

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get current memory statistics.

        Returns:
            Dictionary with memory statistics including:
            - total_allocated_mb: Total allocated memory across all stages
            - allocated_stages: Per-stage allocation details
            - gpu_available: Whether GPU monitoring is available
            - gpu_total_mb, gpu_used_mb, gpu_free_mb: GPU memory stats (if available)
        """
        with self._lock:
            allocated_stages = {}
            total_allocated = 0

            for stage, allocation in self._allocated_stages.items():
                allocated_stages[stage] = {
                    "amount_mb": allocation.amount_mb,
                    "allocated_at": allocation.allocated_at,
                }
                total_allocated += allocation.amount_mb

        stats = {
            "total_allocated_mb": total_allocated,
            "allocated_stages": allocated_stages,
            "stage_defaults": self.stage_allocations.copy(),
        }

        # Add GPU stats if available
        gpu_stats = self._get_gpu_memory_stats()
        stats.update(gpu_stats)

        return stats

    def _get_gpu_memory_stats(self) -> Dict[str, Any]:
        """Get GPU memory statistics via pynvml."""
        if not PYNVML_AVAILABLE or self._gpu_handle is None:
            return {"gpu_available": False}

        try:
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
            return {
                "gpu_available": True,
                "gpu_total_mb": mem_info.total // (1024 * 1024),
                "gpu_used_mb": mem_info.used // (1024 * 1024),
                "gpu_free_mb": mem_info.free // (1024 * 1024),
            }
        except Exception as e:
            logger.warning(f"Failed to get GPU memory stats: {e}")
            return {"gpu_available": False}

    def _cleanup_gpu_memory(self) -> None:
        """Force garbage collection and clear GPU memory cache."""
        gc.collect()

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()
                logger.debug("GPU memory cache cleared")
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Error clearing GPU memory: {e}")

    @contextmanager
    def stage_context(self, stage: str, amount_mb: Optional[int] = None):
        """
        Context manager for stage memory allocation.

        Automatically allocates on entry and releases on exit.

        Args:
            stage: Stage name to allocate.
            amount_mb: Optional custom amount in MB.

        Yields:
            True if allocation successful, False otherwise.

        Example:
            with manager.stage_context("stt") as allocated:
                if allocated:
                    # ... process ...
        """
        allocated = self.allocate(stage, amount_mb)
        try:
            yield allocated
        finally:
            if allocated:
                self.release(stage)

    # ========== SPEC-PIPELINE-001 Enhanced GPU Memory Management ==========

    def reserve_gpu_memory(
        self, stage: str, amount_mb: int
    ) -> Tuple[bool, Optional[GPUMemoryReservation]]:
        """
        Reserve actual GPU memory using PyTorch tensors.

        SPEC-PIPELINE-001 F1: Actual GPU memory reservation (not just state tracking)
        SPEC-PIPELINE-001 U1: Always validate before allocation

        Args:
            stage: Stage name for the reservation
            amount_mb: Amount of memory to reserve in MB

        Returns:
            Tuple of (success, reservation) where:
            - success: True if reservation successful
            - reservation: GPUMemoryReservation object if successful, None otherwise
        """
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, cannot reserve GPU memory")
            return False, None

        # Validate memory availability before attempting allocation
        free_mb = self._get_gpu_free_memory_mb()
        if free_mb < amount_mb:
            logger.warning(
                f"Insufficient GPU memory for stage '{stage}': "
                f"need {amount_mb}MB, available {free_mb}MB"
            )
            return False, None

        try:
            # Create a tensor to reserve the memory
            # Use float32 (4 bytes per element)
            num_elements = (amount_mb * 1024 * 1024) // 4

            if torch.cuda.is_available():
                # Allocate on GPU
                tensor = torch.zeros(num_elements, dtype=torch.float32, device="cuda")
            else:
                # Fallback to CPU
                logger.warning("CUDA not available, allocating on CPU")
                tensor = torch.zeros(num_elements, dtype=torch.float32)

            reservation = GPUMemoryReservation(
                stage=stage,
                reserved_mb=amount_mb,
                tensor=tensor,
            )

            with self._lock:
                self._gpu_reservations[stage] = reservation

            logger.info(f"Reserved {amount_mb} MB for stage '{stage}' with tensor")
            return True, reservation

        except Exception as e:
            logger.error(f"Failed to reserve GPU memory for stage '{stage}': {e}")
            return False, None

    def release_gpu_memory(self, stage: str) -> bool:
        """
        Release GPU memory reservation for a stage.

        SPEC-PIPELINE-001 U3: 모든 서비스 정리 작업은 멱등(idempotent)하게 수행되어야 한다

        Args:
            stage: Stage name to release

        Returns:
            True if release successful, False if not reserved
        """
        with self._lock:
            if stage not in self._gpu_reservations:
                logger.debug(f"Stage '{stage}' has no GPU reservation, nothing to release")
                return False

            reservation = self._gpu_reservations.pop(stage)

        # Delete tensor to free memory
        if reservation.tensor is not None:
            del reservation.tensor
            reservation.tensor = None

        logger.info(f"Released GPU memory reservation for stage '{stage}'")

        # Force garbage collection
        self._cleanup_gpu_memory()

        return True

    def release_all_gpu_memory(self) -> int:
        """
        Release all GPU memory reservations.

        SPEC-PIPELINE-001 U3: Idempotent cleanup

        Returns:
            Number of reservations released
        """
        with self._lock:
            stages_to_release = list(self._gpu_reservations.keys())

        count = 0
        for stage in stages_to_release:
            if self.release_gpu_memory(stage):
                count += 1

        logger.info(f"Released {count} GPU memory reservations")
        return count

    def check_pre_allocation_gpu(
        self,
        batch_size: int,
        estimated_per_file_mb: float,
        safety_margin: float = 1.3,
    ) -> Tuple[bool, str]:
        """
        Check if sufficient GPU memory is available before starting batch processing.

        SPEC-PIPELINE-001 U1: 시스템은 항상 모든 GPU 메모리 할당 전에 가용 메모리를 검증해야 한다
        SPEC-PIPELINE-001 E1: WHEN 배치 처리가 시작되면 THEN 시스템은 GPU 메모리 사전 할당 검증을 수행해야 한다

        Args:
            batch_size: Number of files to process
            estimated_per_file_mb: Estimated memory per file in MB
            safety_margin: Safety margin multiplier (default 1.3 = 30% buffer)

        Returns:
            Tuple of (is_safe, message)
        """
        gpu_stats = self._get_gpu_memory_stats()

        if not gpu_stats.get("gpu_available", False):
            return False, "GPU memory monitoring not available, cannot validate pre-allocation"

        gpu_free_mb = gpu_stats.get("gpu_free_mb", 0)
        gpu_total_mb = gpu_stats.get("gpu_total_mb", 1)

        # Calculate total memory needed with safety margin
        estimated_total_mb = batch_size * estimated_per_file_mb * safety_margin

        # Calculate predicted usage percentage
        gpu_used_mb = gpu_stats.get("gpu_used_mb", 0)
        predicted_used_mb = gpu_used_mb + estimated_total_mb
        predicted_percent = (predicted_used_mb / gpu_total_mb) * 100

        logger.info(
            f"Pre-allocation GPU check: "
            f"free={gpu_free_mb:.1f}MB, "
            f"estimated_needed={estimated_total_mb:.1f}MB, "
            f"predicted_percent={predicted_percent:.1f}%"
        )

        if predicted_percent > self.critical_threshold:
            return False, (
                f"Insufficient GPU memory: predicted {predicted_percent:.1f}% usage "
                f"exceeds critical threshold {self.critical_threshold}%. "
                f"Reduce batch size or free GPU memory."
            )

        if predicted_percent > self.gpu_memory_threshold:
            return True, (
                f"GPU memory will be high: predicted {predicted_percent:.1f}% usage. "
                f"Consider reducing batch size for optimal performance."
            )

        return True, (
            f"Sufficient GPU memory: predicted {predicted_percent:.1f}% usage. "
            f"Safe to proceed with batch processing."
        )

    def monitor_memory_pressure(self) -> MemoryPressureStatus:
        """
        Monitor current GPU memory pressure.

        SPEC-PIPELINE-001 S1: IF GPU 메모리 사용률이 90% 이상이면 THEN 시스템은 즉시 메모리 정리를 트리거해야 한다

        Returns:
            MemoryPressureStatus with current pressure information
        """
        import psutil

        # Get system memory
        system_memory = psutil.virtual_memory()
        system_percent = system_memory.percent
        available_mb = system_memory.available / (1024 * 1024)

        # Get GPU memory
        gpu_percent = self._get_gpu_memory_percent()

        # Determine pressure level based on GPU memory
        # Use GPU as primary indicator for forensic pipeline
        if gpu_percent is not None:
            if gpu_percent >= self.critical_threshold:
                level = MemoryPressureLevel.CRITICAL
                predicted_oom = True
                action = "IMMEDIATE ACTION REQUIRED: Stop processing and run emergency GPU cleanup"
                # Trigger emergency cleanup automatically
                self.emergency_gpu_cleanup()
            elif gpu_percent >= self.gpu_memory_threshold:
                level = MemoryPressureLevel.HIGH
                predicted_oom = True
                action = "Consider pausing batch and running GPU garbage collection"
            elif gpu_percent >= (self.gpu_memory_threshold - 10):
                level = MemoryPressureLevel.MEDIUM
                predicted_oom = False
                action = "Monitor closely, prepare for cleanup if needed"
            else:
                level = MemoryPressureLevel.LOW
                predicted_oom = False
                action = "Normal operation, continue processing"
        else:
            # Fallback to system memory if GPU not available
            if system_percent >= self.critical_threshold:
                level = MemoryPressureLevel.CRITICAL
                predicted_oom = True
                action = "IMMEDIATE ACTION REQUIRED: Stop processing and run emergency cleanup"
            elif system_percent >= self.gpu_memory_threshold:
                level = MemoryPressureLevel.HIGH
                predicted_oom = True
                action = "Consider pausing batch and running garbage collection"
            elif system_percent >= (self.gpu_memory_threshold - 10):
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

    def emergency_gpu_cleanup(self) -> Dict[str, Any]:
        """
        Perform emergency GPU memory cleanup.

        SPEC-PIPELINE-001 E3: WHEN 배치 처리 중 OOM이 감지되면 THEN 시스템은 비상 메모리 정리를 수행해야 한다
        SPEC-PIPELINE-001 N1: 시스템은 OOM 감지 시 프로세스를 강제 종료하지 않아야 한다

        This includes:
        - Release all GPU reservations
        - Clear PyTorch CUDA cache
        - Force Python garbage collection

        Returns:
            Dictionary with cleanup statistics
        """
        cleanup_stats = {
            "reservations_released": 0,
            "gpu_cache_cleared": False,
            "gc_collected": 0,
            "memory_before_mb": 0,
            "memory_after_mb": 0,
            "memory_freed_mb": 0,
            "errors": [],
        }

        logger.warning("Starting emergency GPU cleanup...")

        # Get memory before
        try:
            import psutil

            process = psutil.Process()
            cleanup_stats["memory_before_mb"] = process.memory_info().rss / (1024 * 1024)
        except Exception as e:
            cleanup_stats["errors"].append(f"Failed to get memory before: {e}")

        # Release all GPU reservations
        try:
            released_count = self.release_all_gpu_memory()
            cleanup_stats["reservations_released"] = released_count
        except Exception as e:
            cleanup_stats["errors"].append(f"Failed to release GPU reservations: {e}")

        # Clear GPU cache
        try:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()
                cleanup_stats["gpu_cache_cleared"] = True
                logger.info("GPU cache cleared")
        except Exception as e:
            cleanup_stats["errors"].append(f"Failed to clear GPU cache: {e}")

        # Force garbage collection
        try:
            collected = gc.collect()
            cleanup_stats["gc_collected"] = collected
        except Exception as e:
            cleanup_stats["errors"].append(f"Failed to run garbage collection: {e}")

        # Get memory after
        try:
            import psutil

            process = psutil.Process()
            cleanup_stats["memory_after_mb"] = process.memory_info().rss / (1024 * 1024)
            cleanup_stats["memory_freed_mb"] = (
                cleanup_stats["memory_before_mb"] - cleanup_stats["memory_after_mb"]
            )
        except Exception as e:
            cleanup_stats["errors"].append(f"Failed to get memory after: {e}")

        logger.warning(
            f"Emergency GPU cleanup complete: "
            f"{cleanup_stats['reservations_released']} reservations released, "
            f"{cleanup_stats['gc_collected']} objects collected, "
            f"{cleanup_stats['memory_freed_mb']:.2f}MB freed"
        )

        return cleanup_stats

    def _get_gpu_free_memory_mb(self) -> float:
        """
        Get GPU free memory in MB.

        Returns:
            Free memory in MB, or 0 if unavailable
        """
        gpu_stats = self._get_gpu_memory_stats()
        return gpu_stats.get("gpu_free_mb", 0)

    def _get_gpu_memory_percent(self) -> Optional[float]:
        """
        Get GPU memory usage percentage.

        Returns:
            Memory usage percentage, or None if unavailable
        """
        gpu_stats = self._get_gpu_memory_stats()

        if not gpu_stats.get("gpu_available", False):
            return None

        total_mb = gpu_stats.get("gpu_total_mb", 1)
        used_mb = gpu_stats.get("gpu_used_mb", 0)

        return (used_mb / total_mb) * 100 if total_mb > 0 else None

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.release_all()
            self.release_all_gpu_memory()  # SPEC-PIPELINE-001: Release GPU reservations
            if PYNVML_AVAILABLE:
                try:
                    pynvml.nvmlShutdown()
                except Exception:
                    pass
        except Exception:
            pass
