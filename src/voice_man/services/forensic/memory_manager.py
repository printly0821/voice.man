"""
ForensicMemoryManager - Stage-based GPU memory allocation for forensic pipeline.

SPEC-PERFOPT-001 Phase 2: GPU memory management for forensic analysis stages.

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
"""

import gc
import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Try to import pynvml for GPU memory monitoring
try:
    import pynvml

    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    logger.warning("pynvml not available, GPU memory monitoring will be limited")


# Default stage memory allocations in MB
DEFAULT_STAGE_ALLOCATIONS: Dict[str, int] = {
    "stt": 16384,  # 16GB
    "alignment": 4096,  # 4GB
    "diarization": 8192,  # 8GB
    "ser": 10240,  # 10GB
    "scoring": 2048,  # 2GB
}


@dataclass
class StageAllocation:
    """Represents a memory allocation for a stage."""

    stage: str
    amount_mb: int
    allocated_at: float = 0.0


class ForensicMemoryManager:
    """
    Stage-based GPU memory manager for forensic pipeline.

    Provides thread-safe memory allocation and release for different
    stages of the forensic audio analysis pipeline.

    Attributes:
        stage_allocations: Default memory allocations for each stage (in MB)

    Example:
        manager = ForensicMemoryManager()
        manager.allocate("stt")
        # ... perform STT processing ...
        manager.release("stt")

        # Or with context manager:
        with manager.stage_context("ser") as allocated:
            if allocated:
                # ... perform SER processing ...
    """

    def __init__(
        self,
        stage_allocations: Optional[Dict[str, int]] = None,
        device_index: int = 0,
    ):
        """
        Initialize ForensicMemoryManager.

        Args:
            stage_allocations: Custom stage allocations in MB. If None, uses defaults.
            device_index: GPU device index for pynvml queries.
        """
        self.stage_allocations = (
            stage_allocations.copy() if stage_allocations else DEFAULT_STAGE_ALLOCATIONS.copy()
        )
        self.device_index = device_index

        # Thread-safety lock
        self._lock = threading.Lock()

        # Track allocated stages
        self._allocated_stages: Dict[str, StageAllocation] = {}

        # Initialize pynvml if available
        self._gpu_handle: Optional[Any] = None
        self._init_pynvml()

        logger.info(
            f"ForensicMemoryManager initialized with stages: {list(self.stage_allocations.keys())}"
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

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.release_all()
            if PYNVML_AVAILABLE:
                try:
                    pynvml.nvmlShutdown()
                except Exception:
                    pass
        except Exception:
            pass
