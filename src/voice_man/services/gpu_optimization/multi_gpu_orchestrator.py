"""
Multi-GPU Orchestrator for WhisperX Pipeline

Phase 2 Intermediate Optimization (20-30x speedup target):
- GPU auto-detection and assignment
- Pipeline stage distribution across GPUs
- Workload balancing across multiple GPUs
- EARS Requirements: E3 (multi-GPU detection), S2 (parallel processing)

Reference: SPEC-GPUOPT-001 Phase 2
"""

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from voice_man.services.gpu_monitor_service import GPUMonitorService

logger = logging.getLogger(__name__)


@dataclass
class GPUInfo:
    """Information about a GPU."""

    device_index: int
    name: str
    total_memory_mb: float
    compute_capability: str
    current_load: float = 0.0  # 0.0 to 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "device_index": self.device_index,
            "name": self.name,
            "total_memory_mb": self.total_memory_mb,
            "compute_capability": self.compute_capability,
            "current_load": self.current_load,
        }


@dataclass
class OrchestratorConfig:
    """Multi-GPU orchestrator configuration."""

    # Load balancing strategy
    load_balancing: str = "round_robin"  # round_robin, least_loaded, memory_based

    # Pipeline stage assignment
    auto_assign_stages: bool = True
    stage_gpu_mapping: Optional[Dict[str, int]] = (
        None  # {"transcribe": 0, "align": 1, "diarize": 0}
    )

    # Thresholds
    load_threshold: float = 0.8  # Trigger rebalancing when load > 80%
    memory_threshold_mb: float = 1000  # Minimum free memory


class MultiGPUOrchestrator:
    """
    Multi-GPU orchestrator for parallel WhisperX pipeline execution.

    Features:
    - GPU auto-detection and inventory
    - Pipeline stage distribution across GPUs
    - Workload balancing (round-robin, least-loaded, memory-based)
    - Dynamic GPU assignment based on load

    EARS Requirements:
    - E3: Auto-detect multi-GPU and create distribution strategy
    - S2: Pipeline stages distributed in parallel across GPUs
    - U1: Same GPU context maintained within each stage

    Performance Target: 2-3x speedup with 2 GPUs (cumulative 20-30x with Phase 1)
    """

    # Pipeline stages
    STAGES = ["transcribe", "align", "diarize"]

    def __init__(self, config: Optional[OrchestratorConfig] = None):
        """
        Initialize Multi-GPU Orchestrator.

        Args:
            config: Orchestrator configuration (default: default config)
        """
        self.config = config or OrchestratorConfig()

        # GPU inventory
        self.gpus: List[GPUInfo] = []
        self._detect_gpus()

        # Current assignment state
        self._stage_assignments: Dict[str, int] = {}
        self._load_tracker: Dict[int, float] = {i: 0.0 for i in range(len(self.gpus))}

        # Thread safety
        self._lock = threading.RLock()

        # Initialize stage assignments
        self._initialize_stage_assignments()

        logger.info(
            f"MultiGPUOrchestrator initialized: "
            f"{len(self.gpus)} GPUs detected, "
            f"load_balancing={self.config.load_balancing}"
        )

    def _detect_gpus(self) -> None:
        """Detect available GPUs."""
        try:
            import torch

            if not torch.cuda.is_available():
                logger.warning("CUDA not available, running in CPU-only mode")
                return

            num_gpus = torch.cuda.device_count()
            logger.info(f"Detected {num_gpus} GPU(s)")

            for i in range(num_gpus):
                name = torch.cuda.get_device_name(i)
                capability = torch.cuda.get_device_capability(i)
                props = torch.cuda.get_device_properties(i)

                gpu_info = GPUInfo(
                    device_index=i,
                    name=name,
                    total_memory_mb=props.total_memory / (1024 * 1024),
                    compute_capability=f"{capability[0]}.{capability[1]}",
                )

                self.gpus.append(gpu_info)
                logger.info(f"GPU {i}: {name} ({gpu_info.total_memory_mb:.0f}MB)")

        except Exception as e:
            logger.error(f"Failed to detect GPUs: {e}")

    def _initialize_stage_assignments(self) -> None:
        """Initialize GPU assignments for pipeline stages."""
        # Use custom mapping if provided
        if self.config.stage_gpu_mapping:
            self._stage_assignments = self.config.stage_gpu_mapping.copy()
            logger.info(f"Using custom stage GPU mapping: {self._stage_assignments}")
            return

        # Auto-assign stages to GPUs
        if self.config.auto_assign_stages and len(self.gpus) > 1:
            # Distribute stages across available GPUs
            for i, stage in enumerate(self.STAGES):
                gpu_index = i % len(self.gpus)
                self._stage_assignments[stage] = gpu_index

            logger.info(f"Auto-assigned stages to GPUs: {self._stage_assignments}")
        else:
            # All stages on GPU 0 (single GPU mode)
            for stage in self.STAGES:
                self._stage_assignments[stage] = 0

            logger.info("Single GPU mode: all stages on GPU 0")

    def get_gpu_for_stage(self, stage: str) -> int:
        """
        Get GPU assignment for a pipeline stage.

        Args:
            stage: Pipeline stage (transcribe, align, diarize)

        Returns:
            GPU device index

        E3: Multi-GPU detection and strategy creation
        """
        with self._lock:
            if stage not in self._stage_assignments:
                # Default to GPU 0
                return 0

            # Apply load balancing if needed
            if self.config.load_balancing == "least_loaded":
                return self._get_least_loaded_gpu()

            return self._stage_assignments[stage]

    def _get_least_loaded_gpu(self) -> int:
        """Get the GPU with the lowest current load."""
        if not self.gpus:
            return 0

        # Find GPU with minimum load
        min_load = float("inf")
        min_gpu = 0

        for gpu_idx, load in self._load_tracker.items():
            if gpu_idx < len(self.gpus) and load < min_load:
                min_load = load
                min_gpu = gpu_idx

        return min_gpu

    def update_load(self, gpu_index: int, delta: float) -> None:
        """
        Update load tracker for a GPU.

        Args:
            gpu_index: GPU device index
            delta: Load change (+ for adding work, - for completing work)
        """
        with self._lock:
            if gpu_index in self._load_tracker:
                self._load_tracker[gpu_index] += delta
                # Clamp to [0, 1]
                self._load_tracker[gpu_index] = max(0.0, min(1.0, self._load_tracker[gpu_index]))

    def get_gpu_info(self, device_index: int) -> Optional[GPUInfo]:
        """
        Get information about a specific GPU.

        Args:
            device_index: GPU device index

        Returns:
            GPUInfo or None if not found
        """
        if 0 <= device_index < len(self.gpus):
            return self.gpus[device_index]
        return None

    def get_all_gpu_info(self) -> List[GPUInfo]:
        """
        Get information about all GPUs.

        Returns:
            List of GPUInfo objects
        """
        return self.gpus.copy()

    def get_optimal_distribution(self, num_tasks: int) -> Dict[int, int]:
        """
        Calculate optimal task distribution across GPUs.

        Args:
            num_tasks: Total number of tasks to distribute

        Returns:
            Dictionary mapping GPU index to task count
        """
        if not self.gpus:
            return {0: num_tasks}

        distribution = {}
        num_gpus = len(self.gpus)

        # Load-based distribution
        if self.config.load_balancing == "least_loaded":
            # Distribute based on current load
            for i in range(num_gpus):
                load = self._load_tracker.get(i, 0.0)
                # More tasks for less loaded GPUs
                capacity = max(0.1, 1.0 - load)
                distribution[i] = 0  # Will be calculated

            total_capacity = sum(distribution.get(i, 0.1) for i in range(num_gpus))

            for i in range(num_gpus):
                capacity = max(0.1, 1.0 - self._load_tracker.get(i, 0.0))
                distribution[i] = max(1, int(num_tasks * capacity / total_capacity))

        else:
            # Round-robin distribution
            tasks_per_gpu = num_tasks // num_gpus
            remainder = num_tasks % num_gpus

            for i in range(num_gpus):
                distribution[i] = tasks_per_gpu + (1 if i < remainder else 0)

        return distribution

    def should_rebalance(self) -> bool:
        """
        Check if workload rebalancing is needed.

        Returns:
            True if rebalancing is recommended
        """
        if not self.gpus:
            return False

        # Check if any GPU is overloaded
        for gpu_idx, load in self._load_tracker.items():
            if gpu_idx < len(self.gpus) and load > self.config.load_threshold:
                return True

        return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get orchestrator statistics.

        Returns:
            Dictionary with statistics
        """
        with self._lock:
            gpu_stats = []
            for i, gpu in enumerate(self.gpus):
                gpu_stats.append(
                    {
                        **gpu.to_dict(),
                        "current_load": self._load_tracker.get(i, 0.0),
                        "assigned_stages": [
                            stage
                            for stage, gpu_idx in self._stage_assignments.items()
                            if gpu_idx == i
                        ],
                    }
                )

            return {
                "num_gpus": len(self.gpus),
                "gpus": gpu_stats,
                "stage_assignments": self._stage_assignments.copy(),
                "load_balancing": self.config.load_balancing,
                "load_tracker": self._load_tracker.copy(),
                "should_rebalance": self.should_rebalance(),
            }

    def set_stage_gpu(self, stage: str, gpu_index: int) -> None:
        """
        Manually set GPU assignment for a stage.

        Args:
            stage: Pipeline stage name
            gpu_index: GPU device index
        """
        with self._lock:
            if stage in self.STAGES:
                self._stage_assignments[stage] = gpu_index
                logger.info(f"Set stage '{stage}' to GPU {gpu_index}")

    def rebalance(self) -> Dict[str, int]:
        """
        Rebalance stage assignments based on current load.

        Returns:
            New stage assignments
        """
        with self._lock:
            if not self.gpus or len(self.gpus) <= 1:
                logger.info("Single GPU mode, rebalancing not needed")
                return self._stage_assignments.copy()

            # Redistribute stages based on load
            new_assignments = {}

            for stage in self.STAGES:
                new_assignments[stage] = self._get_least_loaded_gpu()

            old_assignments = self._stage_assignments.copy()
            self._stage_assignments = new_assignments

            logger.info(f"Rebalanced stages: {old_assignments} -> {new_assignments}")

            return new_assignments

    def is_multi_gpu(self) -> bool:
        """
        Check if running in multi-GPU mode.

        Returns:
            True if multiple GPUs available
        """
        return len(self.gpus) > 1

    def get_recommended_device(self, stage: Optional[str] = None) -> str:
        """
        Get recommended device for a stage or general use.

        Args:
            stage: Optional pipeline stage name

        Returns:
            Device string ("cuda:0", "cuda:1", or "cpu")
        """
        if not self.gpus:
            return "cpu"

        if stage:
            gpu_index = self.get_gpu_for_stage(stage)
            return f"cuda:{gpu_index}"

        # Return least loaded GPU
        gpu_index = self._get_least_loaded_gpu()
        return f"cuda:{gpu_index}"
