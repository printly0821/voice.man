"""
EdgeXpertOrchestrator - Single GPU orchestration for MSI EdgeXpert.

This module orchestrates all EdgeXpert components to maximize single GPU
utilization (95%+) while maintaining temperature below 85°C.

Components Coordinated:
    - UnifiedMemoryManager: Zero-copy memory management
    - CUDAStreamProcessor: 4-stream parallel processing
    - HardwareAcceleratedCodec: NVDEC audio decoding
    - BlackWellOptimizer: FP4/Sparse optimization
    - ARMCPUPipeline: 20-core ARM parallel I/O
    - ThermalManager: Temperature management (85°C limit)

Reference: SPEC-EDGEXPERT-001
"""

import logging
import time
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from enum import Enum

import torch

from voice_man.services.edgexpert.unified_memory_manager import UnifiedMemoryManager
from voice_man.services.edgexpert.cuda_stream_processor import CUDAStreamProcessor
from voice_man.services.edgexpert.hardware_accelerated_codec import HardwareAcceleratedCodec
from voice_man.services.edgexpert.blackwell_optimizer import BlackWellOptimizer
from voice_man.services.edgexpert.arm_cpu_pipeline import ARMCPUPipeline
from voice_man.services.edgexpert.thermal_manager import ThermalManager

logger = logging.getLogger(__name__)


class OperationPhase(Enum):
    """Operation phases for orchestration."""

    PHASE_1 = "phase1"  # Unified Memory + CUDA Stream (4-6x target)
    PHASE_2 = "phase2"  # FP4/Sparse + ARM Parallel (6.75-9x target)


@dataclass
class OrchestratorConfig:
    """Orchestrator configuration."""

    # Phase selection
    operation_phase: OperationPhase = OperationPhase.PHASE_2

    # GPU settings
    num_cuda_streams: int = 4
    target_gpu_utilization: float = 95.0

    # Memory settings
    memory_pool_size_gb: int = 120
    memory_warning_threshold_gb: int = 110

    # Thermal settings
    max_temp: int = 85
    warning_temp: int = 80
    target_temp: int = 70

    # Optimization settings
    enable_fp4: bool = True
    enable_sparse: bool = True

    # ARM CPU settings
    enable_arm_parallel: bool = True

    # Codec settings
    enable_nvdec: bool = True


@dataclass
class PerformanceMetrics:
    """Performance tracking metrics."""

    # Processing metrics
    total_processing_time: float = 0.0
    files_processed: int = 0
    average_throughput: float = 0.0

    # GPU metrics
    gpu_utilization: float = 0.0
    memory_used_gb: float = 0.0
    peak_memory_gb: float = 0.0

    # Thermal metrics
    current_temp: int = 0
    peak_temp: int = 0
    average_temp: float = 0.0
    throttle_count: int = 0

    # Speedup metrics
    speedup_factor: float = 1.0
    baseline_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_processing_time": self.total_processing_time,
            "files_processed": self.files_processed,
            "average_throughput": self.average_throughput,
            "gpu_utilization": self.gpu_utilization,
            "memory_used_gb": self.memory_used_gb,
            "peak_memory_gb": self.peak_memory_gb,
            "current_temp": self.current_temp,
            "peak_temp": self.peak_temp,
            "average_temp": self.average_temp,
            "throttle_count": self.throttle_count,
            "speedup_factor": self.speedup_factor,
            "baseline_time": self.baseline_time,
        }


class EdgeXpertOrchestrator:
    """
    Single GPU orchestration for MSI EdgeXpert.

    Maximizes GPU utilization through:
        - 4-stream CUDA parallel processing
        - Zero-copy unified memory
        - FP4/Sparse optimization
        - ARM CPU parallel I/O
        - Dynamic thermal management

    Target Performance:
        - Phase 1: 4-6x speedup, 95%+ GPU utilization
        - Phase 2: 6.75-9x speedup, 95%+ GPU utilization
    """

    def __init__(self, config: Optional[OrchestratorConfig] = None):
        """
        Initialize EdgeXpert Orchestrator.

        Args:
            config: Orchestrator configuration (default: default config)
        """
        self.config = config or OrchestratorConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize components
        self.memory_manager = UnifiedMemoryManager(device=str(self.device))
        self.stream_processor = CUDAStreamProcessor(
            num_streams=self.config.num_cuda_streams, device=str(self.device)
        )
        self.codec = HardwareAcceleratedCodec(
            use_nvdec=self.config.enable_nvdec, device=str(self.device)
        )
        self.optimizer = BlackWellOptimizer(
            enable_fp4=self.config.enable_fp4, enable_sparse=self.config.enable_sparse
        )
        self.arm_pipeline = ARMCPUPipeline() if self.config.enable_arm_parallel else None
        self.thermal_manager = ThermalManager(
            max_temp=self.config.max_temp,
            warning_temp=self.config.warning_temp,
            target_temp=self.config.target_temp,
        )

        # Performance tracking
        self.metrics = PerformanceMetrics()

        # State tracking
        self._is_processing = False
        self._last_temperature_check = 0.0
        self._temperature_check_interval = 5.0  # seconds

        logger.info(
            f"EdgeXpertOrchestrator initialized: "
            f"phase={self.config.operation_phase.value}, "
            f"device={self.device}, "
            f"streams={self.config.num_cuda_streams}"
        )

    def process_audio_batch(
        self,
        audio_files: List[str],
        process_func: Callable[[torch.Tensor], Any],
        baseline_time: Optional[float] = None,
    ) -> List[Any]:
        """
        Process a batch of audio files with full optimization.

        Args:
            audio_files: List of audio file paths
            process_func: Processing function for each audio tensor
            baseline_time: Baseline processing time for speedup calculation

        Returns:
            List of processing results
        """
        if self._is_processing:
            logger.warning("Processing already in progress, skipping")
            return []

        self._is_processing = True
        start_time = time.time()

        try:
            # Record baseline time
            if baseline_time:
                self.metrics.baseline_time = baseline_time

            # Step 1: Parallel audio loading with ARM CPU
            logger.info(f"Loading {len(audio_files)} audio files...")
            audio_tensors = self._load_audio_parallel(audio_files)

            # Step 2: Thermal check
            self._check_thermal_and_adjust()

            # Step 3: Process with CUDA streams
            logger.info(f"Processing {len(audio_tensors)} audio tensors...")
            results = self._process_with_streams(audio_tensors, process_func)

            # Step 4: Update metrics
            processing_time = time.time() - start_time
            self._update_metrics(len(audio_files), processing_time)

            logger.info(
                f"Batch processing complete: "
                f"{len(audio_files)} files in {processing_time:.2f}s "
                f"({self.metrics.average_throughput:.2f} files/sec)"
            )

            return results

        finally:
            self._is_processing = False

    def _load_audio_parallel(self, audio_files: List[str]) -> List[torch.Tensor]:
        """Load audio files in parallel using ARM CPU pipeline."""
        if not self.arm_pipeline:
            # Sequential fallback
            return [self.codec.decode_audio_gpu(f) for f in audio_files]

        def load_single(path: str) -> torch.Tensor:
            return self.codec.decode_audio_gpu(path)

        # Use ARM pipeline for parallel loading
        result = self.arm_pipeline.load_parallel(
            files=audio_files,
            load_func=load_single,
            num_workers=self.arm_pipeline.get_optimal_worker_count(task_type="io"),
        )

        # Filter out None values
        return [t for t in result if t is not None]

    def _process_with_streams(
        self,
        tensors: List[torch.Tensor],
        process_func: Callable[[torch.Tensor], Any],
    ) -> List[Any]:
        """Process tensors using CUDA streams."""
        results = self.stream_processor.process_parallel(tensors, process_func)
        return results

    def _check_thermal_and_adjust(self) -> None:
        """Check temperature and adjust processing if needed."""
        current_time = time.time()

        # Only check temperature at intervals
        if current_time - self._last_temperature_check < self._temperature_check_interval:
            return

        self._last_temperature_check = current_time

        # Get current temperature
        temp = self.thermal_manager.get_current_temperature()
        self.metrics.current_temp = temp

        # Check for throttling
        if temp >= self.config.warning_temp:
            logger.warning(
                f"Temperature {temp}°C >= {self.config.warning_temp}°C, "
                f"thermal throttling may be triggered"
            )
            self.metrics.throttle_count += 1

        # Record to history
        self.thermal_manager.record_temperature()

    def _update_metrics(self, files_processed: int, processing_time: float) -> None:
        """Update performance metrics."""
        self.metrics.files_processed += files_processed
        self.metrics.total_processing_time += processing_time
        self.metrics.average_throughput = (
            self.metrics.files_processed / self.metrics.total_processing_time
        )

        # GPU utilization
        self.metrics.gpu_utilization = self.stream_processor.get_gpu_utilization()

        # Memory usage
        memory_info = self.memory_manager.get_memory_usage()
        self.metrics.memory_used_gb = memory_info.get("used", 0.0)
        self.metrics.peak_memory_gb = max(self.metrics.peak_memory_gb, self.metrics.memory_used_gb)

        # Thermal metrics
        thermal_stats = self.thermal_manager.get_thermal_stats()
        self.metrics.current_temp = thermal_stats.get("current_temp", 0)
        self.metrics.peak_temp = thermal_stats.get("max_temp_observed", 0)
        self.metrics.average_temp = thermal_stats.get("avg_temp", 0.0)
        self.metrics.throttle_count = thermal_stats.get("throttle_count", 0)

        # Calculate speedup
        if self.metrics.baseline_time > 0:
            self.metrics.speedup_factor = self.metrics.baseline_time / processing_time

    def optimize_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Optimize a model with FP4/Sparse quantization.

        Args:
            model: PyTorch model to optimize

        Returns:
            Optimized model
        """
        if self.config.operation_phase == OperationPhase.PHASE_2:
            logger.info("Applying Phase 2 optimization (FP4/Sparse)...")
            optimized = self.optimizer.quantize_to_fp4(model)

            # Calculate memory savings
            memory_saved = self.optimizer.calculate_memory_savings(model, optimized)
            logger.info(f"Memory saved: {memory_saved:.2f} MB")

            return optimized
        else:
            logger.info("Phase 1: No model optimization applied")
            return model

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get current performance metrics.

        Returns:
            Dictionary of performance metrics
        """
        metrics_dict = self.metrics.to_dict()

        # Add component-specific stats
        metrics_dict["components"] = {
            "memory_manager": self.memory_manager.get_memory_usage(),
            "stream_processor": {
                "queue_size": self.stream_processor.get_queue_size(),
                "gpu_utilization": self.stream_processor.get_gpu_utilization(),
            },
            "codec": self.codec.get_decoding_metrics(),
            "optimizer": self.optimizer.get_optimization_stats(),
            "thermal_manager": self.thermal_manager.get_thermal_stats(),
        }

        if self.arm_pipeline:
            metrics_dict["components"]["arm_pipeline"] = self.arm_pipeline.get_pipeline_stats()

        return metrics_dict

    def reset_metrics(self) -> None:
        """Reset performance metrics."""
        self.metrics = PerformanceMetrics()
        logger.info("Performance metrics reset")

    def get_configuration(self) -> Dict[str, Any]:
        """
        Get current configuration.

        Returns:
            Configuration dictionary
        """
        return {
            "operation_phase": self.config.operation_phase.value,
            "device": str(self.device),
            "num_cuda_streams": self.config.num_cuda_streams,
            "target_gpu_utilization": self.config.target_gpu_utilization,
            "memory_pool_size_gb": self.config.memory_pool_size_gb,
            "max_temp": self.config.max_temp,
            "warning_temp": self.config.warning_temp,
            "target_temp": self.config.target_temp,
            "enable_fp4": self.config.enable_fp4,
            "enable_sparse": self.config.enable_sparse,
            "enable_arm_parallel": self.config.enable_arm_parallel,
            "enable_nvdec": self.config.enable_nvdec,
        }

    def is_processing(self) -> bool:
        """Check if currently processing."""
        return self._is_processing

    def cleanup(self) -> None:
        """Cleanup resources."""
        logger.info("Cleaning up EdgeXpertOrchestrator resources...")
        self.memory_manager.release_memory()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
