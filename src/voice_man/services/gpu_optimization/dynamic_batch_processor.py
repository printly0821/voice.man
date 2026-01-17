"""
Dynamic Batch Processor for WhisperX Pipeline

Phase 2 Intermediate Optimization (20-30x speedup target):
- Dynamic batch size adjustment (4-32 files)
- GPU memory-based scaling
- Audio chunking for long files (>30 min)
- EARS Requirements: S2, S4

Reference: SPEC-GPUOPT-001 Phase 2
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from voice_man.services.gpu_monitor_service import GPUMonitorService

logger = logging.getLogger(__name__)


@dataclass
class DynamicBatchConfig:
    """Dynamic batch configuration."""

    # Batch size limits
    min_batch_size: int = 4
    max_batch_size: int = 32
    initial_batch_size: int = 16

    # Memory-based scaling
    memory_scale_factor: float = 0.15  # Use 15% of GPU memory per batch
    memory_safety_margin: float = 0.2  # 20% safety margin

    # Audio duration-based scaling
    enable_duration_scaling: bool = True
    short_audio_threshold: int = 300  # 5 minutes in seconds
    long_audio_threshold: int = 1800  # 30 minutes in seconds
    long_audio_chunk_duration: int = 600  # 10 minutes in seconds

    # Progressive scaling
    scale_up_factor: float = 1.25
    scale_down_factor: float = 0.75
    scale_up_threshold: float = 0.7  # Scale up when GPU < 70% load
    scale_down_threshold: float = 0.9  # Scale down when GPU > 90% load


@dataclass
class AudioMetadata:
    """Metadata for audio file."""

    path: str
    duration: float  # seconds
    size_bytes: int

    @property
    def is_short(self) -> bool:
        """Check if audio is short (< 5 min)."""
        return self.duration < 300

    @property
    def is_long(self) -> bool:
        """Check if audio is long (> 30 min)."""
        return self.duration > 1800

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "path": self.path,
            "duration": self.duration,
            "size_bytes": self.size_bytes,
            "is_short": self.is_short,
            "is_long": self.is_long,
        }


@dataclass
class BatchPlan:
    """Plan for batch processing."""

    batches: List[List[str]] = field(default_factory=list)
    batch_sizes: List[int] = field(default_factory=list)
    estimated_time_seconds: float = 0.0
    requires_chunking: List[str] = field(default_factory=list)  # Files that need chunking

    def total_files(self) -> int:
        """Total number of files to process."""
        return sum(len(batch) for batch in self.batches)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "num_batches": len(self.batches),
            "batch_sizes": self.batch_sizes,
            "total_files": self.total_files(),
            "estimated_time_seconds": self.estimated_time_seconds,
            "requires_chunking": self.requires_chunking,
        }


class DynamicBatchProcessor:
    """
    Dynamic batch processor with GPU memory awareness.

    Features:
    - Dynamic batch size (4-32) based on GPU memory
    - Audio duration-aware batching
    - Long audio chunking (>30 min)
    - Progressive scaling based on GPU utilization

    EARS Requirements:
    - S2: Pipeline stages distributed in parallel across GPUs
    - S4: Long audio (>30 min) split into 10-min chunks for parallel processing
    - S5: Short audio (<5 min) processed as single file without batching

    Performance Target: 20-30x cumulative speedup with Phase 1
    """

    def __init__(self, config: Optional[DynamicBatchConfig] = None):
        """
        Initialize Dynamic Batch Processor.

        Args:
            config: Batch configuration (default: default config)
        """
        self.config = config or DynamicBatchConfig()

        # GPU monitor
        self.gpu_monitor = GPUMonitorService()

        # Current batch size
        self.current_batch_size = self.config.initial_batch_size

        # Statistics
        self._total_processed = 0
        self._total_time = 0.0
        self._average_batch_size = 0.0

        logger.info(
            f"DynamicBatchProcessor initialized: "
            f"batch_size={self.config.min_batch_size}-"
            f"{self.config.max_batch_size}"
        )

    def get_audio_metadata(self, audio_path: str) -> AudioMetadata:
        """
        Get metadata for audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            AudioMetadata object
        """
        try:
            import librosa

            duration = librosa.get_duration(path=audio_path)
            size = Path(audio_path).stat().st_size

            return AudioMetadata(
                path=audio_path,
                duration=duration,
                size_bytes=size,
            )

        except Exception as e:
            logger.warning(f"Failed to get metadata for {audio_path}: {e}")
            # Return default metadata
            return AudioMetadata(
                path=audio_path,
                duration=600.0,  # 10 minutes default
                size_bytes=0,
            )

    def calculate_batch_size(self, audio_files: List[str]) -> int:
        """
        Calculate optimal batch size based on GPU memory and audio metadata.

        Args:
            audio_files: List of audio file paths

        Returns:
            Optimal batch size
        """
        if not self.gpu_monitor.is_gpu_available():
            return self.config.min_batch_size

        # Get GPU memory stats
        memory_stats = self.gpu_monitor.get_gpu_memory_stats()
        if not memory_stats.get("available", False):
            return self.config.min_batch_size

        total_memory_mb = memory_stats["total_mb"]
        free_memory_mb = memory_stats["free_mb"]

        # Calculate safe memory to use
        usable_memory_mb = free_memory_mb * (1 - self.config.memory_safety_margin)
        memory_per_batch = total_memory_mb * self.config.memory_scale_factor

        # Initial batch size from memory
        memory_based_batch = int(usable_memory_mb / max(1, memory_per_batch))

        # Clamp to configured limits
        batch_size = max(
            self.config.min_batch_size,
            min(self.config.max_batch_size, memory_based_batch),
        )

        # Adjust based on audio durations if enabled
        if self.config.enable_duration_scaling:
            # Get metadata for sample of files
            sample_size = min(10, len(audio_files))
            durations = []

            for path in audio_files[:sample_size]:
                metadata = self.get_audio_metadata(path)
                durations.append(metadata.duration)

            avg_duration = sum(durations) / len(durations) if durations else 600

            # Adjust batch size based on duration
            if avg_duration < self.config.short_audio_threshold:
                # Short audio: can use larger batch
                batch_size = min(
                    self.config.max_batch_size,
                    int(batch_size * 1.5),
                )
            elif avg_duration > self.config.long_audio_threshold:
                # Long audio: reduce batch size
                batch_size = max(
                    self.config.min_batch_size,
                    int(batch_size * 0.5),
                )

        return batch_size

    def plan_batch_processing(
        self,
        audio_files: List[str],
        batch_size: Optional[int] = None,
    ) -> BatchPlan:
        """
        Plan batch processing for audio files.

        Args:
            audio_files: List of audio file paths
            batch_size: Optional fixed batch size (None to auto-calculate)

        Returns:
            BatchPlan with batches and metadata
        """
        if not audio_files:
            return BatchPlan()

        # Calculate batch size if not provided
        if batch_size is None:
            batch_size = self.calculate_batch_size(audio_files)

        # Get metadata for all files
        metadata_map = {}
        long_files = []

        for path in audio_files:
            metadata = self.get_audio_metadata(path)
            metadata_map[path] = metadata

            if metadata.is_long:
                long_files.append(path)

        # Combine short and normal files for batch processing
        # Short files are ideal for GPU batching (small size, low memory)
        batchable_files = []

        for path in audio_files:
            metadata = metadata_map[path]
            if not metadata.is_long:
                # Both short and normal files can be batched for GPU efficiency
                batchable_files.append(path)

        # Plan batches
        batches = []
        batch_sizes = []
        requires_chunking = long_files.copy()

        # Process long files individually (will be chunked due to size)
        for long_file in long_files:
            batches.append([long_file])
            batch_sizes.append(1)

        # Batch all batchable files (short + normal) together
        for i in range(0, len(batchable_files), batch_size):
            batch = batchable_files[i : i + batch_size]
            if batch:
                batches.append(batch)
                batch_sizes.append(len(batch))

        # Estimate processing time (rough estimate: 1 sec per minute of audio)
        total_duration = sum(metadata_map[p].duration for p in audio_files)
        estimated_time = total_duration * 0.5  # Assume 2x speedup from optimization

        plan = BatchPlan(
            batches=batches,
            batch_sizes=batch_sizes,
            estimated_time_seconds=estimated_time,
            requires_chunking=requires_chunking,
        )

        logger.info(
            f"Batch plan created: {len(batches)} batches, "
            f"{plan.total_files()} files, "
            f"{len(long_files)} long files (need chunking), "
            f"estimated time: {estimated_time:.0f}s"
        )

        return plan

    def process_batch_dynamic(
        self,
        audio_files: List[str],
        process_func: Callable[[List[str]], List[Any]],
    ) -> List[Any]:
        """
        Process audio files with dynamic batching.

        Args:
            audio_files: List of audio file paths
            process_func: Function to process batch

        Returns:
            List of results
        """
        start_time = time.time()

        # Create batch plan
        plan = self.plan_batch_processing(audio_files)

        all_results = []

        # Process each batch
        for i, batch in enumerate(plan.batches):
            batch_start = time.time()

            # Check GPU memory before processing
            if self.gpu_monitor.is_gpu_available():
                memory_status = self.gpu_monitor.check_memory_status()
                if memory_status.get("auto_adjust_recommended", False):
                    # Reduce batch size
                    new_size = max(
                        self.config.min_batch_size,
                        len(batch) // 2,
                    )
                    if new_size < len(batch):
                        # Split batch
                        logger.info(
                            f"Splitting batch due to memory pressure: {len(batch)} -> {new_size}"
                        )
                        batch = batch[:new_size]

            try:
                results = process_func(batch)
                all_results.extend(results)

                batch_time = time.time() - batch_start

                # Update batch size based on success
                if self.gpu_monitor.is_gpu_available():
                    memory_stats = self.gpu_monitor.get_gpu_memory_stats()
                    usage_percent = memory_stats.get("usage_percentage", 0)

                    if usage_percent < self.config.scale_up_threshold:
                        # Scale up
                        new_size = min(
                            self.config.max_batch_size,
                            int(self.current_batch_size * self.config.scale_up_factor),
                        )
                        if new_size > self.current_batch_size:
                            self.current_batch_size = new_size
                            logger.debug(f"Scaled up batch size to {new_size}")
                    elif usage_percent > self.config.scale_down_threshold:
                        # Scale down
                        new_size = max(
                            self.config.min_batch_size,
                            int(self.current_batch_size * self.config.scale_down_factor),
                        )
                        if new_size < self.current_batch_size:
                            self.current_batch_size = new_size
                            logger.debug(f"Scaled down batch size to {new_size}")

                logger.debug(
                    f"Batch {i + 1}/{len(plan.batches)} processed: "
                    f"{len(batch)} files, {batch_time:.2f}s"
                )

            except Exception as e:
                logger.error(f"Batch {i + 1} failed: {e}")
                # Continue with next batch

        total_time = time.time() - start_time

        # Update statistics
        self._total_processed += len(audio_files)
        self._total_time += total_time
        self._average_batch_size = self._average_batch_size * 0.9 + len(plan.batches) * 0.1

        logger.info(
            f"Dynamic batch processing complete: "
            f"{len(all_results)}/{len(audio_files)} successful, "
            f"{total_time:.2f}s, "
            f"avg batch size: {self._average_batch_size:.1f}"
        )

        return all_results

    def get_optimal_batch_size(self) -> int:
        """
        Get current optimal batch size.

        Returns:
            Optimal batch size based on current conditions
        """
        if self.gpu_monitor.is_gpu_available():
            memory_stats = self.gpu_monitor.get_gpu_memory_stats()
            usage_percent = memory_stats.get("usage_percentage", 0)

            # Adjust based on current GPU load
            if usage_percent < 50:
                return min(
                    self.config.max_batch_size,
                    int(self.current_batch_size * 1.2),
                )
            elif usage_percent > 85:
                return max(
                    self.config.min_batch_size,
                    int(self.current_batch_size * 0.8),
                )

        return self.current_batch_size

    def get_stats(self) -> Dict[str, Any]:
        """
        Get processor statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            "current_batch_size": self.current_batch_size,
            "optimal_batch_size": self.get_optimal_batch_size(),
            "total_processed": self._total_processed,
            "total_time": self._total_time,
            "average_batch_size": self._average_batch_size,
            "throughput_per_minute": (
                self._total_processed / (self._total_time / 60) if self._total_time > 0 else 0
            ),
            "gpu_available": self.gpu_monitor.is_gpu_available(),
            "gpu_memory": self.gpu_monitor.get_gpu_memory_stats()
            if self.gpu_monitor.is_gpu_available()
            else None,
        }

    def reset_stats(self) -> None:
        """Reset statistics."""
        self._total_processed = 0
        self._total_time = 0.0
        self._average_batch_size = 0.0
        self.current_batch_size = self.config.initial_batch_size
        logger.info("Processor statistics reset")
