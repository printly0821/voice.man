"""
Progressive Batch Processor for WhisperX Pipeline

Phase 1 Quick Wins:
- Fixed batch size (8-16) with GPU memory-based auto-adjustment
- Progressive batch size scaling based on success/failure
- GPU memory monitoring for safe batch sizing
- EARS Requirements: E1, E2, S1

Reference: SPEC-GPUOPT-001 Phase 1
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from voice_man.services.gpu_monitor_service import GPUMonitorService, GPUMemoryThresholds

logger = logging.getLogger(__name__)


@dataclass
class BatchResult:
    """Result of batch processing."""

    results: List[Any] = field(default_factory=list)
    successful: int = 0
    failed: int = 0
    processing_time: float = 0.0
    batch_size_used: int = 0
    gpu_memory_peak_mb: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "successful": self.successful,
            "failed": self.failed,
            "processing_time": self.processing_time,
            "batch_size_used": self.batch_size_used,
            "gpu_memory_peak_mb": self.gpu_memory_peak_mb,
        }


@dataclass
class BatchConfig:
    """Batch processing configuration."""

    # Initial batch settings
    initial_batch_size: int = 8
    min_batch_size: int = 2
    max_batch_size: int = 32

    # Progressive scaling
    increase_factor: float = 1.5  # Increase by 50% on success
    decrease_factor: float = 0.5  # Decrease by 50% on failure

    # Memory thresholds
    memory_warning_percent: float = 75.0
    memory_critical_percent: float = 90.0

    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0  # seconds

    # Adaptive behavior
    enable_adaptive_batching: bool = True
    consecutive_success_threshold: int = 3  # Increase batch after N successes
    consecutive_failure_threshold: int = 2  # Decrease batch after N failures


class ProgressiveBatchProcessor:
    """
    Progressive batch processor with GPU memory awareness.

    Features:
    - Initial batch size with auto-adjustment based on GPU memory
    - Progressive scaling: increase on success, decrease on failure
    - GPU memory monitoring for safe batch sizing
    - OOM prevention through memory checks

    EARS Requirements:
    - E1: GPU memory check at batch start
    - E2: Auto-reduce batch size by 50% on memory shortage
    - S1: CPU fallback when GPU memory critical
    """

    def __init__(self, config: Optional[BatchConfig] = None):
        """
        Initialize Progressive Batch Processor.

        Args:
            config: Batch configuration (default: default config)
        """
        self.config = config or BatchConfig()

        # GPU monitor with custom thresholds
        self.gpu_monitor = GPUMonitorService(
            min_batch_size=self.config.min_batch_size,
            thresholds=GPUMemoryThresholds(
                warning_percent=self.config.memory_warning_percent,
                critical_percent=self.config.memory_critical_percent,
            ),
        )

        # Current batch size (starts at initial)
        self.current_batch_size = self.config.initial_batch_size

        # Statistics for adaptive behavior
        self.consecutive_successes = 0
        self.consecutive_failures = 0
        self.total_processed = 0
        self.total_successful = 0
        self.total_failed = 0

        logger.info(
            f"ProgressiveBatchProcessor initialized: "
            f"batch_size={self.config.initial_batch_size}-"
            f"{self.config.max_batch_size}"
        )

    def get_initial_batch_size(self) -> int:
        """
        Get initial batch size with GPU memory check.

        E1: GPU memory check at batch start
        E2: Auto-reduce batch size by 50% on memory shortage

        Returns:
            Recommended initial batch size
        """
        if not self.gpu_monitor.is_gpu_available():
            logger.info("GPU not available, using minimum batch size")
            return self.config.min_batch_size

        # Check current memory status
        memory_stats = self.gpu_monitor.get_gpu_memory_stats()
        if not memory_stats.get("available", False):
            return self.config.min_batch_size

        usage_percent = memory_stats.get("usage_percentage", 0)

        # Adjust based on current memory usage
        if usage_percent > 80:
            # High memory usage, reduce batch size
            recommended = max(
                self.config.min_batch_size,
                int(self.current_batch_size * self.config.decrease_factor),
            )
            logger.info(
                f"High memory usage ({usage_percent:.1f}%), reducing batch size to {recommended}"
            )
            return recommended
        elif usage_percent < 50:
            # Low memory usage, can use larger batch
            return min(
                self.config.max_batch_size,
                int(self.current_batch_size * self.config.increase_factor),
            )

        return self.current_batch_size

    def process_batch(
        self,
        items: List[Any],
        process_func: Callable[[List[Any]], List[Any]],
        batch_size: Optional[int] = None,
    ) -> BatchResult:
        """
        Process items in progressive batches.

        Args:
            items: Items to process
            process_func: Function to process batch (takes list, returns list)
            batch_size: Initial batch size (None to auto-determine)

        Returns:
            BatchResult with all results and statistics
        """
        start_time = time.time()

        # Determine batch size
        if batch_size is None:
            batch_size = self.get_initial_batch_size()

        # Clamp to configured limits
        batch_size = max(self.config.min_batch_size, min(self.config.max_batch_size, batch_size))

        all_results = []
        total_successful = 0
        total_failed = 0
        peak_memory_mb = 0.0

        # Process in batches
        i = 0
        current_batch = batch_size

        while i < len(items):
            batch = items[i : i + current_batch]
            batch_start_time = time.time()

            try:
                # Check GPU memory before processing
                if self.gpu_monitor.is_gpu_available():
                    memory_stats = self.gpu_monitor.get_gpu_memory_stats()
                    current_memory_mb = memory_stats.get("used_mb", 0)
                    peak_memory_mb = max(peak_memory_mb, current_memory_mb)

                    # Pre-processing memory check
                    recommended = self.gpu_monitor.get_recommended_batch_size(current_batch)
                    if recommended < current_batch:
                        logger.info(
                            f"Reducing batch size due to memory: {current_batch} -> {recommended}"
                        )
                        current_batch = recommended
                        batch = items[i : i + current_batch]

                # Process batch
                results = process_func(batch)
                batch_time = time.time() - batch_start_time

                # Count successes
                successful = len(results)
                failed = len(batch) - successful
                total_successful += successful
                total_failed += failed

                all_results.extend(results)

                # Update statistics for adaptive behavior
                if failed == 0:
                    self.consecutive_successes += 1
                    self.consecutive_failures = 0

                    # Consider increasing batch size
                    if (
                        self.config.enable_adaptive_batching
                        and self.consecutive_successes >= self.config.consecutive_success_threshold
                    ):
                        new_size = min(
                            self.config.max_batch_size,
                            int(current_batch * self.config.increase_factor),
                        )
                        if new_size > current_batch:
                            logger.info(
                                f"Increasing batch size after {self.consecutive_successes} successes: "
                                f"{current_batch} -> {new_size}"
                            )
                            current_batch = new_size
                            self.consecutive_successes = 0
                else:
                    self.consecutive_failures += 1
                    self.consecutive_successes = 0

                    # Consider decreasing batch size
                    if (
                        self.config.enable_adaptive_batching
                        and self.consecutive_failures >= self.config.consecutive_failure_threshold
                    ):
                        new_size = max(
                            self.config.min_batch_size,
                            int(current_batch * self.config.decrease_factor),
                        )
                        if new_size < current_batch:
                            logger.warning(
                                f"Decreasing batch size after {self.consecutive_failures} failures: "
                                f"{current_batch} -> {new_size}"
                            )
                            current_batch = new_size
                            self.consecutive_failures = 0

                logger.debug(
                    f"Batch [{i}:{i + len(batch)}] processed: "
                    f"{successful} successful, {failed} failed, "
                    f"{batch_time:.2f}s"
                )

            except Exception as e:
                # Handle batch processing error
                logger.error(f"Batch processing error: {e}")

                # Try with reduced batch size
                if current_batch > self.config.min_batch_size:
                    new_size = max(
                        self.config.min_batch_size,
                        int(current_batch * self.config.decrease_factor),
                    )
                    logger.warning(
                        f"Error occurred, reducing batch size: {current_batch} -> {new_size}"
                    )
                    current_batch = new_size
                    continue

                # Count all as failed
                total_failed += len(batch)
                self.consecutive_failures += 1
                self.consecutive_successes = 0

            i += current_batch

        # Update statistics
        processing_time = time.time() - start_time
        self.total_processed += len(items)
        self.total_successful += total_successful
        self.total_failed += total_failed

        result = BatchResult(
            results=all_results,
            successful=total_successful,
            failed=total_failed,
            processing_time=processing_time,
            batch_size_used=batch_size,
            gpu_memory_peak_mb=peak_memory_mb,
        )

        logger.info(
            f"Batch processing complete: {total_successful}/{len(items)} successful, "
            f"{processing_time:.2f}s, "
            f"peak GPU memory: {peak_memory_mb:.0f}MB"
        )

        return result

    def process_with_retry(
        self,
        items: List[Any],
        process_func: Callable[[List[Any]], List[Any]],
        batch_size: Optional[int] = None,
    ) -> BatchResult:
        """
        Process items with automatic retry on failure.

        Args:
            items: Items to process
            process_func: Function to process batch
            batch_size: Initial batch size

        Returns:
            BatchResult with all results
        """
        max_retries = self.config.max_retries
        retry_delay = self.config.retry_delay

        for attempt in range(max_retries + 1):
            try:
                result = self.process_batch(items, process_func, batch_size)
                return result

            except Exception as e:
                if attempt < max_retries:
                    logger.warning(
                        f"Batch processing failed (attempt {attempt + 1}/{max_retries + 1}): {e}, "
                        f"retrying in {retry_delay}s..."
                    )
                    time.sleep(retry_delay)

                    # Reduce batch size for retry
                    if batch_size and batch_size > self.config.min_batch_size:
                        batch_size = max(
                            self.config.min_batch_size,
                            int(batch_size * self.config.decrease_factor),
                        )
                        logger.info(f"Reduced batch size for retry: {batch_size}")
                else:
                    logger.error(f"Batch processing failed after {max_retries + 1} attempts: {e}")
                    raise

        # Should not reach here
        return BatchResult()

    def get_optimal_batch_size(self) -> int:
        """
        Get optimal batch size based on history.

        Returns:
            Recommended batch size
        """
        if not self.config.enable_adaptive_batching:
            return self.config.initial_batch_size

        # Adjust based on recent performance
        if self.consecutive_failures >= self.config.consecutive_failure_threshold:
            return max(
                self.config.min_batch_size,
                int(self.current_batch_size * self.config.decrease_factor),
            )
        elif self.consecutive_successes >= self.config.consecutive_success_threshold:
            return min(
                self.config.max_batch_size,
                int(self.current_batch_size * self.config.increase_factor),
            )

        # Also consider GPU memory
        if self.gpu_monitor.is_gpu_available():
            recommended = self.gpu_monitor.get_recommended_batch_size(self.current_batch_size)
            return max(self.config.min_batch_size, min(self.config.max_batch_size, recommended))

        return self.current_batch_size

    def get_stats(self) -> Dict[str, Any]:
        """
        Get processor statistics.

        Returns:
            Dictionary with statistics
        """
        success_rate = (
            self.total_successful / self.total_processed * 100 if self.total_processed > 0 else 0
        )

        return {
            "current_batch_size": self.current_batch_size,
            "optimal_batch_size": self.get_optimal_batch_size(),
            "total_processed": self.total_processed,
            "total_successful": self.total_successful,
            "total_failed": self.total_failed,
            "success_rate": success_rate,
            "consecutive_successes": self.consecutive_successes,
            "consecutive_failures": self.consecutive_failures,
            "gpu_available": self.gpu_monitor.is_gpu_available(),
            "gpu_memory": self.gpu_monitor.get_gpu_memory_stats()
            if self.gpu_monitor.is_gpu_available()
            else None,
        }

    def reset_stats(self) -> None:
        """Reset statistics."""
        self.consecutive_successes = 0
        self.consecutive_failures = 0
        self.total_processed = 0
        self.total_successful = 0
        self.total_failed = 0
        self.current_batch_size = self.config.initial_batch_size
        logger.info("Processor statistics reset")
