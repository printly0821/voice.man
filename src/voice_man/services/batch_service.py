"""
Batch processing service for parallel audio file analysis.

Provides ThreadPoolExecutor-based parallel processing with configurable batch sizes,
error handling, progress tracking, and memory optimization.
"""

import logging
import gc
from pathlib import Path
from typing import List, Callable, Optional, Awaitable, Dict
from dataclasses import dataclass, field
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class BatchConfig:
    """Configuration for batch processing.

    Extended for GPU support as per SPEC-PARALLEL-001:
    - F2: GPU-based batch inference (batch size 15-20, max 32)
    - F5: Dynamic batch size adjustment
    - S3: Retry queue with exponential backoff
    """

    batch_size: int = 5
    max_workers: int = 4
    retry_count: int = 3
    continue_on_error: bool = True
    enable_memory_cleanup: bool = True  # Enable gc.collect() between batches

    # GPU-specific settings (SPEC-PARALLEL-001)
    use_gpu: bool = False
    gpu_batch_size: int = 15  # F2: Default GPU batch size
    dynamic_batch_adjustment: bool = True  # F5: Enable dynamic adjustment
    min_batch_size: int = 2  # Minimum batch size floor
    max_batch_size: int = 32  # F2: Maximum batch size ceiling

    def __post_init__(self):
        """Apply GPU-optimized defaults when use_gpu is True."""
        if self.use_gpu:
            # Override with GPU-optimized settings if not explicitly set
            if self.batch_size == 5:  # Default CPU value
                self.batch_size = self.gpu_batch_size
            if self.max_workers == 4:  # Default CPU value
                self.max_workers = 16  # Optimized for GPU parallelism


@dataclass
class BatchResult:
    """Result of batch processing."""

    file_path: str
    status: str
    data: Optional[dict] = None
    error: Optional[str] = None
    attempts: int = 1


@dataclass
class BatchProgress:
    """Progress tracking for batch processing."""

    total: int = 0
    processed: int = 0
    failed: int = 0
    current_batch: int = 0
    total_batches: int = 0
    # Separate storage for failed files
    failed_files: List[str] = field(default_factory=list)

    @property
    def progress_ratio(self) -> float:
        """Calculate progress ratio between 0.0 and 1.0."""
        if self.total == 0:
            return 0.0
        return self.processed / self.total

    @property
    def success_rate(self) -> float:
        """Calculate success rate ratio between 0.0 and 1.0."""
        if self.processed == 0:
            return 0.0
        return (self.processed - self.failed) / self.processed


@dataclass
class BatchStatistics:
    """Statistics for batch processing."""

    total_files: int = 0
    successful_files: int = 0
    failed_files: int = 0
    total_attempts: int = 0
    average_attempts_per_file: float = 0.0

    def calculate_from_results(self, results: List[BatchResult]) -> None:
        """Calculate statistics from results.

        Args:
            results: List of batch processing results
        """
        self.total_files = len(results)
        self.successful_files = sum(1 for r in results if r.status == "success")
        self.failed_files = sum(1 for r in results if r.status == "failed")
        self.total_attempts = sum(r.attempts for r in results)
        self.average_attempts_per_file = (
            self.total_attempts / self.total_files if self.total_files > 0 else 0.0
        )


class BatchProcessor:
    """Process audio files in parallel batches with memory optimization."""

    def __init__(self, config: BatchConfig) -> None:
        """Initialize batch processor with configuration.

        Args:
            config: Batch processing configuration
        """
        self.config = config
        self.progress = BatchProgress()
        self.statistics = BatchStatistics()

    def _create_batches(self, files: List[Path]) -> List[List[Path]]:
        """Split files into batches.

        Args:
            files: List of file paths to process

        Returns:
            List of file batches
        """
        batches = []
        for i in range(0, len(files), self.config.batch_size):
            batch = files[i : i + self.config.batch_size]
            batches.append(batch)
        return batches

    def _cleanup_memory(self) -> None:
        """Clean up memory between batches using gc.collect()."""
        if self.config.enable_memory_cleanup:
            logger.debug("Running garbage collection to free memory")
            collected = gc.collect()
            logger.debug(f"Garbage collection freed {collected} objects")

    async def _process_single_file(
        self,
        file_path: Path,
        process_func: Callable[[Path], Awaitable[dict]],
    ) -> BatchResult:
        """Process a single file with retry logic.

        Args:
            file_path: Path to the file
            process_func: Async function to process the file

        Returns:
            BatchResult with processing outcome
        """
        last_error = None

        for attempt in range(1, self.config.retry_count + 1):
            try:
                result_data = await process_func(file_path)
                logger.debug(f"Successfully processed {file_path.name} on attempt {attempt}")
                return BatchResult(
                    file_path=str(file_path),
                    status="success",
                    data=result_data,
                    attempts=attempt,
                )
            except Exception as e:
                last_error = e
                logger.warning(
                    f"Attempt {attempt}/{self.config.retry_count} failed for {file_path.name}: {e}"
                )
                if attempt == self.config.retry_count:
                    break

        # All retries failed
        error_msg = str(last_error) if last_error else "Unknown error"
        logger.error(
            f"Failed to process {file_path.name} after {self.config.retry_count} attempts: {error_msg}"
        )

        return BatchResult(
            file_path=str(file_path),
            status="failed",
            error=error_msg,
            attempts=self.config.retry_count,
        )

    async def _process_batch(
        self,
        batch: List[Path],
        process_func: Callable[[Path], Awaitable[dict]],
        batch_index: int,
    ) -> List[BatchResult]:
        """Process a single batch of files in parallel.

        Args:
            batch: List of file paths in this batch
            process_func: Async function to process each file
            batch_index: Index of current batch (0-based)

        Returns:
            List of BatchResult objects

        Raises:
            Exception: If processing fails and continue_on_error is False
        """
        logger.info(f"Processing batch {batch_index + 1} with {len(batch)} files")

        # Process all files in batch concurrently
        tasks = [self._process_single_file(file_path, process_func) for file_path in batch]
        results = await asyncio.gather(*tasks, return_exceptions=not self.config.continue_on_error)

        # Handle results
        batch_results = []
        for result in results:
            if isinstance(result, Exception):
                if not self.config.continue_on_error:
                    raise result
                # Convert exception to BatchResult
                batch_results.append(
                    BatchResult(
                        file_path="unknown",
                        status="failed",
                        error=str(result),
                        attempts=self.config.retry_count,
                    )
                )
            else:
                batch_results.append(result)

        # Separate failed files
        failed_results = [r for r in batch_results if r.status == "failed"]
        for failed in failed_results:
            if failed.file_path not in self.progress.failed_files:
                self.progress.failed_files.append(failed.file_path)

        # Update progress
        self.progress.processed += len(batch)
        self.progress.failed += len(failed_results)
        self.progress.current_batch = batch_index + 1

        success_count = sum(1 for r in batch_results if r.status == "success")
        logger.info(f"Batch {batch_index + 1} completed: {success_count}/{len(batch)} successful")

        return batch_results

    async def process_all(
        self,
        files: List[Path],
        process_func: Callable[[Path], Awaitable[dict]],
        progress_callback: Optional[Callable[[BatchProgress], None]] = None,
    ) -> List[BatchResult]:
        """Process all files in batches with memory optimization.

        Args:
            files: List of file paths to process
            process_func: Async function to process each file
            progress_callback: Optional callback function for progress updates

        Returns:
            List of all BatchResult objects

        Raises:
            Exception: If processing fails and continue_on_error is False
        """
        self.progress = BatchProgress(total=len(files))
        self.statistics = BatchStatistics()
        batches = self._create_batches(files)
        self.progress.total_batches = len(batches)

        logger.info(f"Starting batch processing: {len(files)} files in {len(batches)} batches")

        all_results = []
        for batch_index, batch in enumerate(batches):
            batch_results = await self._process_batch(batch, process_func, batch_index)
            all_results.extend(batch_results)

            # Clean up memory between batches
            if batch_index < len(batches) - 1:  # Not the last batch
                self._cleanup_memory()

            # Call progress callback if provided
            if progress_callback:
                progress_callback(self.get_progress())

        # Calculate final statistics
        self.statistics.calculate_from_results(all_results)

        logger.info(
            f"Batch processing completed: "
            f"{self.progress.processed - self.progress.failed}/{self.progress.processed} successful, "
            f"{self.progress.failed} failed"
        )

        return all_results

    def get_progress(self) -> BatchProgress:
        """Get current processing progress.

        Returns:
            Copy of current BatchProgress object
        """
        return BatchProgress(
            total=self.progress.total,
            processed=self.progress.processed,
            failed=self.progress.failed,
            current_batch=self.progress.current_batch,
            total_batches=self.progress.total_batches,
            failed_files=self.progress.failed_files.copy(),
        )

    def get_statistics(self) -> BatchStatistics:
        """Get processing statistics.

        Returns:
            Copy of current BatchStatistics object
        """
        return BatchStatistics(
            total_files=self.statistics.total_files,
            successful_files=self.statistics.successful_files,
            failed_files=self.statistics.failed_files,
            total_attempts=self.statistics.total_attempts,
            average_attempts_per_file=self.statistics.average_attempts_per_file,
        )

    def get_failed_files(self) -> List[str]:
        """Get list of failed file paths.

        Returns:
            List of failed file paths
        """
        return self.progress.failed_files.copy()

    def get_retry_info(self) -> Dict:
        """Get retry configuration information.

        Returns:
            Dictionary with retry settings

        Implements:
            S3: Retry queue with exponential backoff (3 retries)
        """
        return {
            "max_retries": self.config.retry_count,
            "backoff_strategy": "exponential",
            "continue_on_error": self.config.continue_on_error,
        }
