"""
Progress Tracker with ETA Calculation

Real-time progress tracking with:
- Percentage-based progress display
- ETA (Estimated Time of Arrival) calculation
- Pretty console output with progress bars
- Batch-level and file-level tracking
- Performance metrics (files/sec, avg time)
"""

import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ProgressUpdate:
    """
    Progress update information.

    Attributes:
        current: Current progress value
        total: Total target value
        percentage: Progress percentage (0-100)
        eta_seconds: Estimated time remaining in seconds
        elapsed_seconds: Elapsed time since start
        rate_per_second: Processing rate (items per second)
        message: Optional progress message
    """

    current: int
    total: int
    percentage: float
    eta_seconds: float
    elapsed_seconds: float
    rate_per_second: float
    message: str = ""

    @property
    def eta_formatted(self) -> str:
        """Get ETA as formatted string (HH:MM:SS)."""
        hours = int(self.eta_seconds // 3600)
        minutes = int((self.eta_seconds % 3600) // 60)
        seconds = int(self.eta_seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    @property
    def elapsed_formatted(self) -> str:
        """Get elapsed time as formatted string (HH:MM:SS)."""
        hours = int(self.elapsed_seconds // 3600)
        minutes = int((self.elapsed_seconds % 3600) // 60)
        seconds = int(self.elapsed_seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


@dataclass
class BatchProgress:
    """Progress information for a batch."""

    batch_number: int
    total_batches: int
    files_processed: int
    total_files: int
    files_failed: int
    start_time: datetime
    last_update_time: datetime

    @property
    def batch_percentage(self) -> float:
        """Batch progress percentage."""
        if self.total_batches == 0:
            return 0.0
        return (self.batch_number / self.total_batches) * 100

    @property
    def file_percentage(self) -> float:
        """File progress percentage."""
        if self.total_files == 0:
            return 0.0
        return (self.files_processed / self.total_files) * 100


class ProgressTracker:
    """
    Real-time progress tracker with ETA calculation.

    Features:
    - Progress bar display with ASCII art
    - ETA calculation based on recent processing rate
    - Elapsed time tracking
    - Processing rate calculation (files/second)
    - Batch-level and file-level tracking
    - Performance metrics

    Usage:
        tracker = ProgressTracker(total_items=100)

        for i in range(100):
            # Process item
            tracker.update(current=i + 1, message=f"Processing item {i + 1}")

        # Get final progress
        progress = tracker.get_progress()
        print(f"Completed in {progress.elapsed_formatted}")

    Output format:
        [████████████░░░░░░░░] Batch 3/10 (30%) | Processed: 15/50 | Failed: 2 | ETA: 00:15:23
    """

    def __init__(
        self,
        total_items: int,
        bar_width: int = 30,
        update_interval: float = 1.0,
        enable_console: bool = True,
    ):
        """
        Initialize progress tracker.

        Args:
            total_items: Total number of items to process
            bar_width: Width of progress bar in characters
            update_interval: Minimum seconds between console updates
            enable_console: Whether to print progress to console
        """
        self.total_items = total_items
        self.bar_width = bar_width
        self.update_interval = update_interval
        self.enable_console = enable_console

        # Progress state
        self.current = 0
        self.failed = 0
        self.start_time = datetime.now(timezone.utc)
        self.last_update_time = self.start_time
        self.last_console_update = 0.0

        # ETA calculation state
        self.recent_samples: List[float] = []
        self.max_samples = 10  # Number of recent samples for ETA calculation

        # Custom formatters
        self.message_formatter: Optional[Callable[[ProgressUpdate], str]] = None

        logger.info(f"ProgressTracker initialized: {total_items} items")

    def update(
        self,
        current: int,
        failed: int = 0,
        message: str = "",
        force_update: bool = False,
    ) -> ProgressUpdate:
        """
        Update progress and display if needed.

        Args:
            current: Current progress value
            failed: Number of failed items
            message: Optional progress message
            force_update: Force console update regardless of interval

        Returns:
            ProgressUpdate with current progress information
        """
        now = datetime.now(timezone.utc)
        self.current = current
        self.failed = failed
        self.last_update_time = now

        # Calculate elapsed time
        elapsed = (now - self.start_time).total_seconds()

        # Calculate rate
        if elapsed > 0:
            rate = current / elapsed
            # Store sample for ETA calculation
            self.recent_samples.append(rate)
            if len(self.recent_samples) > self.max_samples:
                self.recent_samples.pop(0)
        else:
            rate = 0.0

        # Calculate ETA using average of recent samples
        if self.recent_samples and sum(self.recent_samples) > 0:
            avg_rate = sum(self.recent_samples) / len(self.recent_samples)
            remaining = self.total_items - current
            eta = remaining / avg_rate if avg_rate > 0 else 0.0
        else:
            eta = 0.0

        # Create progress update
        percentage = (current / self.total_items * 100) if self.total_items > 0 else 0.0

        progress = ProgressUpdate(
            current=current,
            total=self.total_items,
            percentage=percentage,
            eta_seconds=eta,
            elapsed_seconds=elapsed,
            rate_per_second=rate,
            message=message,
        )

        # Display progress
        if self.enable_console:
            current_time = time.time()
            if force_update or (current_time - self.last_console_update) >= self.update_interval:
                self._display_progress(progress)
                self.last_console_update = current_time

        return progress

    def get_progress(self) -> ProgressUpdate:
        """Get current progress without updating display."""
        now = datetime.now(timezone.utc)
        elapsed = (now - self.start_time).total_seconds()

        rate = self.current / elapsed if elapsed > 0 else 0.0
        percentage = (self.current / self.total_items * 100) if self.total_items > 0 else 0.0

        remaining = self.total_items - self.current
        avg_rate = (
            sum(self.recent_samples) / len(self.recent_samples) if self.recent_samples else rate
        )
        eta = remaining / avg_rate if avg_rate > 0 else 0.0

        return ProgressUpdate(
            current=self.current,
            total=self.total_items,
            percentage=percentage,
            eta_seconds=eta,
            elapsed_seconds=elapsed,
            rate_per_second=rate,
        )

    def _display_progress(self, progress: ProgressUpdate) -> None:
        """Display progress to console."""
        # Clear line and display progress bar
        sys.stdout.write("\r" + " " * 100 + "\r")  # Clear line

        if self.message_formatter:
            # Use custom formatter
            message = self.message_formatter(progress)
            sys.stdout.write(message)
        else:
            # Use default formatter
            progress_bar = self._create_progress_bar(progress.percentage)
            status_line = (
                f"{progress_bar} "
                f"Processed: {progress.current}/{self.total_items} | "
                f"Failed: {self.failed} | "
                f"ETA: {progress.eta_formatted}"
            )

            if progress.message:
                status_line += f" | {progress.message}"

            sys.stdout.write(status_line)

        sys.stdout.flush()

    def _create_progress_bar(self, percentage: float) -> str:
        """Create ASCII progress bar."""
        filled = int(self.bar_width * percentage / 100)
        bar = "█" * filled + "░" * (self.bar_width - filled)
        return f"[{bar}] {percentage:.1f}%"

    def set_message_formatter(self, formatter: Callable[[ProgressUpdate], str]) -> None:
        """
        Set custom message formatter.

        Args:
            formatter: Function that takes ProgressUpdate and returns formatted string
        """
        self.message_formatter = formatter

    def finish(self, message: str = "Complete!") -> None:
        """
        Mark progress as complete and display final message.

        Args:
            message: Final completion message
        """
        # Final update
        progress = self.update(
            current=self.total_items,
            failed=self.failed,
            message=message,
            force_update=True,
        )

        # Move to next line
        sys.stdout.write("\n")
        sys.stdout.flush()

        logger.info(
            f"Progress completed: {self.total_items} items in {progress.elapsed_formatted} "
            f"(avg: {progress.rate_per_second:.2f} items/sec)"
        )

    def get_summary(self) -> Dict[str, Any]:
        """
        Get progress summary as dictionary.

        Returns:
            Dictionary with progress summary
        """
        progress = self.get_progress()

        return {
            "total_items": self.total_items,
            "current": self.current,
            "failed": self.failed,
            "percentage": progress.percentage,
            "elapsed_seconds": progress.elapsed_seconds,
            "elapsed_formatted": progress.elapsed_formatted,
            "eta_seconds": progress.eta_seconds,
            "eta_formatted": progress.eta_formatted,
            "rate_per_second": progress.rate_per_second,
            "start_time": self.start_time.isoformat(),
            "end_time": self.last_update_time.isoformat(),
        }


class BatchProgressTracker:
    """
    Specialized progress tracker for batch processing.

    Tracks both batch-level and file-level progress with separate progress bars.

    Usage:
        tracker = BatchProgressTracker(
            total_batches=10,
            total_files=100,
            batch_size=10
        )

        # Update for batch start
        tracker.start_batch(1)

        # Update for file processing
        tracker.update_file(5)

        # Complete batch
        tracker.complete_batch()
    """

    def __init__(
        self,
        total_batches: int,
        total_files: int,
        batch_size: int,
        enable_console: bool = True,
    ):
        """
        Initialize batch progress tracker.

        Args:
            total_batches: Total number of batches
            total_files: Total number of files
            batch_size: Number of files per batch
            enable_console: Whether to display progress
        """
        self.total_batches = total_batches
        self.total_files = total_files
        self.batch_size = batch_size
        self.enable_console = enable_console

        # Batch progress
        self.current_batch = 0
        self.batch_files_processed = 0
        self.total_files_processed = 0
        self.total_files_failed = 0

        # File progress within current batch
        self.current_file_in_batch = 0

        # Start time
        self.start_time = datetime.now(timezone.utc)

        # Progress trackers
        self.batch_tracker = ProgressTracker(
            total_items=total_batches,
            enable_console=False,  # We'll handle display ourselves
        )
        self.file_tracker = ProgressTracker(
            total_items=total_files,
            enable_console=False,
        )

        logger.info(
            f"BatchProgressTracker initialized: {total_batches} batches, "
            f"{total_files} files, {batch_size} files/batch"
        )

    def start_batch(self, batch_number: int) -> None:
        """
        Start processing a new batch.

        Args:
            batch_number: Batch number (1-indexed)
        """
        self.current_batch = batch_number
        self.batch_files_processed = 0
        self.current_file_in_batch = 0

        if self.enable_console:
            print(f"\n{'=' * 70}")
            print(f"BATCH {batch_number}/{self.total_batches}")
            print(f"{'=' * 70}")

        logger.info(f"Started batch {batch_number}/{self.total_batches}")

    def update_file(self, files_processed: int, failed: int = 0) -> None:
        """
        Update file processing progress within current batch.

        Args:
            files_processed: Number of files processed in current batch
            failed: Number of failed files
        """
        self.current_file_in_batch = files_processed
        self.batch_files_processed = files_processed
        self.total_files_processed += files_processed - self.batch_files_processed
        self.total_files_failed += failed

        # Update trackers
        self.file_tracker.update(
            current=self.total_files_processed,
            failed=self.total_files_failed,
        )

        # Display progress
        if self.enable_console:
            self._display_batch_progress()

    def complete_batch(self, successful: int, failed: int) -> None:
        """
        Mark current batch as complete.

        Args:
            successful: Number of successful files in batch
            failed: Number of failed files in batch
        """
        # Update totals
        self.total_files_processed += successful
        self.total_files_failed += failed

        # Update batch tracker
        self.batch_tracker.update(current=self.current_batch)

        if self.enable_console:
            print(
                f"\nBatch {self.current_batch} complete: {successful} successful, {failed} failed"
            )

        logger.info(
            f"Completed batch {self.current_batch}: {successful} successful, {failed} failed"
        )

    def _display_batch_progress(self) -> None:
        """Display batch processing progress."""
        # Calculate progress
        batch_percentage = (self.current_file_in_batch / self.batch_size) * 100
        overall_percentage = (self.total_files_processed / self.total_files) * 100

        # Create progress bars
        batch_bar = self._create_progress_bar(batch_percentage, width=20)
        overall_bar = self._create_progress_bar(overall_percentage, width=20)

        # Format output
        output = (
            f"\r{batch_bar} Batch: {self.current_file_in_batch}/{self.batch_size} ({batch_percentage:.0f}%) | "
            f"{overall_bar} Overall: {self.total_files_processed}/{self.total_files} ({overall_percentage:.0f}%) | "
            f"Failed: {self.total_files_failed}"
        )

        sys.stdout.write(output)
        sys.stdout.flush()

    def _create_progress_bar(self, percentage: float, width: int = 20) -> str:
        """Create progress bar with specified width."""
        filled = int(width * percentage / 100)
        bar = "█" * filled + "░" * (width - filled)
        return f"[{bar}]"

    def finish(self) -> None:
        """Mark all batches as complete and display summary."""
        # Move to next line
        if self.enable_console:
            sys.stdout.write("\n")
            sys.stdout.flush()

        elapsed = (datetime.now(timezone.utc) - self.start_time).total_seconds()

        logger.info(
            f"All batches complete: {self.total_files_processed}/{self.total_files} files processed, "
            f"{self.total_files_failed} failed in {elapsed:.1f}s"
        )

    def get_summary(self) -> Dict[str, Any]:
        """
        Get batch processing summary.

        Returns:
            Dictionary with batch processing summary
        """
        elapsed = (datetime.now(timezone.utc) - self.start_time).total_seconds()

        return {
            "total_batches": self.total_batches,
            "current_batch": self.current_batch,
            "total_files": self.total_files,
            "batch_size": self.batch_size,
            "total_files_processed": self.total_files_processed,
            "total_files_failed": self.total_files_failed,
            "success_rate": (
                self.total_files_processed / self.total_files * 100 if self.total_files > 0 else 0
            ),
            "elapsed_seconds": elapsed,
            "rate_per_second": self.total_files_processed / elapsed if elapsed > 0 else 0,
        }
