"""
Progress tracking service for batch processing.

Tracks single file, batch, and overall progress with ETA calculation using moving average.
"""

import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class ProgressConfig:
    """Configuration for progress tracking."""

    eta_window_size: int = 10  # Number of data points for moving average
    update_interval_ms: int = 100  # Minimum interval between updates


@dataclass
class FileProgress:
    """Progress tracking for a single file."""

    file_path: Path
    progress_percentage: float = 0.0
    status: str = "pending"  # pending, in_progress, completed, failed
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None

    @property
    def elapsed_seconds(self) -> float:
        """Calculate elapsed time in seconds."""
        if not self.start_time:
            return 0.0

        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()

    def start(self) -> None:
        """Start progress tracking."""
        self.status = "in_progress"
        self.start_time = datetime.now()
        logger.debug(f"Started processing {self.file_path.name}")

    def update(self, percentage: float) -> None:
        """Update progress percentage.

        Args:
            percentage: Progress percentage (0-100)
        """
        self.progress_percentage = max(0.0, min(100.0, percentage))

    def complete(self) -> None:
        """Mark as completed."""
        self.status = "completed"
        self.progress_percentage = 100.0
        self.end_time = datetime.now()
        logger.debug(f"Completed {self.file_path.name} in {self.elapsed_seconds:.2f}s")

    def fail(self, error: str) -> None:
        """Mark as failed.

        Args:
            error: Error message
        """
        self.status = "failed"
        self.error_message = error
        self.end_time = datetime.now()
        logger.warning(f"Failed {self.file_path.name}: {error}")


@dataclass
class BatchProgress:
    """Progress tracking for a batch of files."""

    batch_id: int
    total_files: int
    completed_files: int = 0
    failed_files: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    @property
    def progress_percentage(self) -> float:
        """Calculate batch progress percentage."""
        if self.total_files == 0:
            return 0.0
        return (self.completed_files / self.total_files) * 100

    @property
    def is_complete(self) -> bool:
        """Check if batch is complete."""
        return self.completed_files + self.failed_files >= self.total_files

    def update_file_completed(self) -> None:
        """Update that a file was completed."""
        self.completed_files += 1

    def update_file_failed(self) -> None:
        """Update that a file failed."""
        self.failed_files += 1

    def start(self) -> None:
        """Start batch processing."""
        self.start_time = datetime.now()

    def complete(self) -> None:
        """Mark batch as complete."""
        self.end_time = datetime.now()


@dataclass
class OverallProgress:
    """Overall progress tracking across all batches."""

    total_files: int
    total_batches: int
    current_batch: int = 0
    completed_files: int = 0
    failed_files: int = 0
    start_time: Optional[datetime] = None
    eta_window_size: int = 10

    # Internal tracking for ETA calculation
    _progress_history: List[tuple[datetime, int]] = field(default_factory=list)
    _speed_window: Deque[float] = field(default_factory=lambda: deque(maxlen=10))

    @property
    def overall_progress_percentage(self) -> float:
        """Calculate overall progress percentage."""
        if self.total_files == 0:
            return 0.0
        return (self.completed_files / self.total_files) * 100

    @property
    def success_rate(self) -> float:
        """Calculate success rate (0-1)."""
        total_processed = self.completed_files + self.failed_files
        if total_processed == 0:
            return 0.0
        return self.completed_files / total_processed

    def start_overall(self) -> None:
        """Start overall processing."""
        self.start_time = datetime.now()
        self._progress_history.clear()
        self._speed_window.clear()
        logger.info(f"Started processing {self.total_files} files in {self.total_batches} batches")

    def update_batch_progress(self, batch_id: int, completed_files: int, failed_files: int) -> None:
        """Update progress for a batch.

        Args:
            batch_id: Batch identifier
            completed_files: Number of completed files in this batch
            failed_files: Number of failed files in this batch
        """
        self.current_batch = batch_id
        self.completed_files = completed_files
        self.failed_files = failed_files

        # Record progress for ETA calculation
        if self.start_time:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            self._progress_history.append((datetime.now(), self.completed_files))

            # Calculate processing speed (files/second)
            if elapsed > 0 and self.completed_files > 0:
                speed = self.completed_files / elapsed
                self._speed_window.append(speed)

    def get_eta_seconds(self) -> float:
        """Calculate estimated time to completion using moving average.

        Returns:
            ETA in seconds (0 if cannot calculate)
        """
        if not self.start_time or self.completed_files == 0:
            return 0.0

        remaining_files = self.total_files - self.completed_files
        if remaining_files <= 0:
            return 0.0

        # Use moving average of processing speed
        if len(self._speed_window) > 0:
            avg_speed = sum(self._speed_window) / len(self._speed_window)
            if avg_speed > 0:
                return remaining_files / avg_speed

        # Fallback to overall average
        elapsed = (datetime.now() - self.start_time).total_seconds()
        if elapsed > 0:
            overall_speed = self.completed_files / elapsed
            if overall_speed > 0:
                return remaining_files / overall_speed

        return 0.0


class ProgressTracker:
    """Integrated progress tracker for batch processing."""

    def __init__(self, config: Optional[ProgressConfig] = None) -> None:
        """Initialize progress tracker.

        Args:
            config: Progress tracking configuration
        """
        self.config = config or ProgressConfig()
        self.file_progress: Dict[Path, FileProgress] = {}
        self.batch_progress: Dict[int, BatchProgress] = {}
        self.overall_progress = OverallProgress(
            total_files=0, total_batches=0, eta_window_size=self.config.eta_window_size
        )
        self._last_update_time: Optional[datetime] = None

    def start_overall(self, total_files: int, total_batches: int) -> None:
        """Start overall progress tracking.

        Args:
            total_files: Total number of files to process
            total_batches: Total number of batches
        """
        self.overall_progress = OverallProgress(
            total_files=total_files,
            total_batches=total_batches,
            eta_window_size=self.config.eta_window_size,
        )
        self.overall_progress.start_overall()

    def start_file(self, file_path: Path) -> None:
        """Start tracking a file.

        Args:
            file_path: Path to the file
        """
        if file_path not in self.file_progress:
            self.file_progress[file_path] = FileProgress(file_path=file_path)
        self.file_progress[file_path].start()

    def update_file_progress(self, file_path: Path, percentage: float) -> None:
        """Update file progress.

        Args:
            file_path: Path to the file
            percentage: Progress percentage (0-100)
        """
        if file_path in self.file_progress:
            self.file_progress[file_path].update(percentage)

    def complete_file(self, file_path: Path) -> None:
        """Mark file as completed.

        Args:
            file_path: Path to the file
        """
        if file_path in self.file_progress:
            self.file_progress[file_path].complete()

    def fail_file(self, file_path: Path, error: str) -> None:
        """Mark file as failed.

        Args:
            file_path: Path to the file
            error: Error message
        """
        if file_path in self.file_progress:
            self.file_progress[file_path].fail(error)

    def start_batch(self, batch_id: int, total_files: int) -> None:
        """Start tracking a batch.

        Args:
            batch_id: Batch identifier
            total_files: Number of files in this batch
        """
        if batch_id not in self.batch_progress:
            self.batch_progress[batch_id] = BatchProgress(
                batch_id=batch_id, total_files=total_files
            )
        self.batch_progress[batch_id].start()

    def update_batch_progress(self, batch_id: int, completed_files: int, failed_files: int) -> None:
        """Update batch progress.

        Args:
            batch_id: Batch identifier
            completed_files: Number of completed files
            failed_files: Number of failed files
        """
        # Update overall progress
        self.overall_progress.update_batch_progress(batch_id, completed_files, failed_files)

        # Update batch progress if tracking
        if batch_id in self.batch_progress:
            batch = self.batch_progress[batch_id]
            batch.completed_files = completed_files
            batch.failed_files = failed_files

    def get_overall_progress(self) -> OverallProgress:
        """Get current overall progress.

        Returns:
            Copy of overall progress object
        """
        return OverallProgress(
            total_files=self.overall_progress.total_files,
            total_batches=self.overall_progress.total_batches,
            current_batch=self.overall_progress.current_batch,
            completed_files=self.overall_progress.completed_files,
            failed_files=self.overall_progress.failed_files,
            start_time=self.overall_progress.start_time,
            eta_window_size=self.overall_progress.eta_window_size,
        )

    def get_eta_seconds(self) -> float:
        """Get estimated time to completion.

        Returns:
            ETA in seconds
        """
        return self.overall_progress.get_eta_seconds()

    def get_progress_summary(self) -> dict:
        """Get comprehensive progress summary.

        Returns:
            Dictionary with progress information
        """
        eta = self.get_eta_seconds()

        return {
            "overall_progress_percentage": round(
                self.overall_progress.overall_progress_percentage, 2
            ),
            "completed_files": self.overall_progress.completed_files,
            "failed_files": self.overall_progress.failed_files,
            "total_files": self.overall_progress.total_files,
            "current_batch": self.overall_progress.current_batch,
            "total_batches": self.overall_progress.total_batches,
            "success_rate": round(self.overall_progress.success_rate, 2),
            "eta_seconds": round(eta, 2),
            "eta_formatted": str(timedelta(seconds=int(eta))) if eta > 0 else "Unknown",
        }
