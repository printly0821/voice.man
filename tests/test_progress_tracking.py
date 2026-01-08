"""
Unit tests for progress tracking service.

Tests single file, batch, and overall progress tracking with ETA calculation.
"""

import time
import pytest
from pathlib import Path

from voice_man.services.progress_service import (
    ProgressTracker,
    FileProgress,
    BatchProgress,
    OverallProgress,
    ProgressConfig,
)


class TestProgressConfig:
    """Test progress configuration."""

    def test_default_config(self):
        """Test default configuration."""
        config = ProgressConfig()
        assert config.eta_window_size == 10
        assert config.update_interval_ms == 100

    def test_custom_config(self):
        """Test custom configuration."""
        config = ProgressConfig(eta_window_size=20, update_interval_ms=500)
        assert config.eta_window_size == 20
        assert config.update_interval_ms == 500


class TestFileProgress:
    """Test single file progress tracking."""

    def test_initial_state(self):
        """Test initial file progress state."""
        progress = FileProgress(file_path=Path("test.m4a"))
        assert progress.file_path == Path("test.m4a")
        assert progress.progress_percentage == 0.0
        assert progress.status == "pending"
        assert progress.start_time is None
        assert progress.end_time is None
        assert progress.elapsed_seconds == 0.0

    def test_start_progress(self):
        """Test starting progress tracking."""
        progress = FileProgress(file_path=Path("test.m4a"))
        progress.start()
        assert progress.status == "in_progress"
        assert progress.start_time is not None

    def test_update_progress(self):
        """Test updating progress."""
        progress = FileProgress(file_path=Path("test.m4a"))
        progress.start()
        progress.update(50.0)
        assert progress.progress_percentage == 50.0

    def test_complete_progress(self):
        """Test completing progress."""
        progress = FileProgress(file_path=Path("test.m4a"))
        progress.start()
        time.sleep(0.01)
        progress.complete()
        assert progress.status == "completed"
        assert progress.progress_percentage == 100.0
        assert progress.end_time is not None
        assert progress.elapsed_seconds > 0

    def test_fail_progress(self):
        """Test failing progress."""
        progress = FileProgress(file_path=Path("test.m4a"))
        progress.start()
        progress.fail("Test error")
        assert progress.status == "failed"
        assert progress.error_message == "Test error"
        assert progress.end_time is not None


class TestBatchProgress:
    """Test batch progress tracking."""

    def test_initial_state(self):
        """Test initial batch progress state."""
        progress = BatchProgress(batch_id=1, total_files=10)
        assert progress.batch_id == 1
        assert progress.total_files == 10
        assert progress.completed_files == 0
        assert progress.failed_files == 0
        assert progress.progress_percentage == 0.0

    def test_update_progress(self):
        """Test updating batch progress."""
        progress = BatchProgress(batch_id=1, total_files=10)
        progress.update_file_completed()
        progress.update_file_completed()
        assert progress.completed_files == 2
        assert progress.progress_percentage == 20.0

    def test_update_with_failures(self):
        """Test updating with failed files."""
        progress = BatchProgress(batch_id=1, total_files=10)
        progress.update_file_completed()
        progress.update_file_failed()
        progress.update_file_completed()
        assert progress.completed_files == 2
        assert progress.failed_files == 1

    def test_is_complete(self):
        """Test checking if batch is complete."""
        progress = BatchProgress(batch_id=1, total_files=5)
        assert progress.is_complete is False

        for _ in range(5):
            progress.update_file_completed()
        assert progress.is_complete is True


class TestOverallProgress:
    """Test overall progress tracking."""

    def test_initial_state(self):
        """Test initial overall progress state."""
        progress = OverallProgress(total_files=100, total_batches=20)
        assert progress.total_files == 100
        assert progress.total_batches == 20
        assert progress.current_batch == 0
        assert progress.completed_files == 0
        assert progress.overall_progress_percentage == 0.0

    def test_update_batch_progress(self):
        """Test updating batch progress."""
        progress = OverallProgress(total_files=100, total_batches=20)
        progress.update_batch_progress(batch_id=1, completed_files=5, failed_files=0)
        assert progress.current_batch == 1
        assert progress.completed_files == 5

    def test_eta_calculation(self):
        """Test ETA calculation using moving average."""
        progress = OverallProgress(total_files=100, total_batches=10, eta_window_size=5)

        # Simulate processing speed
        progress.start_overall()
        time.sleep(0.1)
        progress.update_batch_progress(batch_id=1, completed_files=10, failed_files=0)

        eta = progress.get_eta_seconds()
        assert eta >= 0  # ETA should be non-negative


class TestProgressTracker:
    """Test integrated progress tracker."""

    @pytest.fixture
    def tracker(self):
        """Create a progress tracker."""
        config = ProgressConfig(eta_window_size=5)
        return ProgressTracker(config=config)

    def test_initialization(self, tracker):
        """Test tracker initialization."""
        assert tracker.config.eta_window_size == 5
        assert tracker.overall_progress.total_files == 0

    def test_start_file(self, tracker):
        """Test starting file tracking."""
        file_path = Path("test.m4a")
        tracker.start_file(file_path)
        assert file_path in tracker.file_progress
        assert tracker.file_progress[file_path].status == "in_progress"

    def test_update_file_progress(self, tracker):
        """Test updating file progress."""
        file_path = Path("test.m4a")
        tracker.start_file(file_path)
        tracker.update_file_progress(file_path, 50.0)
        assert tracker.file_progress[file_path].progress_percentage == 50.0

    def test_complete_file(self, tracker):
        """Test completing file."""
        file_path = Path("test.m4a")
        tracker.start_file(file_path)
        tracker.complete_file(file_path)
        assert tracker.file_progress[file_path].status == "completed"

    def test_fail_file(self, tracker):
        """Test failing file."""
        file_path = Path("test.m4a")
        tracker.start_file(file_path)
        tracker.fail_file(file_path, "Test error")
        assert tracker.file_progress[file_path].status == "failed"
        assert tracker.file_progress[file_path].error_message == "Test error"

    def test_get_overall_progress(self, tracker):
        """Test getting overall progress."""
        tracker.start_overall(total_files=100, total_batches=20)
        progress = tracker.get_overall_progress()
        assert progress.total_files == 100
        assert progress.total_batches == 20

    def test_get_eta_seconds(self, tracker):
        """Test ETA calculation."""
        tracker.start_overall(total_files=100, total_batches=10)

        # Simulate some progress
        tracker.update_batch_progress(batch_id=1, completed_files=10, failed_files=0)
        time.sleep(0.1)
        tracker.update_batch_progress(batch_id=2, completed_files=20, failed_files=0)

        eta = tracker.get_eta_seconds()
        assert eta >= 0

    def test_get_progress_summary(self, tracker):
        """Test getting progress summary."""
        tracker.start_overall(total_files=100, total_batches=20)
        tracker.update_batch_progress(batch_id=1, completed_files=5, failed_files=0)

        summary = tracker.get_progress_summary()
        assert "overall_progress_percentage" in summary
        assert "completed_files" in summary
        assert "failed_files" in summary
        assert "eta_seconds" in summary
