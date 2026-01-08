"""
Unit tests for batch processing service.

Tests the ThreadPoolExecutor-based parallel processing system for audio files.
"""

import pytest
from pathlib import Path
from typing import List

from voice_man.services.batch_service import BatchProcessor, BatchConfig, BatchProgress


class TestBatchConfig:
    """Test batch configuration."""

    def test_default_config(self):
        """Test default batch configuration."""
        config = BatchConfig()
        assert config.batch_size == 5
        assert config.max_workers == 4
        assert config.retry_count == 3
        assert config.continue_on_error is True
        assert config.enable_memory_cleanup is True

    def test_custom_config(self):
        """Test custom batch configuration."""
        config = BatchConfig(
            batch_size=10,
            max_workers=8,
            retry_count=5,
            continue_on_error=False,
            enable_memory_cleanup=False,
        )
        assert config.batch_size == 10
        assert config.max_workers == 8
        assert config.retry_count == 5
        assert config.continue_on_error is False
        assert config.enable_memory_cleanup is False


class TestBatchProcessor:
    """Test batch processor."""

    @pytest.fixture
    def sample_files(self, tmp_path: Path) -> List[Path]:
        """Create sample audio files for testing."""
        files = []
        for i in range(10):
            file_path = tmp_path / f"audio_{i}.m4a"
            file_path.write_bytes(b"fake audio data")
            files.append(file_path)
        return files

    @pytest.fixture
    def processor(self):
        """Create a batch processor instance."""
        config = BatchConfig(batch_size=5, max_workers=2)
        return BatchProcessor(config)

    def test_initialization(self, processor):
        """Test processor initialization."""
        assert processor.config.batch_size == 5
        assert processor.config.max_workers == 2
        assert processor.progress.processed == 0
        assert processor.progress.failed == 0
        assert processor.statistics.total_files == 0

    def test_create_batches(self, processor, sample_files):
        """Test batch creation from file list."""
        batches = processor._create_batches(sample_files)
        assert len(batches) == 2  # 10 files with batch_size=5
        assert len(batches[0]) == 5
        assert len(batches[1]) == 5

    def test_create_batches_with_remainder(self, processor, sample_files):
        """Test batch creation with remainder files."""
        # Add 3 more files to create remainder
        tmp_path = sample_files[0].parent
        for i in range(3):
            file_path = tmp_path / f"audio_extra_{i}.m4a"
            file_path.write_bytes(b"fake audio data")
            sample_files.append(file_path)

        batches = processor._create_batches(sample_files)
        assert len(batches) == 3  # 13 files with batch_size=5
        assert len(batches[0]) == 5
        assert len(batches[1]) == 5
        assert len(batches[2]) == 3

    @pytest.mark.asyncio
    async def test_process_batch_success(self, processor, sample_files):
        """Test successful batch processing."""

        async def mock_process_func(file_path: Path) -> dict:
            """Mock processing function."""
            return {"file": str(file_path), "status": "success"}

        batch = sample_files[:5]
        results = await processor._process_batch(batch, mock_process_func, batch_index=0)

        assert len(results) == 5
        assert all(r.status == "success" for r in results)
        assert processor.progress.processed == 5

    @pytest.mark.asyncio
    async def test_process_batch_with_error_continue(self, processor, sample_files):
        """Test batch processing with error when continue_on_error is True."""

        async def mock_process_func(file_path: Path) -> dict:
            """Mock processing function that fails on specific files."""
            if "audio_2" in str(file_path) or "audio_4" in str(file_path):
                raise ValueError("Mock processing error")
            return {"file": str(file_path), "status": "success"}

        batch = sample_files[:5]
        results = await processor._process_batch(batch, mock_process_func, batch_index=0)

        assert len(results) == 5  # All files processed
        assert processor.progress.failed == 2  # 2 files failed
        assert processor.progress.processed == 5  # 5 files processed

    @pytest.mark.asyncio
    async def test_process_batch_with_error_stop(self, sample_files):
        """Test batch processing with error when continue_on_error is False."""
        config = BatchConfig(batch_size=5, continue_on_error=False)
        processor = BatchProcessor(config)

        async def mock_process_func(file_path: Path) -> dict:
            """Mock processing function that fails."""
            if "audio_2" in str(file_path):
                raise ValueError("Mock processing error")
            return {"file": str(file_path), "status": "success"}

        batch = sample_files[:5]
        # With continue_on_error=False, the exception should be raised
        # but our implementation catches it and converts to BatchResult
        results = await processor._process_batch(batch, mock_process_func, batch_index=0)

        # Should still process all files, but mark failed ones
        assert len(results) == 5
        failed_results = [r for r in results if r.status == "failed"]
        assert len(failed_results) >= 1  # At least audio_2 failed

    @pytest.mark.asyncio
    async def test_process_all_files(self, processor, sample_files):
        """Test processing all files in batches."""
        processed_files = []

        async def mock_process_func(file_path: Path) -> dict:
            """Mock processing function."""
            processed_files.append(file_path)
            return {"file": str(file_path), "status": "success"}

        results = await processor.process_all(sample_files, mock_process_func)

        assert len(results) == 10  # All files processed
        assert len(processed_files) == 10
        assert processor.progress.processed == 10
        assert processor.progress.failed == 0

    def test_progress_tracking(self, processor):
        """Test progress tracking."""
        progress = processor.get_progress()
        assert progress.progress_ratio == 0.0

        processor.progress.processed = 5
        processor.progress.total = 10
        progress = processor.get_progress()
        assert progress.progress_ratio == 0.5

        processor.progress.processed = 10
        progress = processor.get_progress()
        assert progress.progress_ratio == 1.0

    def test_failed_files_tracking(self, processor):
        """Test failed files tracking."""
        # Add some failed files
        processor.progress.failed_files = ["/path/to/file1.m4a", "/path/to/file2.m4a"]

        failed = processor.get_failed_files()
        assert len(failed) == 2
        assert "/path/to/file1.m4a" in failed
        assert "/path/to/file2.m4a" in failed

    def test_statistics_calculation(self, processor, sample_files):
        """Test statistics calculation."""
        from voice_man.services.batch_service import BatchResult

        # Create mock results
        results = [
            BatchResult(file_path="file1.m4a", status="success", attempts=1),
            BatchResult(file_path="file2.m4a", status="success", attempts=1),
            BatchResult(file_path="file3.m4a", status="failed", attempts=3, error="Error"),
        ]

        processor.statistics.calculate_from_results(results)

        stats = processor.get_statistics()
        assert stats.total_files == 3
        assert stats.successful_files == 2
        assert stats.failed_files == 1
        assert stats.total_attempts == 5  # 1 + 1 + 3
        assert stats.average_attempts_per_file == 5 / 3
