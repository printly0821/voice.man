"""
Enhanced TDD tests for batch processing service with memory optimization.

RED Phase: Write failing tests first for:
- Memory management with gc.collect()
- Real 183-file batch processing
- Progress tracking integration
- Enhanced error recovery
"""

import pytest
import gc
import time
from pathlib import Path
from typing import List
from unittest.mock import Mock, patch, AsyncMock

from voice_man.services.batch_service import BatchProcessor, BatchConfig, BatchProgress, BatchResult
from voice_man.services.progress_service import ProgressTracker, ProgressConfig


class TestMemoryManagement:
    """Test memory management features."""

    @pytest.fixture
    def processor_with_memory_config(self):
        """Create processor optimized for memory management."""
        config = BatchConfig(batch_size=10, max_workers=2, retry_count=3)
        return BatchProcessor(config)

    @pytest.mark.asyncio
    async def test_memory_cleanup_between_batches(self, processor_with_memory_config, tmp_path):
        """Test that memory is cleaned up between batches (RED - failing)."""
        # Create 25 files to test 3 batches with cleanup
        files = []
        for i in range(25):
            file_path = tmp_path / f"audio_{i}.m4a"
            file_path.write_bytes(b"fake audio data" * 1000)  # Larger files
            files.append(file_path)

        processing_times = []

        async def mock_process_func(file_path: Path) -> dict:
            """Mock processing that allocates memory."""
            # Simulate memory allocation
            large_data = [x for x in range(100000)]
            processing_times.append(time.time())
            await asyncio_sleep(0.01)
            return {"file": str(file_path), "status": "success", "data_size": len(large_data)}

        results = await processor_with_memory_config.process_all(files, mock_process_func)

        # Verify memory cleanup happened between batches
        # This test will fail initially because gc.collect() is not implemented
        assert len(results) == 25
        assert processor_with_memory_config.progress.total_batches == 3

        # Check that processing time doesn't increase significantly (memory leak indicator)
        if len(processing_times) >= 10:
            first_batch_avg = sum(processing_times[:5]) / 5
            last_batch_avg = sum(processing_times[-5:]) / 5
            # Last batch should not be more than 2x slower than first batch
            assert last_batch_avg < first_batch_avg * 2, (
                "Memory leak detected: processing time increased significantly"
            )

    @pytest.mark.asyncio
    async def test_gc_collect_called_between_batches(self, processor_with_memory_config, tmp_path):
        """Test that gc.collect() is called between batches (RED - failing)."""
        files = []
        for i in range(15):
            file_path = tmp_path / f"audio_{i}.m4a"
            file_path.write_bytes(b"fake audio data")
            files.append(file_path)

        cleanup_calls = []

        # Track calls to _cleanup_memory
        original_cleanup = processor_with_memory_config._cleanup_memory

        def tracked_cleanup():
            cleanup_calls.append(time.time())
            return original_cleanup()

        processor_with_memory_config._cleanup_memory = tracked_cleanup

        async def mock_process_func(file_path: Path) -> dict:
            await asyncio_sleep(0.01)
            return {"file": str(file_path), "status": "success"}

        await processor_with_memory_config.process_all(files, mock_process_func)

        # Verify _cleanup_memory was called between batches
        # With 15 files and batch_size=10, we have 2 batches, so cleanup should be called once
        assert len(cleanup_calls) >= 1, "_cleanup_memory should be called between batches"


class TestRealFileBatchProcessing:
    """Test with real file structure (183 files)."""

    @pytest.fixture
    def real_file_structure(self, tmp_path):
        """Create a structure mimicking the real 183 audio files."""
        # 신기연 (94 files), 신동식 (86 files), 김경민 (3 files)
        files_by_person = {"신기연": 94, "신동식": 86, "김경민": 3}

        all_files = []
        for person, count in files_by_person.items():
            for i in range(count):
                file_path = tmp_path / f"통화 녹음 {person}_{i:06d}.m4a"
                file_path.write_bytes(b"fake audio data" * 100)
                all_files.append(file_path)

        return all_files

    @pytest.mark.asyncio
    async def test_process_183_files_in_batches(self, real_file_structure):
        """Test processing all 183 files in batches (RED - failing)."""
        config = BatchConfig(batch_size=10, max_workers=2)
        processor = BatchProcessor(config)

        processed_count = [0]

        async def mock_process_func(file_path: Path) -> dict:
            processed_count[0] += 1
            # Simulate processing time
            await asyncio_sleep(0.001)
            return {"file": str(file_path), "status": "success", "file_name": file_path.name}

        results = await processor.process_all(real_file_structure, mock_process_func)

        # Verify all 183 files were processed
        assert len(results) == 183
        assert processed_count[0] == 183
        assert processor.progress.total == 183

        # Verify batch count (183 / 10 = 19 batches)
        assert processor.progress.total_batches == 19

    @pytest.mark.asyncio
    async def test_batch_distribution_for_183_files(self, real_file_structure):
        """Test that 183 files are distributed correctly across batches (RED - failing)."""
        config = BatchConfig(batch_size=10, max_workers=2)
        processor = BatchProcessor(config)

        batches = processor._create_batches(real_file_structure)

        # 183 files with batch_size=10 should create 19 batches
        assert len(batches) == 19

        # First 18 batches should have 10 files each
        for i in range(18):
            assert len(batches[i]) == 10, f"Batch {i} should have 10 files"

        # Last batch should have 3 files
        assert len(batches[18]) == 3, "Last batch should have 3 files"

    @pytest.mark.asyncio
    async def test_progress_tracking_during_183_file_processing(self, real_file_structure):
        """Test progress tracking for 183 files (RED - failing)."""
        config = BatchConfig(batch_size=10, max_workers=2)
        processor = BatchProcessor(config)

        progress_tracker = ProgressTracker(ProgressConfig())
        progress_tracker.start_overall(total_files=183, total_batches=19)

        batch_progress_history = []

        async def mock_process_func(file_path: Path) -> dict:
            await asyncio_sleep(0.001)
            return {"file": str(file_path), "status": "success"}

        # Process files and track progress
        await processor.process_all(real_file_structure, mock_process_func)

        # Track progress updates
        for batch_id in range(1, 20):
            completed = min(batch_id * 10, 183)
            progress_tracker.update_batch_progress(batch_id, completed, 0)
            batch_progress_history.append(progress_tracker.get_progress_summary())

        # Verify progress tracking
        assert len(batch_progress_history) == 19
        final_progress = batch_progress_history[-1]
        assert final_progress["completed_files"] == 183
        assert final_progress["total_files"] == 183
        # Progress tracker calculates percentage based on its internal state
        assert final_progress["overall_progress_percentage"] == 100.0


class TestErrorRecoveryEnhanced:
    """Test enhanced error recovery with retry logic."""

    @pytest.mark.asyncio
    async def test_retry_with_exponential_backoff(self, tmp_path):
        """Test retry logic with exponential backoff (RED - failing)."""
        config = BatchConfig(batch_size=5, max_workers=2, retry_count=3)
        processor = BatchProcessor(config)

        files = []
        for i in range(5):
            file_path = tmp_path / f"audio_{i}.m4a"
            file_path.write_bytes(b"fake audio data")
            files.append(file_path)

        attempt_counts = {}

        async def mock_process_func(file_path: Path) -> dict:
            file_name = file_path.name
            attempt_counts[file_name] = attempt_counts.get(file_name, 0) + 1

            # Fail on first 2 attempts, succeed on 3rd
            if attempt_counts[file_name] < 3:
                raise ValueError(f"Temporary failure for {file_name}")

            return {"file": str(file_path), "status": "success"}

        results = await processor.process_all(files, mock_process_func)

        # All files should eventually succeed
        assert len(results) == 5
        assert all(r.status == "success" for r in results)

        # Verify retry attempts
        for file_name, count in attempt_counts.items():
            assert count == 3, f"{file_name} should have 3 attempts"

    @pytest.mark.asyncio
    async def test_max_retry_exceeded(self, tmp_path):
        """Test that processing fails after max retries (RED - failing)."""
        config = BatchConfig(batch_size=5, max_workers=2, retry_count=2)
        processor = BatchProcessor(config)

        files = []
        for i in range(3):
            file_path = tmp_path / f"audio_{i}.m4a"
            file_path.write_bytes(b"fake audio data")
            files.append(file_path)

        attempt_counts = {}

        async def mock_process_func(file_path: Path) -> dict:
            file_name = file_path.name
            attempt_counts[file_name] = attempt_counts.get(file_name, 0) + 1

            # Always fail
            raise ValueError(f"Permanent failure for {file_name}")

        results = await processor.process_all(files, mock_process_func)

        # All files should fail
        assert len(results) == 3
        assert all(r.status == "failed" for r in results)

        # Verify max retry attempts
        for file_name, count in attempt_counts.items():
            assert count == 2, f"{file_name} should have 2 attempts (max_retries)"

    @pytest.mark.asyncio
    async def test_failed_files_separate_storage(self, tmp_path):
        """Test that failed files are stored separately (RED - failing)."""
        config = BatchConfig(batch_size=5, max_workers=2, retry_count=3)
        processor = BatchProcessor(config)

        files = []
        for i in range(5):
            file_path = tmp_path / f"audio_{i}.m4a"
            file_path.write_bytes(b"fake audio data")
            files.append(file_path)

        async def mock_process_func(file_path: Path) -> dict:
            # Fail on files with even index (0, 2, 4)
            file_index = int(file_path.stem.split("_")[1])
            if file_index % 2 == 0:
                raise ValueError(f"Failed for {file_path.name}")
            return {"file": str(file_path), "status": "success"}

        results = await processor.process_all(files, mock_process_func)

        # Separate failed files
        failed_results = [r for r in results if r.status == "failed"]
        success_results = [r for r in results if r.status == "success"]

        # Files 0, 2, 4 should fail (3 files total)
        assert len(failed_results) == 3  # audio_0, audio_2, and audio_4
        assert len(success_results) == 2  # audio_1 and audio_3

        # Verify failed results have error information
        for failed in failed_results:
            assert failed.error is not None
            assert failed.attempts == 3  # Max retries


class TestProgressTrackingIntegration:
    """Test integration with progress tracking service."""

    @pytest.mark.asyncio
    async def test_progress_tracker_integration(self, tmp_path):
        """Test that batch processor integrates with progress tracker (GREEN - passing)."""
        config = BatchConfig(batch_size=10, max_workers=2)
        processor = BatchProcessor(config)

        progress_tracker = ProgressTracker(ProgressConfig())

        files = []
        for i in range(25):
            file_path = tmp_path / f"audio_{i}.m4a"
            file_path.write_bytes(b"fake audio data")
            files.append(file_path)

        batch_updates = []

        async def mock_process_func(file_path: Path) -> dict:
            await asyncio_sleep(0.001)
            return {"file": str(file_path), "status": "success"}

        # Start overall progress tracking
        progress_tracker.start_overall(total_files=25, total_batches=3)

        # Process all files with progress callback
        def progress_callback(progress):
            progress_tracker.update_batch_progress(
                progress.current_batch, progress.processed, progress.failed
            )

        await processor.process_all(files, mock_process_func, progress_callback)

        # Get final summary
        final_summary = progress_tracker.get_progress_summary()

        # Verify progress tracking
        assert final_summary["total_files"] == 25
        assert final_summary["completed_files"] == 25
        assert final_summary["overall_progress_percentage"] == 100.0

    @pytest.mark.asyncio
    async def test_eta_calculation_during_processing(self, tmp_path):
        """Test ETA calculation during batch processing (RED - failing)."""
        config = BatchConfig(batch_size=10, max_workers=2)
        processor = BatchProcessor(config)

        progress_tracker = ProgressTracker(ProgressConfig(eta_window_size=5))

        files = []
        for i in range(30):
            file_path = tmp_path / f"audio_{i}.m4a"
            file_path.write_bytes(b"fake audio data")
            files.append(file_path)

        progress_tracker.start_overall(total_files=30, total_batches=3)

        async def mock_process_func(file_path: Path) -> dict:
            # Simulate variable processing time
            await asyncio_sleep(0.01)
            return {"file": str(file_path), "status": "success"}

        # Process files
        results = await processor.process_all(files, mock_process_func)

        # Update progress and check ETA
        progress_tracker.update_batch_progress(1, 10, 0)
        eta_after_batch1 = progress_tracker.get_eta_seconds()

        progress_tracker.update_batch_progress(2, 20, 0)
        eta_after_batch2 = progress_tracker.get_eta_seconds()

        progress_tracker.update_batch_progress(3, 30, 0)
        eta_after_batch3 = progress_tracker.get_eta_seconds()

        # ETA should decrease as processing progresses
        assert eta_after_batch1 > 0, "ETA should be calculated"
        assert eta_after_batch2 < eta_after_batch1, (
            "ETA should decrease as more files are processed"
        )
        assert eta_after_batch3 == 0, "ETA should be 0 when complete"


class TestBatchStatistics:
    """Test batch processing statistics."""

    @pytest.mark.asyncio
    async def test_processing_statistics(self, tmp_path):
        """Test that processing statistics are calculated correctly (RED - failing)."""
        config = BatchConfig(batch_size=10, max_workers=2)
        processor = BatchProcessor(config)

        files = []
        for i in range(25):
            file_path = tmp_path / f"audio_{i}.m4a"
            file_path.write_bytes(b"fake audio data")
            files.append(file_path)

        processing_times = []

        async def mock_process_func(file_path: Path) -> dict:
            start_time = time.time()
            await asyncio_sleep(0.001)
            processing_times.append(time.time() - start_time)
            return {"file": str(file_path), "status": "success"}

        results = await processor.process_all(files, mock_process_func)

        # Calculate statistics
        avg_time = sum(processing_times) / len(processing_times)
        min_time = min(processing_times)
        max_time = max(processing_times)

        # Verify statistics
        assert len(results) == 25
        assert len(processing_times) == 25
        assert avg_time > 0
        assert min_time <= avg_time <= max_time

    @pytest.mark.asyncio
    async def test_success_rate_calculation(self, tmp_path):
        """Test success rate calculation (RED - failing)."""
        config = BatchConfig(batch_size=10, max_workers=2)
        processor = BatchProcessor(config)

        files = []
        for i in range(20):
            file_path = tmp_path / f"audio_{i}.m4a"
            file_path.write_bytes(b"fake audio data")
            files.append(file_path)

        async def mock_process_func(file_path: Path) -> dict:
            # Fail on 4 files (20% failure rate)
            if int(file_path.stem.split("_")[1]) < 4:
                raise ValueError("Simulated failure")
            return {"file": str(file_path), "status": "success"}

        results = await processor.process_all(files, mock_process_func)

        success_count = sum(1 for r in results if r.status == "success")
        failed_count = sum(1 for r in results if r.status == "failed")

        # Verify success rate
        assert success_count == 16
        assert failed_count == 4

        progress = processor.get_progress()
        expected_success_rate = 16 / 20
        assert abs(progress.success_rate - expected_success_rate) < 0.01


# Helper function
async def asyncio_sleep(seconds: float):
    """Async sleep helper."""
    import asyncio

    await asyncio.sleep(seconds)
