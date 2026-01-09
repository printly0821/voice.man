"""
RED Phase Tests for E2ETestRunner.

Tests for E2ETestRunner class functionality.
TASK-004: E2ETestRunner class implementation.
TASK-005: Progress callback system.
TASK-006: Dynamic batch size adjustment.
TASK-007: Retry logic with exponential backoff.
"""

import asyncio
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch


class TestE2ETestRunnerCollectFiles:
    """Tests for E2ETestRunner.collect_files method."""

    @pytest.mark.asyncio
    async def test_collect_files_from_directory(self, tmp_path: Path):
        """Test collecting audio files from directory."""
        from voice_man.services.e2e_test_service import E2ETestRunner, E2ETestConfig

        # Create test files
        (tmp_path / "audio1.m4a").touch()
        (tmp_path / "audio2.mp3").touch()
        (tmp_path / "audio3.wav").touch()
        (tmp_path / "document.pdf").touch()  # Should be ignored

        config = E2ETestConfig()
        runner = E2ETestRunner(config)

        files = await runner.collect_files(tmp_path)

        assert len(files) == 3
        assert all(f.suffix in {".m4a", ".mp3", ".wav"} for f in files)

    @pytest.mark.asyncio
    async def test_collect_files_empty_directory(self, tmp_path: Path):
        """Test collecting from empty directory returns empty list."""
        from voice_man.services.e2e_test_service import E2ETestRunner, E2ETestConfig

        config = E2ETestConfig()
        runner = E2ETestRunner(config)

        files = await runner.collect_files(tmp_path)

        assert files == []

    @pytest.mark.asyncio
    async def test_collect_files_nonexistent_directory(self):
        """Test collecting from nonexistent directory raises error."""
        from voice_man.services.e2e_test_service import E2ETestRunner, E2ETestConfig

        config = E2ETestConfig()
        runner = E2ETestRunner(config)

        with pytest.raises(FileNotFoundError):
            await runner.collect_files(Path("/nonexistent/directory"))

    @pytest.mark.asyncio
    async def test_collect_files_sorted(self, tmp_path: Path):
        """Test collected files are sorted by name."""
        from voice_man.services.e2e_test_service import E2ETestRunner, E2ETestConfig

        # Create files in non-alphabetical order
        (tmp_path / "zebra.m4a").touch()
        (tmp_path / "alpha.m4a").touch()
        (tmp_path / "middle.m4a").touch()

        config = E2ETestConfig()
        runner = E2ETestRunner(config)

        files = await runner.collect_files(tmp_path)

        assert files[0].name == "alpha.m4a"
        assert files[1].name == "middle.m4a"
        assert files[2].name == "zebra.m4a"


class TestE2ETestRunnerProcessSingleFile:
    """Tests for E2ETestRunner.process_single_file method."""

    @pytest.mark.asyncio
    async def test_process_single_file_success(self, tmp_path: Path, mock_whisperx_service):
        """Test successful single file processing."""
        from voice_man.services.e2e_test_service import E2ETestRunner, E2ETestConfig

        test_file = tmp_path / "test.m4a"
        test_file.touch()

        config = E2ETestConfig()
        runner = E2ETestRunner(config, whisperx_service=mock_whisperx_service)

        result = await runner.process_single_file(test_file)

        assert result.status == "success"
        assert result.file_path == str(test_file)
        assert result.processing_time_seconds > 0
        assert result.error is None

    @pytest.mark.asyncio
    async def test_process_single_file_failure(self, tmp_path: Path):
        """Test single file processing failure."""
        from voice_man.services.e2e_test_service import E2ETestRunner, E2ETestConfig

        test_file = tmp_path / "test.m4a"
        test_file.touch()

        # Create mock that raises exception
        mock_service = MagicMock()
        mock_service.process_audio = AsyncMock(side_effect=RuntimeError("Processing failed"))

        config = E2ETestConfig()
        runner = E2ETestRunner(config, whisperx_service=mock_service)

        result = await runner.process_single_file(test_file)

        assert result.status == "failed"
        assert "Processing failed" in result.error

    @pytest.mark.asyncio
    async def test_process_single_file_no_service(self, tmp_path: Path):
        """Test processing without WhisperX service fails gracefully."""
        from voice_man.services.e2e_test_service import E2ETestRunner, E2ETestConfig

        test_file = tmp_path / "test.m4a"
        test_file.touch()

        config = E2ETestConfig()
        runner = E2ETestRunner(config, whisperx_service=None)

        result = await runner.process_single_file(test_file)

        assert result.status == "failed"
        assert "not initialized" in result.error.lower()


class TestE2ETestRunnerRetryLogic:
    """Tests for retry logic with exponential backoff (TASK-007)."""

    @pytest.mark.asyncio
    async def test_retry_on_failure(self, tmp_path: Path):
        """Test retry logic attempts multiple times on failure."""
        from voice_man.services.e2e_test_service import E2ETestRunner, E2ETestConfig

        test_file = tmp_path / "test.m4a"
        test_file.touch()

        # Mock service that fails twice then succeeds
        call_count = 0

        async def mock_process(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError(f"Attempt {call_count} failed")
            return MagicMock(text="Success", segments=[], speakers=[])

        mock_service = MagicMock()
        mock_service.process_audio = mock_process

        config = E2ETestConfig(max_retries=3, retry_delays=[0, 0, 0])  # No delays for test
        runner = E2ETestRunner(config, whisperx_service=mock_service)

        result = await runner._process_with_retry(test_file)

        assert result.status == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_exhausted(self, tmp_path: Path):
        """Test retry logic returns failure after all attempts exhausted."""
        from voice_man.services.e2e_test_service import E2ETestRunner, E2ETestConfig

        test_file = tmp_path / "test.m4a"
        test_file.touch()

        # Mock service that always fails
        mock_service = MagicMock()
        mock_service.process_audio = AsyncMock(side_effect=RuntimeError("Always fails"))

        config = E2ETestConfig(max_retries=3, retry_delays=[0, 0, 0])
        runner = E2ETestRunner(config, whisperx_service=mock_service)

        result = await runner._process_with_retry(test_file)

        assert result.status == "failed"
        assert mock_service.process_audio.call_count == 3

    @pytest.mark.asyncio
    async def test_retry_delays_exponential(self, tmp_path: Path):
        """Test retry uses configured exponential backoff delays."""
        from voice_man.services.e2e_test_service import E2ETestRunner, E2ETestConfig

        test_file = tmp_path / "test.m4a"
        test_file.touch()

        mock_service = MagicMock()
        mock_service.process_audio = AsyncMock(side_effect=RuntimeError("Fails"))

        # Test with very short delays
        config = E2ETestConfig(max_retries=3, retry_delays=[0.01, 0.02, 0.03])
        runner = E2ETestRunner(config, whisperx_service=mock_service)

        import time

        start = time.time()
        await runner._process_with_retry(test_file)
        elapsed = time.time() - start

        # Should have waited at least some time for delays
        assert elapsed >= 0.03  # At least first two delays


class TestE2ETestRunnerDynamicBatchAdjustment:
    """Tests for dynamic batch size adjustment (TASK-006)."""

    @pytest.mark.asyncio
    async def test_reduce_batch_size_on_critical_memory(self, mock_gpu_monitor):
        """Test batch size reduction when GPU memory is critical (>95%)."""
        from voice_man.services.e2e_test_service import E2ETestRunner, E2ETestConfig

        # Configure mock to return critical memory status
        mock_gpu_monitor.check_memory_status.return_value = {
            "warning": True,
            "critical": True,
            "auto_adjust_recommended": True,
            "usage_percentage": 96.0,
            "message": "Critical memory",
        }

        config = E2ETestConfig(batch_size=16, min_batch_size=2)
        runner = E2ETestRunner(config, gpu_monitor=mock_gpu_monitor)
        runner._current_batch_size = 16

        await runner._adjust_batch_size()

        # Should reduce by 50%
        assert runner._current_batch_size == 8
        mock_gpu_monitor.clear_gpu_cache.assert_called_once()

    @pytest.mark.asyncio
    async def test_increase_batch_size_on_low_memory(self, mock_gpu_monitor):
        """Test batch size increase when GPU memory is low (<50%)."""
        from voice_man.services.e2e_test_service import E2ETestRunner, E2ETestConfig

        # Configure mock to return low memory status
        mock_gpu_monitor.check_memory_status.return_value = {
            "warning": False,
            "critical": False,
            "auto_adjust_recommended": False,
            "usage_percentage": 30.0,
            "message": "Low memory usage",
        }

        config = E2ETestConfig(batch_size=15, max_batch_size=32)
        runner = E2ETestRunner(config, gpu_monitor=mock_gpu_monitor)
        runner._current_batch_size = 15

        await runner._adjust_batch_size()

        # Should increase by 2
        assert runner._current_batch_size == 17

    @pytest.mark.asyncio
    async def test_batch_size_respects_minimum(self, mock_gpu_monitor):
        """Test batch size doesn't go below minimum."""
        from voice_man.services.e2e_test_service import E2ETestRunner, E2ETestConfig

        mock_gpu_monitor.check_memory_status.return_value = {
            "warning": True,
            "critical": True,
            "auto_adjust_recommended": True,
            "usage_percentage": 96.0,
            "message": "Critical",
        }

        config = E2ETestConfig(batch_size=4, min_batch_size=2)
        runner = E2ETestRunner(config, gpu_monitor=mock_gpu_monitor)
        runner._current_batch_size = 3

        await runner._adjust_batch_size()

        # Should not go below min_batch_size
        assert runner._current_batch_size >= 2

    @pytest.mark.asyncio
    async def test_batch_size_respects_maximum(self, mock_gpu_monitor):
        """Test batch size doesn't exceed maximum."""
        from voice_man.services.e2e_test_service import E2ETestRunner, E2ETestConfig

        mock_gpu_monitor.check_memory_status.return_value = {
            "warning": False,
            "critical": False,
            "auto_adjust_recommended": False,
            "usage_percentage": 20.0,
            "message": "Low",
        }

        config = E2ETestConfig(batch_size=30, max_batch_size=32)
        runner = E2ETestRunner(config, gpu_monitor=mock_gpu_monitor)
        runner._current_batch_size = 31

        await runner._adjust_batch_size()

        # Should not exceed max_batch_size
        assert runner._current_batch_size <= 32

    @pytest.mark.asyncio
    async def test_no_adjustment_without_gpu_monitor(self):
        """Test no adjustment happens without GPU monitor."""
        from voice_man.services.e2e_test_service import E2ETestRunner, E2ETestConfig

        config = E2ETestConfig(batch_size=15, dynamic_batch_adjustment=True)
        runner = E2ETestRunner(config, gpu_monitor=None)
        runner._current_batch_size = 15

        await runner._adjust_batch_size()

        assert runner._current_batch_size == 15

    @pytest.mark.asyncio
    async def test_no_adjustment_when_disabled(self, mock_gpu_monitor):
        """Test no adjustment when dynamic adjustment is disabled."""
        from voice_man.services.e2e_test_service import E2ETestRunner, E2ETestConfig

        mock_gpu_monitor.check_memory_status.return_value = {
            "critical": True,
            "usage_percentage": 96.0,
        }

        config = E2ETestConfig(batch_size=15, dynamic_batch_adjustment=False)
        runner = E2ETestRunner(config, gpu_monitor=mock_gpu_monitor)
        runner._current_batch_size = 15

        await runner._adjust_batch_size()

        assert runner._current_batch_size == 15


class TestE2ETestRunnerProgressCallback:
    """Tests for progress callback system (TASK-005)."""

    @pytest.mark.asyncio
    async def test_progress_callback_called(self, tmp_path: Path, mock_whisperx_service):
        """Test progress callback is called for each file."""
        from voice_man.services.e2e_test_service import E2ETestRunner, E2ETestConfig

        # Create test files
        for i in range(3):
            (tmp_path / f"test{i}.m4a").touch()

        config = E2ETestConfig(enable_checksum_verification=False)
        runner = E2ETestRunner(config, whisperx_service=mock_whisperx_service)

        callback_calls = []

        def progress_callback(current, total, elapsed, filename, status):
            callback_calls.append(
                {
                    "current": current,
                    "total": total,
                    "elapsed": elapsed,
                    "filename": filename,
                    "status": status,
                }
            )

        files = await runner.collect_files(tmp_path)
        await runner.run(files, progress_callback=progress_callback)

        assert len(callback_calls) == 3
        assert callback_calls[0]["current"] == 1
        assert callback_calls[0]["total"] == 3
        assert callback_calls[2]["current"] == 3

    @pytest.mark.asyncio
    async def test_progress_callback_includes_elapsed_time(
        self, tmp_path: Path, mock_whisperx_service
    ):
        """Test progress callback includes elapsed time."""
        from voice_man.services.e2e_test_service import E2ETestRunner, E2ETestConfig

        (tmp_path / "test.m4a").touch()

        config = E2ETestConfig(enable_checksum_verification=False)
        runner = E2ETestRunner(config, whisperx_service=mock_whisperx_service)

        elapsed_times = []

        def progress_callback(current, total, elapsed, filename, status):
            elapsed_times.append(elapsed)

        files = await runner.collect_files(tmp_path)
        await runner.run(files, progress_callback=progress_callback)

        assert len(elapsed_times) == 1
        assert elapsed_times[0] >= 0


class TestE2ETestRunnerRun:
    """Tests for E2ETestRunner.run method."""

    @pytest.mark.asyncio
    async def test_run_empty_file_list(self):
        """Test run with empty file list returns empty result."""
        from voice_man.services.e2e_test_service import E2ETestRunner, E2ETestConfig

        config = E2ETestConfig()
        runner = E2ETestRunner(config)

        result = await runner.run([])

        assert result.total_files == 0
        assert result.processed_files == 0
        assert result.failed_files == 0
        assert result.checksum_verified is True

    @pytest.mark.asyncio
    async def test_run_verifies_checksums(self, tmp_path: Path, mock_whisperx_service):
        """Test run verifies file checksums before and after."""
        from voice_man.services.e2e_test_service import E2ETestRunner, E2ETestConfig

        test_file = tmp_path / "test.m4a"
        test_file.write_bytes(b"test content")

        config = E2ETestConfig(enable_checksum_verification=True)
        runner = E2ETestRunner(config, whisperx_service=mock_whisperx_service)

        result = await runner.run([test_file])

        assert result.checksum_verified is True

    @pytest.mark.asyncio
    async def test_run_collects_gpu_stats(
        self, tmp_path: Path, mock_whisperx_service, mock_gpu_monitor
    ):
        """Test run collects GPU statistics."""
        from voice_man.services.e2e_test_service import E2ETestRunner, E2ETestConfig

        test_file = tmp_path / "test.m4a"
        test_file.touch()

        config = E2ETestConfig(enable_checksum_verification=False)
        runner = E2ETestRunner(
            config,
            whisperx_service=mock_whisperx_service,
            gpu_monitor=mock_gpu_monitor,
        )

        result = await runner.run([test_file])

        assert "total_mb" in result.gpu_stats
        assert "usage_percentage" in result.gpu_stats

    @pytest.mark.asyncio
    async def test_run_calculates_timing(self, tmp_path: Path, mock_whisperx_service):
        """Test run calculates total and average timing."""
        from voice_man.services.e2e_test_service import E2ETestRunner, E2ETestConfig

        for i in range(2):
            (tmp_path / f"test{i}.m4a").touch()

        config = E2ETestConfig(enable_checksum_verification=False)
        runner = E2ETestRunner(config, whisperx_service=mock_whisperx_service)

        files = await runner.collect_files(tmp_path)
        result = await runner.run(files)

        assert result.total_time_seconds > 0
        assert result.avg_time_per_file == result.total_time_seconds / 2
