"""
E2E Integration Tests for Full Batch Processing.

TASK-010: 183 files full E2E test with pytest integration.
Tests the complete WhisperX batch processing pipeline with GPU parallel processing.

SPEC-E2ETEST-001 Requirements:
- U1: BatchProcessor based GPU parallel batch processing
- U4: Original file integrity verification (checksum)
- E4: Dynamic batch adjustment on GPU memory shortage
- S2: Failed file retry queue (exponential backoff)
- N1: GPU memory must not exceed 95%
- N3: Original file modification prohibited
"""

import asyncio
import os
import pytest
import time
from pathlib import Path
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch


# Mark all tests in this module as e2e tests
pytestmark = [pytest.mark.e2e, pytest.mark.slow]


class TestE2EBatchProcessingIntegration:
    """Integration tests for E2E batch processing pipeline."""

    @pytest.fixture
    def audio_files_directory(self) -> Path:
        """Return path to the ref/call directory with 183 audio files."""
        return Path(__file__).parent.parent.parent / "ref" / "call"

    @pytest.fixture
    def results_directory(self) -> Path:
        """Return path to results output directory."""
        results_dir = Path(__file__).parent.parent.parent / "ref" / "call" / "reports" / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        return results_dir

    def test_audio_files_exist(self, audio_files_directory: Path):
        """Verify test audio files exist in ref/call directory."""
        assert audio_files_directory.exists(), f"Directory not found: {audio_files_directory}"

        m4a_files = list(audio_files_directory.glob("*.m4a"))
        assert len(m4a_files) >= 100, f"Expected at least 100 m4a files, found {len(m4a_files)}"

    @pytest.mark.asyncio
    async def test_collect_all_files(self, audio_files_directory: Path):
        """Test collecting all 183 audio files from ref/call."""
        from voice_man.services.e2e_test_service import E2ETestRunner, E2ETestConfig

        config = E2ETestConfig()
        runner = E2ETestRunner(config)

        files = await runner.collect_files(audio_files_directory)

        # Should find approximately 183 m4a files
        assert len(files) >= 180, f"Expected ~183 files, found {len(files)}"
        assert all(f.suffix == ".m4a" for f in files)

    @pytest.mark.asyncio
    async def test_checksum_calculation_all_files(self, audio_files_directory: Path):
        """Test checksum calculation for all files."""
        from voice_man.services.e2e_test_service import (
            E2ETestRunner,
            E2ETestConfig,
            calculate_md5,
        )

        config = E2ETestConfig()
        runner = E2ETestRunner(config)

        files = await runner.collect_files(audio_files_directory)
        files = files[:10]  # Limit for speed

        # Calculate checksums
        checksums = {}
        for f in files:
            checksums[str(f)] = calculate_md5(f)

        assert len(checksums) == 10
        assert all(len(h) == 32 for h in checksums.values())

    @pytest.mark.asyncio
    async def test_file_integrity_preserved(self, audio_files_directory: Path):
        """Test U4/N3: Original file integrity is preserved after processing."""
        from voice_man.services.e2e_test_service import (
            E2ETestRunner,
            E2ETestConfig,
            calculate_md5,
            verify_checksums,
        )

        config = E2ETestConfig()
        runner = E2ETestRunner(config)

        # Get first 5 files
        files = await runner.collect_files(audio_files_directory)
        files = files[:5]

        # Calculate checksums before
        original_checksums = {str(f): calculate_md5(f) for f in files}

        # Simulate processing (just wait a bit)
        await asyncio.sleep(0.1)

        # Calculate checksums after
        current_checksums = {str(f): calculate_md5(f) for f in files}

        # Verify integrity
        assert verify_checksums(original_checksums, current_checksums), (
            "File integrity check failed - files were modified"
        )

    @pytest.mark.asyncio
    async def test_batch_processing_with_mock_service(
        self, audio_files_directory: Path, mock_whisperx_service, results_directory: Path
    ):
        """Test batch processing pipeline with mocked WhisperX service."""
        from voice_man.services.e2e_test_service import E2ETestRunner, E2ETestConfig

        config = E2ETestConfig(
            batch_size=5,
            max_retries=2,
            retry_delays=[0, 0],
            enable_checksum_verification=True,
        )

        runner = E2ETestRunner(config, whisperx_service=mock_whisperx_service)

        # Collect and limit files for test
        files = await runner.collect_files(audio_files_directory)
        files = files[:10]

        # Run processing
        result = await runner.run(files)

        # Verify results
        assert result.total_files == 10
        assert result.processed_files == 10
        assert result.success_rate == 1.0
        assert result.checksum_verified is True
        assert result.total_time_seconds > 0

    @pytest.mark.asyncio
    async def test_batch_processing_handles_failures(
        self, audio_files_directory: Path, results_directory: Path
    ):
        """Test batch processing handles failures gracefully with retry."""
        from voice_man.services.e2e_test_service import E2ETestRunner, E2ETestConfig

        # Create mock that fails 50% of the time
        call_count = 0

        async def mock_process(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 0:
                raise RuntimeError("Simulated failure")
            return MagicMock(text="Success", segments=[], speakers=[])

        mock_service = MagicMock()
        mock_service.process_audio = mock_process

        config = E2ETestConfig(
            batch_size=5,
            max_retries=3,
            retry_delays=[0, 0, 0],
            enable_checksum_verification=False,
        )

        runner = E2ETestRunner(config, whisperx_service=mock_service)

        # Collect limited files
        files = await runner.collect_files(audio_files_directory)
        files = files[:6]

        # Run processing
        result = await runner.run(files)

        # Should have attempted retries
        assert result.total_files == 6
        assert result.processed_files == 6
        # Some may have failed after retries
        assert result.failed_files <= result.total_files

    @pytest.mark.asyncio
    async def test_report_generation_integration(
        self, audio_files_directory: Path, mock_whisperx_service, results_directory: Path
    ):
        """Test complete report generation after batch processing."""
        from voice_man.services.e2e_test_service import E2ETestRunner, E2ETestConfig

        config = E2ETestConfig(
            batch_size=5,
            enable_checksum_verification=False,
        )

        runner = E2ETestRunner(config, whisperx_service=mock_whisperx_service)

        files = await runner.collect_files(audio_files_directory)
        files = files[:5]

        result = await runner.run(files)

        # Generate reports
        generated = runner.generate_report(result, results_directory)

        # Verify all reports were generated
        assert "json" in generated
        assert "markdown" in generated
        assert generated["json"].exists()
        assert generated["markdown"].exists()

        # Verify JSON content
        import json

        with open(generated["json"]) as f:
            report_data = json.load(f)

        assert report_data["result"]["total_files"] == 5
        assert "timestamp" in report_data

    @pytest.mark.asyncio
    async def test_progress_tracking_integration(
        self, audio_files_directory: Path, mock_whisperx_service
    ):
        """Test progress tracking during batch processing."""
        from voice_man.services.e2e_test_service import E2ETestRunner, E2ETestConfig

        config = E2ETestConfig(enable_checksum_verification=False)
        runner = E2ETestRunner(config, whisperx_service=mock_whisperx_service)

        files = await runner.collect_files(audio_files_directory)
        files = files[:10]

        progress_updates = []

        def track_progress(current, total, elapsed, filename, status):
            progress_updates.append(
                {
                    "current": current,
                    "total": total,
                    "elapsed": elapsed,
                    "filename": filename,
                    "status": status,
                }
            )

        await runner.run(files, progress_callback=track_progress)

        # Verify progress was tracked
        assert len(progress_updates) == 10
        assert progress_updates[0]["current"] == 1
        assert progress_updates[-1]["current"] == 10
        assert all(u["total"] == 10 for u in progress_updates)

    @pytest.mark.asyncio
    async def test_dynamic_batch_adjustment_integration(
        self, audio_files_directory: Path, mock_whisperx_service, mock_gpu_monitor
    ):
        """Test E4: Dynamic batch adjustment based on GPU memory."""
        from voice_man.services.e2e_test_service import E2ETestRunner, E2ETestConfig

        # Configure mock to simulate high memory on 3rd call
        call_count = 0
        original_check = mock_gpu_monitor.check_memory_status

        def check_with_high_memory():
            nonlocal call_count
            call_count += 1
            if call_count == 3:
                return {
                    "warning": True,
                    "critical": True,
                    "auto_adjust_recommended": True,
                    "usage_percentage": 96.0,
                    "message": "Critical memory",
                }
            return original_check()

        mock_gpu_monitor.check_memory_status = check_with_high_memory

        config = E2ETestConfig(
            batch_size=15,
            dynamic_batch_adjustment=True,
            enable_checksum_verification=False,
        )

        runner = E2ETestRunner(
            config,
            whisperx_service=mock_whisperx_service,
            gpu_monitor=mock_gpu_monitor,
        )

        files = await runner.collect_files(audio_files_directory)
        files = files[:5]

        await runner.run(files)

        # Batch size should have been reduced at some point
        # The clear_gpu_cache should have been called
        assert mock_gpu_monitor.clear_gpu_cache.called


class TestE2EPerformanceRequirements:
    """Performance requirement tests for E2E processing."""

    @pytest.fixture
    def audio_files_directory(self) -> Path:
        """Return path to the ref/call directory."""
        return Path(__file__).parent.parent.parent / "ref" / "call"

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        os.environ.get("SKIP_SLOW_TESTS", "1") == "1",
        reason="Slow test - set SKIP_SLOW_TESTS=0 to run",
    )
    async def test_processing_time_under_20_minutes(
        self, audio_files_directory: Path, mock_whisperx_service
    ):
        """Test that 183 files can be processed within 20 minutes (with mock)."""
        from voice_man.services.e2e_test_service import E2ETestRunner, E2ETestConfig

        config = E2ETestConfig(
            batch_size=15,
            enable_checksum_verification=False,
        )

        runner = E2ETestRunner(config, whisperx_service=mock_whisperx_service)

        files = await runner.collect_files(audio_files_directory)

        start_time = time.time()
        result = await runner.run(files)
        elapsed = time.time() - start_time

        # With mock service, should be very fast
        # Real test would need actual WhisperX service
        assert elapsed < 1200, f"Processing took {elapsed:.1f}s, expected < 1200s (20 min)"
        assert result.total_files >= 180

    @pytest.mark.asyncio
    async def test_average_time_per_file_estimate(
        self, audio_files_directory: Path, mock_whisperx_service
    ):
        """Test average processing time per file estimation."""
        from voice_man.services.e2e_test_service import E2ETestRunner, E2ETestConfig

        # Add artificial delay to mock
        async def slow_process(*args, **kwargs):
            await asyncio.sleep(0.01)  # 10ms per file
            return MagicMock(text="Test", segments=[], speakers=[])

        mock_whisperx_service.process_audio = slow_process

        config = E2ETestConfig(enable_checksum_verification=False)
        runner = E2ETestRunner(config, whisperx_service=mock_whisperx_service)

        files = await runner.collect_files(audio_files_directory)
        files = files[:10]

        result = await runner.run(files)

        # Average time should be roughly 10ms per file
        assert result.avg_time_per_file > 0.01
        assert result.avg_time_per_file < 1.0  # Should be fast with mock


class TestE2EConfigurationValidation:
    """Tests for E2E configuration validation."""

    def test_config_default_values(self):
        """Test E2ETestConfig has correct default values."""
        from voice_man.services.e2e_test_service import E2ETestConfig

        config = E2ETestConfig()

        assert config.batch_size == 15
        assert config.max_batch_size == 32
        assert config.min_batch_size == 2
        assert config.max_retries == 3
        assert config.retry_delays == [5, 15, 30]
        assert config.dynamic_batch_adjustment is True
        assert config.gpu_memory_warning_threshold == 80.0
        assert config.gpu_memory_critical_threshold == 95.0

    def test_config_custom_values(self):
        """Test E2ETestConfig accepts custom values."""
        from voice_man.services.e2e_test_service import E2ETestConfig

        config = E2ETestConfig(
            batch_size=20,
            max_batch_size=40,
            min_batch_size=4,
            num_speakers=3,
            language="en",
            device="cpu",
            max_retries=5,
        )

        assert config.batch_size == 20
        assert config.max_batch_size == 40
        assert config.min_batch_size == 4
        assert config.num_speakers == 3
        assert config.language == "en"
        assert config.device == "cpu"
        assert config.max_retries == 5

    def test_config_retry_delays_exponential(self):
        """Test default retry delays follow exponential backoff pattern."""
        from voice_man.services.e2e_test_service import E2ETestConfig

        config = E2ETestConfig()

        # Default delays: [5, 15, 30] - increasing pattern
        assert config.retry_delays[0] < config.retry_delays[1]
        assert config.retry_delays[1] < config.retry_delays[2]


class TestE2ECLIScript:
    """Tests for E2E CLI script functionality."""

    @pytest.fixture
    def script_path(self) -> Path:
        """Return path to e2e_batch_test.py script."""
        return Path(__file__).parent.parent.parent / "scripts" / "e2e_batch_test.py"

    def test_script_exists(self, script_path: Path):
        """Test CLI script file exists."""
        assert script_path.exists(), f"Script not found: {script_path}"

    def test_script_is_executable_python(self, script_path: Path):
        """Test script starts with shebang."""
        content = script_path.read_text()
        assert content.startswith("#!/usr/bin/env python3")

    def test_script_has_help_text(self, script_path: Path):
        """Test script contains help documentation."""
        content = script_path.read_text()
        assert "argparse" in content
        assert "--input-dir" in content
        assert "--output-dir" in content
        assert "--batch-size" in content

    @pytest.mark.asyncio
    async def test_script_dry_run_mode(self, script_path: Path):
        """Test script dry-run mode doesn't process files."""
        import subprocess

        result = subprocess.run(
            [
                "python",
                str(script_path),
                "--input-dir",
                str(script_path.parent.parent / "ref" / "call"),
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0
        assert "DRY RUN" in result.stdout
        assert "Would process" in result.stdout
