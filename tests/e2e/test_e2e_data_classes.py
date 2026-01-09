"""
RED Phase Tests for E2E Data Classes.

Tests for FileProcessingResult and E2ETestResult dataclasses.
"""

import pytest
from typing import List, Dict, Any, Optional


class TestFileProcessingResult:
    """Tests for FileProcessingResult dataclass."""

    def test_file_processing_result_creation_success(self):
        """Test creating FileProcessingResult with success status."""
        from voice_man.services.e2e_test_service import FileProcessingResult

        result = FileProcessingResult(
            file_path="/path/to/audio.m4a",
            status="success",
            processing_time_seconds=5.5,
            transcript_text="Hello world",
            segments=[{"start": 0.0, "end": 1.0, "text": "Hello", "speaker": "SPEAKER_00"}],
            speakers=["SPEAKER_00", "SPEAKER_01"],
            error=None,
        )

        assert result.file_path == "/path/to/audio.m4a"
        assert result.status == "success"
        assert result.processing_time_seconds == 5.5
        assert result.transcript_text == "Hello world"
        assert len(result.segments) == 1
        assert result.speakers == ["SPEAKER_00", "SPEAKER_01"]
        assert result.error is None

    def test_file_processing_result_creation_failed(self):
        """Test creating FileProcessingResult with failed status."""
        from voice_man.services.e2e_test_service import FileProcessingResult

        result = FileProcessingResult(
            file_path="/path/to/audio.m4a",
            status="failed",
            processing_time_seconds=1.2,
            transcript_text=None,
            segments=None,
            speakers=None,
            error="GPU out of memory",
        )

        assert result.status == "failed"
        assert result.transcript_text is None
        assert result.segments is None
        assert result.error == "GPU out of memory"

    def test_file_processing_result_creation_skipped(self):
        """Test creating FileProcessingResult with skipped status."""
        from voice_man.services.e2e_test_service import FileProcessingResult

        result = FileProcessingResult(
            file_path="/path/to/audio.m4a",
            status="skipped",
            processing_time_seconds=0.0,
            transcript_text=None,
            segments=None,
            speakers=None,
            error="File already processed",
        )

        assert result.status == "skipped"
        assert result.processing_time_seconds == 0.0

    def test_file_processing_result_to_dict(self):
        """Test FileProcessingResult to_dict method."""
        from voice_man.services.e2e_test_service import FileProcessingResult

        result = FileProcessingResult(
            file_path="/path/to/audio.m4a",
            status="success",
            processing_time_seconds=5.5,
            transcript_text="Hello",
            segments=[],
            speakers=["SPEAKER_00"],
            error=None,
        )

        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert result_dict["file_path"] == "/path/to/audio.m4a"
        assert result_dict["status"] == "success"


class TestE2ETestResult:
    """Tests for E2ETestResult dataclass."""

    def test_e2e_test_result_creation(self):
        """Test creating E2ETestResult with all fields."""
        from voice_man.services.e2e_test_service import E2ETestResult, FileProcessingResult

        file_result = FileProcessingResult(
            file_path="/path/to/audio.m4a",
            status="success",
            processing_time_seconds=5.5,
            transcript_text="Hello",
            segments=[],
            speakers=["SPEAKER_00"],
            error=None,
        )

        result = E2ETestResult(
            total_files=10,
            processed_files=10,
            failed_files=1,
            total_time_seconds=120.5,
            avg_time_per_file=12.05,
            gpu_stats={"usage_percentage": 45.0, "total_mb": 24576},
            file_results=[file_result],
            checksum_verified=True,
        )

        assert result.total_files == 10
        assert result.processed_files == 10
        assert result.failed_files == 1
        assert result.total_time_seconds == 120.5
        assert result.avg_time_per_file == 12.05
        assert result.checksum_verified is True
        assert len(result.file_results) == 1

    def test_e2e_test_result_success_rate(self):
        """Test E2ETestResult success_rate property."""
        from voice_man.services.e2e_test_service import E2ETestResult

        result = E2ETestResult(
            total_files=10,
            processed_files=10,
            failed_files=2,
            total_time_seconds=100.0,
            avg_time_per_file=10.0,
            gpu_stats={},
            file_results=[],
            checksum_verified=True,
        )

        assert result.success_rate == 0.8  # (10 - 2) / 10

    def test_e2e_test_result_success_rate_zero_files(self):
        """Test E2ETestResult success_rate with zero files."""
        from voice_man.services.e2e_test_service import E2ETestResult

        result = E2ETestResult(
            total_files=0,
            processed_files=0,
            failed_files=0,
            total_time_seconds=0.0,
            avg_time_per_file=0.0,
            gpu_stats={},
            file_results=[],
            checksum_verified=True,
        )

        assert result.success_rate == 0.0

    def test_e2e_test_result_to_dict(self):
        """Test E2ETestResult to_dict method."""
        from voice_man.services.e2e_test_service import E2ETestResult

        result = E2ETestResult(
            total_files=10,
            processed_files=10,
            failed_files=1,
            total_time_seconds=120.5,
            avg_time_per_file=12.05,
            gpu_stats={"usage_percentage": 45.0},
            file_results=[],
            checksum_verified=True,
        )

        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert result_dict["total_files"] == 10
        assert result_dict["success_rate"] == 0.9
        assert "gpu_stats" in result_dict

    def test_e2e_test_result_get_failed_files(self):
        """Test E2ETestResult get_failed_files method."""
        from voice_man.services.e2e_test_service import E2ETestResult, FileProcessingResult

        file_results = [
            FileProcessingResult(
                file_path="/path/audio1.m4a",
                status="success",
                processing_time_seconds=5.0,
                transcript_text="Test",
                segments=[],
                speakers=[],
                error=None,
            ),
            FileProcessingResult(
                file_path="/path/audio2.m4a",
                status="failed",
                processing_time_seconds=1.0,
                transcript_text=None,
                segments=None,
                speakers=None,
                error="Error occurred",
            ),
        ]

        result = E2ETestResult(
            total_files=2,
            processed_files=2,
            failed_files=1,
            total_time_seconds=6.0,
            avg_time_per_file=3.0,
            gpu_stats={},
            file_results=file_results,
            checksum_verified=True,
        )

        failed = result.get_failed_files()
        assert len(failed) == 1
        assert failed[0].file_path == "/path/audio2.m4a"
