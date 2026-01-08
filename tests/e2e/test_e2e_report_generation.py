"""
RED Phase Tests for Report Generation.

Tests for E2ETestRunner.generate_report method.
TASK-008: Comprehensive report generation service.
"""

import json
import pytest
from pathlib import Path


class TestReportGeneration:
    """Tests for report generation functionality."""

    @pytest.fixture
    def sample_e2e_result(self):
        """Create a sample E2ETestResult for testing."""
        from voice_man.services.e2e_test_service import E2ETestResult, FileProcessingResult

        file_results = [
            FileProcessingResult(
                file_path="/path/audio1.m4a",
                status="success",
                processing_time_seconds=5.5,
                transcript_text="Hello world",
                segments=[{"start": 0.0, "end": 1.0, "text": "Hello"}],
                speakers=["SPEAKER_00"],
                error=None,
            ),
            FileProcessingResult(
                file_path="/path/audio2.m4a",
                status="failed",
                processing_time_seconds=1.2,
                transcript_text=None,
                segments=None,
                speakers=None,
                error="GPU out of memory",
            ),
            FileProcessingResult(
                file_path="/path/audio3.m4a",
                status="success",
                processing_time_seconds=4.3,
                transcript_text="Test transcript",
                segments=[],
                speakers=["SPEAKER_00", "SPEAKER_01"],
                error=None,
            ),
        ]

        return E2ETestResult(
            total_files=3,
            processed_files=3,
            failed_files=1,
            total_time_seconds=11.0,
            avg_time_per_file=3.67,
            gpu_stats={
                "total_mb": 24576,
                "used_mb": 8192,
                "free_mb": 16384,
                "usage_percentage": 33.3,
            },
            file_results=file_results,
            checksum_verified=True,
        )

    def test_generate_report_creates_json_file(self, tmp_path: Path, sample_e2e_result):
        """Test generate_report creates JSON report file."""
        from voice_man.services.e2e_test_service import E2ETestRunner, E2ETestConfig

        config = E2ETestConfig()
        runner = E2ETestRunner(config)

        generated = runner.generate_report(sample_e2e_result, tmp_path)

        assert "json" in generated
        assert generated["json"].exists()
        assert generated["json"].name == "e2e_test_report.json"

    def test_generate_report_json_content(self, tmp_path: Path, sample_e2e_result):
        """Test JSON report contains expected content."""
        from voice_man.services.e2e_test_service import E2ETestRunner, E2ETestConfig

        config = E2ETestConfig()
        runner = E2ETestRunner(config)

        generated = runner.generate_report(sample_e2e_result, tmp_path)

        with open(generated["json"]) as f:
            data = json.load(f)

        assert "timestamp" in data
        assert "config" in data
        assert "result" in data
        assert data["result"]["total_files"] == 3
        assert data["result"]["failed_files"] == 1
        assert data["result"]["success_rate"] == pytest.approx(0.667, rel=0.01)

    def test_generate_report_creates_markdown_file(self, tmp_path: Path, sample_e2e_result):
        """Test generate_report creates Markdown summary file."""
        from voice_man.services.e2e_test_service import E2ETestRunner, E2ETestConfig

        config = E2ETestConfig()
        runner = E2ETestRunner(config)

        generated = runner.generate_report(sample_e2e_result, tmp_path)

        assert "markdown" in generated
        assert generated["markdown"].exists()
        assert generated["markdown"].name == "e2e_test_summary.md"

    def test_generate_report_markdown_content(self, tmp_path: Path, sample_e2e_result):
        """Test Markdown report contains expected sections."""
        from voice_man.services.e2e_test_service import E2ETestRunner, E2ETestConfig

        config = E2ETestConfig()
        runner = E2ETestRunner(config)

        generated = runner.generate_report(sample_e2e_result, tmp_path)

        content = generated["markdown"].read_text()

        assert "# E2E Test Summary Report" in content
        assert "## Overview" in content
        assert "Total Files" in content
        assert "Success Rate" in content
        assert "## GPU Statistics" in content
        assert "## Failed Files" in content
        assert "GPU out of memory" in content

    def test_generate_report_creates_failed_files_json(self, tmp_path: Path, sample_e2e_result):
        """Test generate_report creates failed_files.json when failures exist."""
        from voice_man.services.e2e_test_service import E2ETestRunner, E2ETestConfig

        config = E2ETestConfig()
        runner = E2ETestRunner(config)

        generated = runner.generate_report(sample_e2e_result, tmp_path)

        assert "failed_files" in generated
        assert generated["failed_files"].exists()
        assert generated["failed_files"].name == "failed_files.json"

    def test_generate_report_failed_files_content(self, tmp_path: Path, sample_e2e_result):
        """Test failed_files.json contains expected content."""
        from voice_man.services.e2e_test_service import E2ETestRunner, E2ETestConfig

        config = E2ETestConfig()
        runner = E2ETestRunner(config)

        generated = runner.generate_report(sample_e2e_result, tmp_path)

        with open(generated["failed_files"]) as f:
            data = json.load(f)

        assert "timestamp" in data
        assert data["failed_count"] == 1
        assert len(data["files"]) == 1
        assert data["files"][0]["file_path"] == "/path/audio2.m4a"
        assert data["files"][0]["error"] == "GPU out of memory"

    def test_generate_report_no_failed_files_json_when_no_failures(self, tmp_path: Path):
        """Test failed_files.json not created when no failures."""
        from voice_man.services.e2e_test_service import (
            E2ETestRunner,
            E2ETestConfig,
            E2ETestResult,
            FileProcessingResult,
        )

        # Result with no failures
        result = E2ETestResult(
            total_files=2,
            processed_files=2,
            failed_files=0,
            total_time_seconds=10.0,
            avg_time_per_file=5.0,
            gpu_stats={},
            file_results=[
                FileProcessingResult(
                    file_path="/path/audio1.m4a",
                    status="success",
                    processing_time_seconds=5.0,
                    transcript_text="Test",
                    segments=[],
                    speakers=[],
                    error=None,
                ),
            ],
            checksum_verified=True,
        )

        config = E2ETestConfig()
        runner = E2ETestRunner(config)

        generated = runner.generate_report(result, tmp_path)

        assert "failed_files" not in generated

    def test_generate_report_creates_output_directory(self, tmp_path: Path, sample_e2e_result):
        """Test generate_report creates output directory if not exists."""
        from voice_man.services.e2e_test_service import E2ETestRunner, E2ETestConfig

        output_dir = tmp_path / "nested" / "output" / "dir"
        assert not output_dir.exists()

        config = E2ETestConfig()
        runner = E2ETestRunner(config)

        generated = runner.generate_report(sample_e2e_result, output_dir)

        assert output_dir.exists()
        assert len(generated) >= 2

    def test_generate_report_returns_absolute_paths(self, tmp_path: Path, sample_e2e_result):
        """Test generate_report returns absolute file paths."""
        from voice_man.services.e2e_test_service import E2ETestRunner, E2ETestConfig

        config = E2ETestConfig()
        runner = E2ETestRunner(config)

        generated = runner.generate_report(sample_e2e_result, tmp_path)

        for path in generated.values():
            assert path.is_absolute()

    def test_generate_report_markdown_no_gpu_stats(self, tmp_path: Path):
        """Test Markdown report handles missing GPU stats gracefully."""
        from voice_man.services.e2e_test_service import (
            E2ETestRunner,
            E2ETestConfig,
            E2ETestResult,
        )

        result = E2ETestResult(
            total_files=1,
            processed_files=1,
            failed_files=0,
            total_time_seconds=5.0,
            avg_time_per_file=5.0,
            gpu_stats={},  # Empty GPU stats
            file_results=[],
            checksum_verified=True,
        )

        config = E2ETestConfig()
        runner = E2ETestRunner(config)

        generated = runner.generate_report(result, tmp_path)

        content = generated["markdown"].read_text()
        assert "No GPU statistics available" in content

    def test_generate_report_checksum_failed_status(self, tmp_path: Path):
        """Test Markdown report shows checksum verification failure."""
        from voice_man.services.e2e_test_service import (
            E2ETestRunner,
            E2ETestConfig,
            E2ETestResult,
        )

        result = E2ETestResult(
            total_files=1,
            processed_files=1,
            failed_files=0,
            total_time_seconds=5.0,
            avg_time_per_file=5.0,
            gpu_stats={},
            file_results=[],
            checksum_verified=False,  # Checksum failed!
        )

        config = E2ETestConfig()
        runner = E2ETestRunner(config)

        generated = runner.generate_report(result, tmp_path)

        content = generated["markdown"].read_text()
        assert "FAILED" in content
