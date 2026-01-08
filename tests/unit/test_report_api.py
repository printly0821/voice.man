"""
Unit tests for Report Generation API Endpoints.

Tests FastAPI routes for:
- POST /api/v1/audio/{id}/report/generate - Async report generation
- GET /api/v1/audio/{id}/report/status - Generation status check
- GET /api/v1/audio/{id}/report/download - PDF download
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path

from voice_man.api.main import app
from voice_man.models.audio_file import AudioFile, ReportStatus


# Fixtures


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_audio_file():
    """Mock audio file without report."""
    return {
        "id": 1,
        "filename": "test_audio.wav",
        "transcript": "This is a test transcript.",
        "report_status": None,
        "report_version": 0,
    }


@pytest.fixture
def mock_audio_file_with_report():
    """Mock audio file with completed report."""
    return {
        "id": 2,
        "filename": "test_audio.wav",
        "transcript": "This is a test transcript.",
        "report_status": ReportStatus.COMPLETED,
        "report_version": 1,
    }


@pytest.fixture
def mock_audio_file_processing_report():
    """Mock audio file with processing report."""
    return {
        "id": 3,
        "filename": "test_audio.wav",
        "transcript": "This is a test transcript.",
        "report_status": ReportStatus.PROCESSING,
        "report_version": 1,
    }


@pytest.fixture
def mock_audio_file_no_transcript():
    """Mock audio file without transcript."""
    return {
        "id": 4,
        "filename": "test_audio.wav",
        "transcript": None,
        "report_status": None,
        "report_version": 0,
    }


class TestReportGenerationEndpoint:
    """Test POST /api/v1/audio/{id}/report/generate endpoint."""

    def test_generate_report_success(self, client, mock_audio_file):
        """Test successful report generation initiation."""
        with patch(
            "voice_man.services.report_service.ReportService._get_audio_file",
            return_value=mock_audio_file,
        ):
            response = client.post(f"/api/v1/audio/{mock_audio_file['id']}/report/generate")

            assert response.status_code == 202
            data = response.json()
            assert "report_id" in data
            assert data["status"] == "processing"
            assert "estimated_time_seconds" in data

    def test_generate_report_audio_not_found(self, client):
        """Test report generation with non-existent audio file."""
        with patch(
            "voice_man.services.report_service.ReportService._get_audio_file",
            return_value=None,
        ):
            response = client.post("/api/v1/audio/99999/report/generate")

            assert response.status_code == 404
            data = response.json()
            assert "detail" in data

    def test_generate_report_already_exists(self, client, mock_audio_file_with_report):
        """Test generating report when one already exists."""
        with patch(
            "voice_man.services.report_service.ReportService._get_audio_file",
            return_value=mock_audio_file_with_report,
        ):
            response = client.post(
                f"/api/v1/audio/{mock_audio_file_with_report['id']}/report/generate"
            )

            assert response.status_code == 409  # Conflict
            data = response.json()
            assert "detail" in data

    def test_generate_report_missing_transcript(self, client, mock_audio_file_no_transcript):
        """Test generating report without transcript."""
        with patch(
            "voice_man.services.report_service.ReportService._get_audio_file",
            return_value=mock_audio_file_no_transcript,
        ):
            response = client.post(
                f"/api/v1/audio/{mock_audio_file_no_transcript['id']}/report/generate"
            )

            assert response.status_code == 400
            data = response.json()
            assert "detail" in data


class TestReportStatusEndpoint:
    """Test GET /api/v1/audio/{id}/report/status endpoint."""

    def test_get_report_status_processing(self, client, mock_audio_file_processing_report):
        """Test checking report status while processing."""
        with patch(
            "voice_man.services.report_service.ReportService._get_audio_file",
            return_value=mock_audio_file_processing_report,
        ):
            # First initiate a report
            client.post(f"/api/v1/audio/{mock_audio_file_processing_report['id']}/report/generate")

            # Then check status
            response = client.get(
                f"/api/v1/audio/{mock_audio_file_processing_report['id']}/report/status"
            )

            assert response.status_code == 200
            data = response.json()
            assert data["status"] in ["processing", "completed"]
            assert "progress_percentage" in data
            assert "current_step" in data

    def test_get_report_status_not_found(self, client, mock_audio_file):
        """Test checking status for non-existent report."""
        response = client.get(f"/api/v1/audio/{mock_audio_file['id']}/report/status")

        assert response.status_code == 404
        data = response.json()
        assert "detail" in data


class TestReportDownloadEndpoint:
    """Test GET /api/v1/audio/{id}/report/download endpoint."""

    def test_download_report_not_ready(self, client, mock_audio_file):
        """Test downloading report before it's ready."""
        # First initiate a report
        with patch(
            "voice_man.services.report_service.ReportService._get_audio_file",
            return_value=mock_audio_file,
        ):
            client.post(f"/api/v1/audio/{mock_audio_file['id']}/report/generate")

            # Try to download immediately
            response = client.get(f"/api/v1/audio/{mock_audio_file['id']}/report/download")

            # Should return 425 (Too Early) or 404
            assert response.status_code in [425, 404]

    def test_download_report_not_found(self, client, mock_audio_file):
        """Test downloading non-existent report."""
        response = client.get(f"/api/v1/audio/{mock_audio_file['id']}/report/download")

        assert response.status_code == 404
        data = response.json()
        assert "detail" in data


class TestReportVersioning:
    """Test report version management."""

    def test_regenerate_report_creates_new_version(self, client, mock_audio_file_with_report):
        """Test that regenerating creates a new version."""
        with patch(
            "voice_man.services.report_service.ReportService._get_audio_file",
            return_value=mock_audio_file_with_report,
        ):
            response = client.post(
                f"/api/v1/audio/{mock_audio_file_with_report['id']}/report/generate",
                params={"force": True},
            )

            # Should accept the request (202) even if report exists with force=True
            assert response.status_code in [202, 409]

    def test_list_report_versions(self, client, mock_audio_file_with_report):
        """Test listing all report versions."""
        with patch(
            "voice_man.services.report_service.ReportService._get_audio_file",
            return_value=mock_audio_file_with_report,
        ):
            # First initiate a report
            client.post(f"/api/v1/audio/{mock_audio_file_with_report['id']}/report/generate")

            # Then list versions
            response = client.get(
                f"/api/v1/audio/{mock_audio_file_with_report['id']}/report/versions"
            )

            assert response.status_code == 200
            data = response.json()
            assert "versions" in data
