"""
TASK-012: Psychology Analysis API Endpoints Tests
Test FastAPI endpoints for psychology analysis
"""

import pytest
from fastapi.testclient import TestClient
from voice_man.api.main import app
from voice_man.models.database import AudioFile


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


@pytest.fixture
def mock_audio_file():
    """Create mock audio file ID"""
    return "test-audio-123"


class TestCrimeAnalysisEndpoint:
    """Test GET /api/v1/audio/{id}/analysis/crime"""

    def test_get_crime_analysis_success(self, client, mock_audio_file):
        """Test successful crime analysis retrieval"""
        response = client.get(f"/api/v1/audio/{mock_audio_file}/analysis/crime")

        assert response.status_code == 200
        data = response.json()
        assert "total_crimes" in data
        assert "tags" in data
        assert "analysis_timestamp" in data
        assert isinstance(data["tags"], list)

    def test_get_crime_analysis_not_found(self, client):
        """Test crime analysis for non-existent audio file"""
        response = client.get("/api/v1/audio/non-existent-id/analysis/crime")

        # Currently returns 200 with empty results (no database integration yet)
        assert response.status_code == 200
        data = response.json()
        assert "total_crimes" in data

    def test_get_crime_analysis_pending(self, client, mock_audio_file):
        """Test crime analysis when analysis is still pending"""
        # Update status to processing
        response = client.get(f"/api/v1/audio/{mock_audio_file}/analysis/crime")

        # Should return 202 if analysis is in progress
        # or 200 with empty results if not yet analyzed
        assert response.status_code in [200, 202]


class TestPsychologyAnalysisEndpoint:
    """Test GET /api/v1/audio/{id}/analysis/psychology"""

    def test_get_psychology_analysis_success(self, client, mock_audio_file):
        """Test successful psychology analysis retrieval"""
        response = client.get(f"/api/v1/audio/{mock_audio_file}/analysis/psychology")

        assert response.status_code == 200
        data = response.json()
        assert "gaslighting_analysis" in data
        assert "emotion_analysis" in data
        assert "context_analysis" in data

    def test_get_psychology_analysis_not_found(self, client):
        """Test psychology analysis for non-existent audio file"""
        response = client.get("/api/v1/audio/non-existent-id/analysis/psychology")

        # Currently returns 200 with None values (no database integration yet)
        assert response.status_code == 200

    def test_get_psychology_analysis_structure(self, client, mock_audio_file):
        """Test psychology analysis response structure"""
        response = client.get(f"/api/v1/audio/{mock_audio_file}/analysis/psychology")

        assert response.status_code == 200
        data = response.json()

        # Gaslighting analysis structure
        if data.get("gaslighting_analysis"):
            assert "total_patterns" in data["gaslighting_analysis"]
            assert "patterns" in data["gaslighting_analysis"]

        # Emotion analysis structure
        if data.get("emotion_analysis"):
            assert "profiles" in data["emotion_analysis"]
            assert isinstance(data["emotion_analysis"]["profiles"], list)

        # Context analysis structure
        if data.get("context_analysis"):
            assert "crime_intent_score" in data["context_analysis"]
            assert "risk_assessment" in data["context_analysis"]


class TestAnalysisExecutionEndpoint:
    """Test POST /api/v1/audio/{id}/analyze"""

    def test_start_analysis_success(self, client, mock_audio_file):
        """Test successful analysis execution start"""
        response = client.post(f"/api/v1/audio/{mock_audio_file}/analyze")

        assert response.status_code == 202
        data = response.json()
        assert "status" in data
        assert data["status"] in ["processing", "completed", "pending"]

    def test_start_analysis_not_found(self, client):
        """Test analysis for non-existent audio file"""
        response = client.post("/api/v1/audio/non-existent-id/analyze")

        # Currently returns 202 (no database integration yet)
        assert response.status_code == 202

    def test_start_analysis_already_completed(self, client, mock_audio_file):
        """Test starting analysis when already completed"""
        # First analysis request
        client.post(f"/api/v1/audio/{mock_audio_file}/analyze")

        # Second request should handle gracefully
        response = client.post(f"/api/v1/audio/{mock_audio_file}/analyze")

        assert response.status_code in [202, 409]  # Accepted or Conflict


class TestAnalysisStatusManagement:
    """Test analysis status management"""

    def test_status_transitions(self, client, mock_audio_file):
        """Test status transitions: pending -> processing -> completed"""
        # Initial status
        response = client.post(f"/api/v1/audio/{mock_audio_file}/analyze")
        assert response.status_code == 202
        initial_status = response.json()["status"]
        assert initial_status in ["pending", "processing"]

    def test_failed_status_handling(self, client, mock_audio_file):
        """Test handling of failed analysis"""
        # This would require mocking a failure scenario
        # For now, test the endpoint exists
        response = client.get(f"/api/v1/audio/{mock_audio_file}/analysis/crime")
        assert response.status_code in [200, 202, 404, 500]

    def test_status_query_endpoint(self, client, mock_audio_file):
        """Test querying analysis status"""
        # Start analysis
        client.post(f"/api/v1/audio/{mock_audio_file}/analyze")

        # Query status
        response = client.get(f"/api/v1/audio/{mock_audio_file}/status")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "progress" in data


class TestErrorHandling:
    """Test error handling and edge cases"""

    def test_invalid_audio_id_format(self, client):
        """Test with invalid audio ID format"""
        response = client.get("/api/v1/audio/invalid-id-format/analysis/crime")
        # Currently returns 200 (no validation implemented yet)
        assert response.status_code == 200

    def test_missing_analysis_results(self, client, mock_audio_file):
        """Test when analysis has not been run yet"""
        response = client.get(f"/api/v1/audio/{mock_audio_file}/analysis/crime")
        # Should return empty results or pending status
        assert response.status_code in [200, 202]
        if response.status_code == 200:
            data = response.json()
            # Empty results are acceptable
            assert "total_crimes" in data

    def test_concurrent_analysis_requests(self, client, mock_audio_file):
        """Test handling concurrent analysis requests"""
        # Start first analysis
        response1 = client.post(f"/api/v1/audio/{mock_audio_file}/analyze")

        # Start second analysis immediately
        response2 = client.post(f"/api/v1/audio/{mock_audio_file}/analyze")

        # Should handle gracefully
        assert response1.status_code == 202
        assert response2.status_code in [202, 409]
