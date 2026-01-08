"""
Unit tests for report template system.

Test coverage for:
- Jinja2 template structure
- Legal report sections (cover, table of contents, body, appendix)
- Data model-template binding
- Mock data rendering
"""

import pytest
from pathlib import Path
from voice_man.services.report_template_service import ReportTemplateService


class TestReportTemplateServiceInitialization:
    """Test report template service initialization."""

    def test_template_service_initialization(self):
        """Test that template service can be initialized."""
        # Arrange & Act
        service = ReportTemplateService()

        # Assert
        assert service is not None
        assert service.template_dir is not None
        assert service.env is not None

    def test_template_service_custom_directory(self, tmp_path):
        """Test template service with custom template directory."""
        # Arrange
        template_dir = tmp_path / "templates"
        template_dir.mkdir()

        # Act
        service = ReportTemplateService(template_dir=str(template_dir))

        # Assert
        assert service.template_dir == template_dir


class TestLegalReportTemplates:
    """Test legal report template structure."""

    def test_cover_page_template_exists(self):
        """Test that cover page template exists."""
        # Arrange
        service = ReportTemplateService()

        # Act
        template_exists = service.template_exists("cover_page.html")

        # Assert
        assert template_exists, "Cover page template should exist"

    def test_table_of_contents_template_exists(self):
        """Test that table of contents template exists."""
        # Arrange
        service = ReportTemplateService()

        # Act
        template_exists = service.template_exists("table_of_contents.html")

        # Assert
        assert template_exists, "Table of contents template should exist"

    def test_body_section_template_exists(self):
        """Test that body section template exists."""
        # Arrange
        service = ReportTemplateService()

        # Act
        template_exists = service.template_exists("body_section.html")

        # Assert
        assert template_exists, "Body section template should exist"

    def test_appendix_template_exists(self):
        """Test that appendix template exists."""
        # Arrange
        service = ReportTemplateService()

        # Act
        template_exists = service.template_exists("appendix.html")

        # Assert
        assert template_exists, "Appendix template should exist"


class TestTemplateRendering:
    """Test template rendering with mock data."""

    def test_render_cover_page_with_mock_data(self):
        """Test rendering cover page with mock data."""
        # Arrange
        service = ReportTemplateService()
        mock_data = {
            "report_title": "음성 녹취 증거 분석 보고서",
            "report_number": "VOICE-2025-001",
            "report_date": "2025-01-08",
            "analyst_name": "지니",
            "case_number": "CASE-2025-123",
        }

        # Act
        rendered_html = service.render_template("cover_page.html", mock_data)

        # Assert
        assert rendered_html is not None
        assert "음성 녹취 증거 분석 보고서" in rendered_html
        assert "VOICE-2025-001" in rendered_html
        assert "2025-01-08" in rendered_html

    def test_render_body_section_with_analysis_results(self):
        """Test rendering body section with analysis results."""
        # Arrange
        service = ReportTemplateService()
        mock_data = {
            "section_title": "범죄 발언 분석 결과",
            "total_crime_statements": 15,
            "threat_count": 5,
            "fraud_count": 3,
            "insult_count": 7,
            "crime_statements": [
                {
                    "timestamp": "00:01:23",
                    "speaker": "Speaker A",
                    "text": "너 그렇게 살면 큰일 난다.",
                    "crime_type": "협박",
                    "confidence": 0.92,
                },
                {
                    "timestamp": "00:02:45",
                    "speaker": "Speaker A",
                    "text": "돈 내놔라.",
                    "crime_type": "공갈",
                    "confidence": 0.88,
                },
            ],
        }

        # Act
        rendered_html = service.render_template("body_section.html", mock_data)

        # Assert
        assert rendered_html is not None
        assert "범죄 발언 분석 결과" in rendered_html
        assert "15" in rendered_html  # total_crime_statements
        assert "협박" in rendered_html

    def test_render_appendix_with_legal_references(self):
        """Test rendering appendix with legal references."""
        # Arrange
        service = ReportTemplateService()
        mock_data = {
            "legal_references": [
                {
                    "article": "형법 제283조",
                    "title": "협박죄",
                    "description": "사람을 협박하여 공포심을 일으킨 자는 3년 이하의 징역 또는 500만원 이하의 벌금에 처한다.",
                },
                {
                    "article": "형법 제328조",
                    "title": "공갈죄",
                    "description": "사람을 공갈하여 재물을 교부받은 자는 10년 이하의 징역에 처한다.",
                },
            ]
        }

        # Act
        rendered_html = service.render_template("appendix.html", mock_data)

        # Assert
        assert rendered_html is not None
        assert "형법 제283조" in rendered_html
        assert "협박죄" in rendered_html


class TestDataModelBinding:
    """Test data model to template binding."""

    def test_bind_audio_file_model_to_template(self):
        """Test binding AudioFile model to template."""
        # Arrange
        service = ReportTemplateService()
        mock_audio_file = {
            "id": "audio-123",
            "original_filename": "recording.mp3",
            "duration_seconds": 300,
            "upload_timestamp": "2025-01-08T10:00:00",
            "file_hash": "abc123def456",
        }

        # Act
        context = service.prepare_context_from_audio_file(mock_audio_file)

        # Assert
        assert context is not None
        assert "audio-123" in context.values()
        assert "recording.mp3" in context.values()

    def test_bind_transcript_model_to_template(self):
        """Test binding Transcript model to template."""
        # Arrange
        service = ReportTemplateService()
        mock_transcript = {
            "id": "transcript-456",
            "audio_id": "audio-123",
            "content": "전체 대본 내용...",
            "segments": [
                {
                    "speaker_id": "Speaker A",
                    "start_time": 0,
                    "end_time": 10,
                    "text": "안녕하세요",
                },
                {
                    "speaker_id": "Speaker B",
                    "start_time": 10,
                    "end_time": 20,
                    "text": "네 안녕하세요",
                },
            ],
        }

        # Act
        context = service.prepare_context_from_transcript(mock_transcript)

        # Assert
        assert context is not None
        assert "transcript-456" in context.values()
        assert len(context["segments"]) == 2


class TestCompleteReportGeneration:
    """Test complete report generation with all sections."""

    def test_generate_complete_report_html(self):
        """Test generating complete report HTML with all sections."""
        # Arrange
        service = ReportTemplateService()
        report_data = {
            "report_title": "음성 녹취 증거 분석 보고서",
            "report_number": "VOICE-2025-001",
            "report_date": "2025-01-08",
            "analyst_name": "지니",
            "case_number": "CASE-2025-123",
            "sections": [
                {
                    "type": "body_section",
                    "data": {
                        "section_title": "범죄 발언 분석",
                        "crime_statements": [
                            {
                                "timestamp": "00:01:23",
                                "speaker": "Speaker A",
                                "text": "협박 발언",
                                "crime_type": "협박",
                                "confidence": 0.92,
                            }
                        ],
                    },
                }
            ],
        }

        # Act
        complete_html = service.generate_complete_report(report_data)

        # Assert
        assert complete_html is not None
        assert "음성 녹취 증거 분석 보고서" in complete_html
        assert "범죄 발언 분석" in complete_html
        assert "<!DOCTYPE html>" in complete_html
        assert "</html>" in complete_html
