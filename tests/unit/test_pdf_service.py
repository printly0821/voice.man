"""
Unit tests for PDF generation service.

Test coverage for:
- WeasyPrint initialization
- Korean font rendering
- Basic PDF generation with Korean text
- PDF metadata validation
"""

import pytest
from pathlib import Path
from io import BytesIO
from voice_man.services.pdf_service import PDFService


class TestPDFServiceInitialization:
    """Test PDF service initialization and configuration."""

    def test_pdf_service_initialization(self):
        """Test that PDF service can be initialized with default settings."""
        # Arrange & Act
        service = PDFService()

        # Assert
        assert service is not None
        assert service.template_dir is not None
        assert service.font_dir is not None

    def test_pdf_service_custom_directories(self, tmp_path):
        """Test PDF service with custom template and font directories."""
        # Arrange
        template_dir = tmp_path / "templates"
        font_dir = tmp_path / "fonts"
        template_dir.mkdir()
        font_dir.mkdir()

        # Act
        service = PDFService(template_dir=str(template_dir), font_dir=str(font_dir))

        # Assert
        assert service.template_dir == template_dir
        assert service.font_dir == font_dir


class TestKoreanFontRendering:
    """Test Korean font embedding and rendering."""

    def test_korean_font_available(self):
        """Test that Korean fonts are available for rendering."""
        # Arrange
        service = PDFService()

        # Act
        fonts = service.get_available_fonts()

        # Assert
        assert len(fonts) > 0
        assert any("nanum" in font.lower() or "gothic" in font.lower() for font in fonts)

    def test_generate_pdf_with_korean_text(self):
        """Test generating PDF with Korean text content."""
        # Arrange
        service = PDFService()
        html_content = """
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body { font-family: 'NanumGothic', sans-serif; }
            </style>
        </head>
        <body>
            <h1>음성 녹취 증거 분석 보고서</h1>
            <p>이 보고서는 음성 녹취 파일을 분석한 결과입니다.</p>
            <p>Korean text: 한글 테스트</p>
        </body>
        </html>
        """

        # Act
        pdf_bytes = service.generate_pdf_from_html(html_content)

        # Assert
        assert pdf_bytes is not None
        assert len(pdf_bytes) > 0
        # PDF file should start with %PDF-
        assert pdf_bytes.startswith(b"%PDF-")

    def test_pdf_korean_text_readability(self):
        """Test that Korean text in PDF is properly encoded."""
        # Arrange
        service = PDFService()
        html_content = """
        <html>
        <head><meta charset="UTF-8"></head>
        <body>
            <h1>협박 발언 분석</h1>
            <p>분석 결과: 위협적인 언어가 감지되었습니다.</p>
        </body>
        </html>
        """

        # Act
        pdf_bytes = service.generate_pdf_from_html(html_content)

        # Assert
        assert pdf_bytes is not None
        assert len(pdf_bytes) > 1000  # Should have substantial content


class TestPDFGeneration:
    """Test basic PDF generation functionality."""

    def test_generate_simple_pdf(self):
        """Test generating a simple PDF with basic HTML."""
        # Arrange
        service = PDFService()
        html_content = "<html><body><h1>Test Report</h1><p>Content here</p></body></html>"

        # Act
        pdf_bytes = service.generate_pdf_from_html(html_content)

        # Assert
        assert pdf_bytes is not None
        assert len(pdf_bytes) > 0
        assert b"%PDF-" in pdf_bytes
        assert b"%%EOF" in pdf_bytes

    def test_save_pdf_to_file(self, tmp_path):
        """Test saving generated PDF to a file."""
        # Arrange
        service = PDFService()
        html_content = "<html><body><h1>Test</h1></body></html>"
        output_path = tmp_path / "test_report.pdf"

        # Act
        service.save_pdf(html_content, str(output_path))

        # Assert
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_pdf_with_custom_metadata(self):
        """Test PDF generation with custom metadata."""
        # Arrange
        service = PDFService()
        html_content = "<html><body><h1>Metadata Test</h1></body></html>"
        metadata = {
            "title": "음성 증거 분석 보고서",
            "author": "Voice Analysis System",
            "subject": "Legal Evidence Report",
            "keywords": "voice, analysis, evidence",
        }

        # Act
        pdf_bytes = service.generate_pdf_from_html(html_content, metadata=metadata)

        # Assert
        assert pdf_bytes is not None
        # Metadata should be embedded (basic check)
        assert len(pdf_bytes) > 0


class TestPDFErrorHandling:
    """Test error handling in PDF generation."""

    def test_invalid_html_raises_error(self):
        """Test that invalid HTML raises appropriate error."""
        # Arrange
        service = PDFService()
        invalid_html = ""  # Empty string should raise error

        # Act & Assert
        with pytest.raises(ValueError):
            service.generate_pdf_from_html(invalid_html)

    def test_empty_html_raises_error(self):
        """Test that empty HTML raises appropriate error."""
        # Arrange
        service = PDFService()

        # Act & Assert
        with pytest.raises(ValueError):
            service.generate_pdf_from_html("")

    def test_missing_font_directory_fallback(self, tmp_path):
        """Test that missing font directory falls back to system fonts."""
        # Arrange
        non_existent_dir = tmp_path / "non_existent_fonts"
        service = PDFService(font_dir=str(non_existent_dir))
        html_content = "<html><body><h1>Test</h1></body></html>"

        # Act & Assert - Should not crash, may use system fonts
        pdf_bytes = service.generate_pdf_from_html(html_content)
        assert pdf_bytes is not None
