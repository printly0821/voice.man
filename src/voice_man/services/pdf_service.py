"""
PDF generation service using ReportLab.

Provides Korean font support and PDF generation capabilities
for legal evidence reports.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from io import BytesIO

try:
    from reportlab.lib.pagesizes import A4, letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        PageBreak,
        Table,
        TableStyle,
    )
    from reportlab.lib import colors
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
except ImportError:
    raise ImportError("ReportLab is required. Install with: pip install reportlab")


class PDFService:
    """
    PDF generation service with Korean font support.

    Attributes:
        template_dir: Directory containing Jinja2 templates
        font_dir: Directory containing Korean font files
    """

    def __init__(
        self,
        template_dir: Optional[str] = None,
        font_dir: Optional[str] = None,
    ):
        """
        Initialize PDF service.

        Args:
            template_dir: Path to template directory (default: project templates/)
            font_dir: Path to font directory (default: system fonts)
        """
        # Set default directories
        base_dir = Path(__file__).parent.parent.parent
        self.template_dir = Path(template_dir) if template_dir else base_dir / "templates"

        # Font directory - try multiple locations
        if font_dir:
            self.font_dir = Path(font_dir)
        else:
            # Check common font locations
            possible_font_dirs = [
                base_dir / "fonts",
                Path("/System/Library/Fonts/"),  # macOS
                Path("/usr/share/fonts/"),  # Linux
                Path("C:/Windows/Fonts/"),  # Windows
            ]
            self.font_dir = None
            for font_dir in possible_font_dirs:
                if font_dir.exists():
                    self.font_dir = font_dir
                    break

        # Register Korean fonts
        self._register_korean_fonts()

    def _register_korean_fonts(self) -> None:
        """Register Korean fonts for use in PDF generation."""
        font_registered = False

        # Try to find and register Korean fonts
        korean_fonts = [
            ("NanumGothic", "NanumGothic.ttf"),
            ("Malgun Gothic", "malgun.ttf"),
            ("AppleGothic", "AppleGothic.ttf"),
        ]

        if self.font_dir and self.font_dir.exists():
            for font_name, font_file in korean_fonts:
                font_path = self.font_dir / font_file
                if font_path.exists():
                    try:
                        pdfmetrics.registerFont(TTFont(font_name, str(font_path)))
                        font_registered = True
                        break
                    except Exception:
                        continue

        # Fallback: use default font if no Korean font found
        self.korean_font = "NanumGothic" if font_registered else "Helvetica"

    def get_available_fonts(self) -> list[str]:
        """
        Get list of available Korean fonts.

        Returns:
            List of font names that support Korean characters
        """
        korean_fonts = []

        # Common Korean font names
        korean_font_names = [
            "NanumGothic",
            "Nanum Gothic",
            "Malgun Gothic",
            "AppleGothic",
            "Dotum",
            "Batang",
        ]

        # Check if fonts exist in font directory
        if self.font_dir and self.font_dir.exists():
            for font_file in self.font_dir.glob("*.ttf"):
                font_name = font_file.stem
                if any(kf.lower() in font_name.lower() for kf in korean_font_names):
                    korean_fonts.append(font_name)

        # Add system font fallbacks
        korean_fonts.extend(korean_font_names)

        return list(set(korean_fonts))  # Remove duplicates

    def generate_pdf_from_html(
        self,
        html_content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bytes:
        """
        Generate PDF from HTML content (simplified - extracts text).

        Note: This is a simplified implementation that extracts text from HTML.
        For full HTML rendering, use WeasyPrint or implement HTML parser.

        Args:
            html_content: HTML string to convert to PDF
            metadata: Optional PDF metadata (title, author, subject, keywords)

        Returns:
            PDF file as bytes

        Raises:
            ValueError: If html_content is empty
            Exception: If PDF generation fails
        """
        if not html_content or not html_content.strip():
            raise ValueError("HTML content cannot be empty")

        try:
            # Create PDF buffer
            buffer = BytesIO()

            # Create document template
            doc = SimpleDocTemplate(
                buffer,
                pagesize=A4,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=18,
            )

            # Create story (content elements)
            story = []
            styles = getSampleStyleSheet()

            # Add custom style for Korean text
            styles.add(
                ParagraphStyle(
                    name="Korean",
                    fontName=self.korean_font,
                    fontSize=12,
                    leading=16,
                )
            )

            # Simple HTML tag extraction (very basic)
            lines = html_content.replace("<br>", "\n").split("\n")
            in_body = False

            for line in lines:
                line = line.strip()

                # Detect body content
                if "<body" in line:
                    in_body = True
                    continue
                if "</body>" in line:
                    in_body = False
                    continue

                if not in_body:
                    continue

                # Remove HTML tags
                clean_line = line
                for tag in ["<h1>", "</h1>", "<h2>", "</h2>", "<h3>", "</h3>", "<p>", "</p>"]:
                    clean_line = clean_line.replace(tag, "")

                if clean_line:
                    # Determine heading level
                    if "<h1" in line:
                        style = ParagraphStyle(
                            name="Heading1",
                            fontName=self.korean_font,
                            fontSize=24,
                            leading=30,
                            spaceAfter=12,
                        )
                    elif "<h2" in line:
                        style = ParagraphStyle(
                            name="Heading2",
                            fontName=self.korean_font,
                            fontSize=18,
                            leading=24,
                            spaceAfter=10,
                        )
                    else:
                        style = styles["Korean"]

                    story.append(Paragraph(clean_line, style))
                    story.append(Spacer(1, 0.2 * inch))

            # Build PDF
            doc.build(story)

            # Get PDF bytes
            pdf_bytes = buffer.getvalue()
            buffer.close()

            return pdf_bytes

        except Exception as e:
            raise Exception(f"PDF generation failed: {str(e)}")

    def save_pdf(
        self,
        html_content: str,
        output_path: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Generate PDF and save to file.

        Args:
            html_content: HTML string to convert to PDF
            output_path: Path where PDF file will be saved
            metadata: Optional PDF metadata

        Raises:
            ValueError: If html_content is empty
            IOError: If file cannot be written
        """
        pdf_bytes = self.generate_pdf_from_html(html_content, metadata)

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "wb") as f:
            f.write(pdf_bytes)

    def generate_report_pdf(
        self,
        title: str,
        content: list[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bytes:
        """
        Generate a structured report PDF.

        Args:
            title: Report title
            content: List of content sections with 'title' and 'text' keys
            metadata: Optional PDF metadata

        Returns:
            PDF file as bytes
        """
        buffer = BytesIO()

        # Create document template
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18,
        )

        # Create story
        story = []

        # Add title
        title_style = ParagraphStyle(
            name="Title",
            fontName=self.korean_font,
            fontSize=24,
            leading=30,
            alignment=TA_CENTER,
            spaceAfter=30,
        )
        story.append(Paragraph(title, title_style))
        story.append(Spacer(1, 0.3 * inch))

        # Add content sections
        for section in content:
            if "title" in section:
                section_title_style = ParagraphStyle(
                    name="SectionTitle",
                    fontName=self.korean_font,
                    fontSize=16,
                    leading=20,
                    spaceAfter=10,
                )
                story.append(Paragraph(section["title"], section_title_style))

            if "text" in section:
                text_style = ParagraphStyle(
                    name="BodyText",
                    fontName=self.korean_font,
                    fontSize=11,
                    leading=14,
                    spaceAfter=12,
                )
                story.append(Paragraph(section["text"], text_style))
                story.append(Spacer(1, 0.2 * inch))

        # Build PDF
        doc.build(story)

        pdf_bytes = buffer.getvalue()
        buffer.close()

        return pdf_bytes

    def generate_pdf_from_template(
        self,
        template_name: str,
        context: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bytes:
        """
        Generate PDF from Jinja2 template.

        Args:
            template_name: Name of template file
            context: Template context variables
            metadata: Optional PDF metadata

        Returns:
            PDF file as bytes
        """
        from jinja2 import Environment, FileSystemLoader, select_autoescape

        # Setup Jinja2 environment
        env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            autoescape=select_autoescape(["html", "xml"]),
        )

        # Load and render template
        template = env.get_template(template_name)
        html_content = template.render(**context)

        # Generate PDF
        return self.generate_pdf_from_html(html_content, metadata)
