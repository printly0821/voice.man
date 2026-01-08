"""
Report template service using Jinja2.

Provides template rendering for legal evidence reports
with Korean language support.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

try:
    from jinja2 import Environment, FileSystemLoader, select_autoescape
except ImportError:
    raise ImportError("Jinja2 is required. Install with: pip install jinja2")


class ReportTemplateService:
    """
    Report template service for legal evidence reports.

    Attributes:
        template_dir: Directory containing Jinja2 templates
        env: Jinja2 environment for template rendering
    """

    # Template file names
    TEMPLATE_COVER_PAGE = "cover_page.html"
    TEMPLATE_TABLE_OF_CONTENTS = "table_of_contents.html"
    TEMPLATE_BODY_SECTION = "body_section.html"
    TEMPLATE_APPENDIX = "appendix.html"

    def __init__(self, template_dir: Optional[str] = None):
        """
        Initialize report template service.

        Args:
            template_dir: Path to template directory (default: project templates/)
        """
        # Set default template directory
        if template_dir:
            self.template_dir = Path(template_dir)
        else:
            # Try multiple possible locations
            possible_locations = [
                Path(__file__).parent.parent.parent / "templates",  # project root
                Path(__file__).parent.parent / "templates",  # src directory
                Path("templates"),  # current directory
            ]
            for location in possible_locations:
                if location.exists():
                    self.template_dir = location
                    break
            else:
                # Default to project root / templates
                self.template_dir = Path(__file__).parent.parent.parent / "templates"

        # Setup Jinja2 environment
        self.env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            autoescape=select_autoescape(["html", "xml"]),
        )

    def template_exists(self, template_name: str) -> bool:
        """
        Check if template exists.

        Args:
            template_name: Name of template file

        Returns:
            True if template exists, False otherwise
        """
        try:
            self.env.get_template(template_name)
            return True
        except Exception:
            return False

    def render_template(
        self,
        template_name: str,
        context: Dict[str, Any],
    ) -> str:
        """
        Render template with context data.

        Args:
            template_name: Name of template file
            context: Template context variables

        Returns:
            Rendered HTML string

        Raises:
            Exception: If template rendering fails
        """
        try:
            template = self.env.get_template(template_name)
            return template.render(**context)
        except Exception as e:
            raise Exception(f"Template rendering failed: {str(e)}")

    def prepare_context_from_audio_file(self, audio_file: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare template context from AudioFile model.

        Args:
            audio_file: Audio file data dictionary

        Returns:
            Template context dictionary
        """
        # Format upload timestamp
        upload_time = audio_file.get("upload_timestamp", "")
        if isinstance(upload_time, str):
            try:
                dt = datetime.fromisoformat(upload_time.replace("Z", "+00:00"))
                formatted_date = dt.strftime("%Y년 %m월 %d일")
            except Exception:
                formatted_date = upload_time
        else:
            formatted_date = str(upload_time)

        # Format duration
        duration_seconds = audio_file.get("duration_seconds", 0)
        duration_minutes = duration_seconds // 60
        duration_remaining = duration_seconds % 60
        formatted_duration = f"{duration_minutes}분 {duration_remaining}초"

        return {
            "audio_id": audio_file.get("id", ""),
            "original_filename": audio_file.get("original_filename", ""),
            "duration": formatted_duration,
            "upload_date": formatted_date,
            "file_hash": audio_file.get("file_hash", "")[:16] + "...",  # Truncate for display
        }

    def prepare_context_from_transcript(self, transcript: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare template context from Transcript model.

        Args:
            transcript: Transcript data dictionary

        Returns:
            Template context dictionary
        """
        segments = transcript.get("segments", [])

        # Format segments for display
        formatted_segments = []
        for segment in segments:
            start_time = segment.get("start_time", 0)

            # Format timestamp
            minutes = int(start_time // 60)
            seconds = int(start_time % 60)
            timestamp = f"{minutes:02d}:{seconds:02d}"

            formatted_segments.append(
                {
                    "timestamp": timestamp,
                    "speaker": segment.get("speaker_id", "Unknown"),
                    "text": segment.get("text", ""),
                    "confidence": segment.get("confidence", 0.0),
                }
            )

        return {
            "transcript_id": transcript.get("id", ""),
            "audio_id": transcript.get("audio_id", ""),
            "segments": formatted_segments,
            "segment_count": len(formatted_segments),
        }

    def generate_complete_report(self, report_data: Dict[str, Any]) -> str:
        """
        Generate complete report HTML with all sections.

        Args:
            report_data: Report data including title, sections, etc.

        Returns:
            Complete report HTML string
        """
        # Render cover page
        cover_html = self.render_template(self.TEMPLATE_COVER_PAGE, report_data)

        # Render table of contents
        toc_html = self.render_template(self.TEMPLATE_TABLE_OF_CONTENTS, report_data)

        # Render body sections
        body_sections_html = ""
        if "sections" in report_data:
            for section in report_data["sections"]:
                section_type = section.get("type", "body_section")
                section_data = section.get("data", {})
                section_html = self.render_template(f"{section_type}.html", section_data)
                body_sections_html += section_html

        # Render appendix
        appendix_html = self.render_template(self.TEMPLATE_APPENDIX, report_data)

        # Combine all sections
        complete_html = f"""
        <!DOCTYPE html>
        <html lang="ko">
        <head>
            <meta charset="UTF-8">
            <title>{report_data.get("report_title", "보고서")}</title>
            <style>
                body {{
                    font-family: 'NanumGothic', 'Malgun Gothic', sans-serif;
                    margin: 0;
                    padding: 0;
                }}
                .page-break {{
                    page-break-after: always;
                }}
            </style>
        </head>
        <body>
            {cover_html}
            <div class="page-break"></div>
            {toc_html}
            <div class="page-break"></div>
            {body_sections_html}
            <div class="page-break"></div>
            {appendix_html}
        </body>
        </html>
        """

        return complete_html

    def generate_cover_page(
        self,
        report_title: str,
        report_number: str,
        report_date: str,
        analyst_name: str,
        case_number: str,
    ) -> str:
        """
        Generate cover page HTML.

        Args:
            report_title: Title of the report
            report_number: Report identifier
            report_date: Report creation date
            analyst_name: Name of analyst
            case_number: Case identifier

        Returns:
            Cover page HTML string
        """
        context = {
            "report_title": report_title,
            "report_number": report_number,
            "report_date": report_date,
            "analyst_name": analyst_name,
            "case_number": case_number,
        }

        return self.render_template(self.TEMPLATE_COVER_PAGE, context)

    def generate_body_section(
        self,
        section_title: str,
        crime_statements: List[Dict[str, Any]],
        summary_stats: Optional[Dict[str, int]] = None,
    ) -> str:
        """
        Generate body section HTML.

        Args:
            section_title: Section heading
            crime_statements: List of crime statement dictionaries
            summary_stats: Optional summary statistics

        Returns:
            Body section HTML string
        """
        context = {
            "section_title": section_title,
            "crime_statements": crime_statements,
        }

        if summary_stats:
            context.update(summary_stats)

        return self.render_template(self.TEMPLATE_BODY_SECTION, context)

    def generate_appendix(
        self,
        legal_references: List[Dict[str, str]],
    ) -> str:
        """
        Generate appendix HTML.

        Args:
            legal_references: List of legal reference dictionaries

        Returns:
            Appendix HTML string
        """
        context = {
            "legal_references": legal_references,
        }

        return self.render_template(self.TEMPLATE_APPENDIX, context)
