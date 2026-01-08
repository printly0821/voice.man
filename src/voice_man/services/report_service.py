"""
Report Generation Service

Orchestrates the generation of legal evidence reports including:
- PDF generation
- Chart creation
- Template rendering
- Version management
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from voice_man.services.pdf_service import PDFService
from voice_man.services.report_template_service import ReportTemplateService
from voice_man.services.chart_service import ChartService
from voice_man.models.audio_file import AudioFile, ReportStatus

logger = logging.getLogger(__name__)


class ReportService:
    """
    Service for managing report generation workflow.

    Handles asynchronous report generation with status tracking
    and version management.
    """

    def __init__(
        self,
        pdf_service: PDFService | None = None,
        template_service: ReportTemplateService | None = None,
        chart_service: ChartService | None = None,
    ):
        """
        Initialize ReportService.

        Args:
            pdf_service: PDF generation service
            template_service: Template rendering service
            chart_service: Chart generation service
        """
        self.pdf_service = pdf_service or PDFService()
        self.template_service = template_service or ReportTemplateService()
        self.chart_service = chart_service or ChartService()

        # In-memory storage for report status (in production, use database)
        self._report_status: dict[int, dict[str, Any]] = {}
        self._report_tasks: dict[str, asyncio.Task] = {}

    async def start_report_generation(self, audio_id: int, force: bool = False) -> dict[str, Any]:
        """
        Start asynchronous report generation.

        Args:
            audio_id: Audio file ID
            force: Force regeneration even if report exists

        Returns:
            dict with report_id, status, estimated_time_seconds

        Raises:
            ValueError: If audio not found, transcript missing, or report exists
        """
        # Check if audio file exists (mock)
        # In production: audio_file = await db.get_audio_file(audio_id)
        audio_file = await self._get_audio_file(audio_id)
        if not audio_file:
            raise ValueError(f"Audio file {audio_id} not found")

        # Check if transcript exists
        if not audio_file.get("transcript"):
            raise ValueError("Transcript required for report generation")

        # Check if report already exists
        if not force and audio_file.get("report_status") == ReportStatus.COMPLETED:
            raise ValueError("Report already exists. Use force=True to regenerate")

        # Generate report ID
        report_id = str(uuid4())

        # Initialize status
        self._report_status[audio_id] = {
            "report_id": report_id,
            "status": "processing",
            "progress_percentage": 0,
            "current_step": "initializing",
            "created_at": datetime.now().isoformat(),
            "completed_at": None,
            "file_path": None,
            "version": audio_file.get("report_version", 0) + 1,
        }

        # Start async generation task
        task = asyncio.create_task(self._generate_report(audio_id, report_id))
        self._report_tasks[report_id] = task

        return {
            "report_id": report_id,
            "status": "processing",
            "estimated_time_seconds": 30,  # Estimate
        }

    async def get_report_status(self, audio_id: int) -> dict[str, Any]:
        """
        Get current report generation status.

        Args:
            audio_id: Audio file ID

        Returns:
            dict with status information

        Raises:
            ValueError: If report not found
        """
        if audio_id not in self._report_status:
            raise ValueError(f"Report for audio {audio_id} not found")

        return self._report_status[audio_id]

    async def get_report_file_path(self, audio_id: int) -> str:
        """
        Get path to generated PDF report.

        Args:
            audio_id: Audio file ID

        Returns:
            str: Path to PDF file

        Raises:
            ValueError: If report not found or not ready
        """
        status_info = await self.get_report_status(audio_id)

        if status_info["status"] != "completed":
            raise ValueError("Report is not ready yet")

        if not status_info["file_path"]:
            raise ValueError("Report file path not available")

        return status_info["file_path"]

    async def get_report_versions(self, audio_id: int) -> list[dict[str, Any]]:
        """
        Get all report versions for an audio file.

        Args:
            audio_id: Audio file ID

        Returns:
            list of version info dictionaries

        Raises:
            ValueError: If audio not found
        """
        # Mock implementation
        # In production: query database for all versions
        if audio_id not in self._report_status:
            raise ValueError(f"Audio file {audio_id} not found")

        current_status = self._report_status[audio_id]
        return [
            {
                "version": current_status["version"],
                "created_at": current_status["created_at"],
                "file_path": current_status["file_path"] or "pending",
            }
        ]

    async def _generate_report(self, audio_id: int, report_id: str) -> None:
        """
        Internal method to generate report asynchronously.

        Args:
            audio_id: Audio file ID
            report_id: Report generation ID
        """
        try:
            # Update status
            self._update_status(audio_id, 10, "generating_charts")

            # Step 1: Generate charts
            await self._generate_charts(audio_id)

            # Update status
            self._update_status(audio_id, 40, "rendering_template")

            # Step 2: Render HTML template
            html_content = await self._render_template(audio_id)

            # Update status
            self._update_status(audio_id, 70, "creating_pdf")

            # Step 3: Generate PDF
            pdf_path = await self._create_pdf(audio_id, html_content)

            # Update status
            self._update_status(audio_id, 100, "completed", file_path=pdf_path)

            logger.info(f"Report generation completed for audio {audio_id}")

        except Exception as e:
            logger.error(f"Report generation failed for audio {audio_id}: {e}")
            self._update_status(audio_id, 0, "failed")
            raise

    async def _generate_charts(self, audio_id: int) -> None:
        """Generate all required charts for the report."""
        # Mock chart generation
        # In production: use actual analysis data
        chart_dir = Path("charts") / str(audio_id)
        chart_dir.mkdir(parents=True, exist_ok=True)

        # Generate mock charts
        timeline_data = [
            {"speaker": "A", "timestamp": 0.0, "duration": 5.0, "emotion": "neutral"},
            {"speaker": "B", "timestamp": 5.0, "duration": 3.0, "emotion": "angry"},
        ]

        timeline_fig = await self.chart_service.generate_timeline_chart(timeline_data)
        await self.chart_service.save_chart(timeline_fig, str(chart_dir / "timeline.png"))

    async def _render_template(self, audio_id: int) -> str:
        """Render HTML template with audio file data."""
        # Mock template rendering
        # In production: use actual audio file and analysis data
        audio_file = await self._get_audio_file(audio_id)

        # Prepare context from audio file
        context = self.template_service.prepare_context_from_audio_file(audio_file)

        # Add chart paths
        context["analysis_results"] = {
            "timeline_chart": "charts/timeline.png",
            "emotion_chart": "charts/emotion.png",
            "crime_chart": "charts/crime.png",
            "gaslighting_chart": "charts/gaslighting.png",
        }

        # Render template
        html_content = await self.template_service.render_template("complete_report.html", context)

        return html_content

    async def _create_pdf(self, audio_id: int, html_content: str) -> str:
        """Create PDF from HTML content."""
        output_dir = Path("reports")
        output_dir.mkdir(parents=True, exist_ok=True)

        pdf_path = output_dir / f"report_{audio_id}.pdf"

        await self.pdf_service.generate_pdf_from_html(html_content, str(pdf_path))

        return str(pdf_path)

    def _update_status(
        self,
        audio_id: int,
        progress_percentage: int,
        current_step: str,
        file_path: str | None = None,
    ) -> None:
        """Update report generation status."""
        if audio_id in self._report_status:
            self._report_status[audio_id]["progress_percentage"] = progress_percentage
            self._report_status[audio_id]["current_step"] = current_step

            if current_step == "completed":
                self._report_status[audio_id]["status"] = "completed"
                self._report_status[audio_id]["completed_at"] = datetime.now().isoformat()
                if file_path:
                    self._report_status[audio_id]["file_path"] = file_path

            elif current_step == "failed":
                self._report_status[audio_id]["status"] = "failed"

    async def _get_audio_file(self, audio_id: int) -> dict[str, Any] | None:
        """
        Get audio file from database (mock).

        In production, this would query the database.
        """
        # Mock implementation
        return {
            "id": audio_id,
            "filename": "test_audio.wav",
            "transcript": "Mock transcript content",
            "report_status": None,
            "report_version": 0,
        }
