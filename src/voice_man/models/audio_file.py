"""
Audio File Model

Represents an uploaded audio file and its analysis results.
"""

from enum import Enum
from typing import Any


class ReportStatus(str, Enum):
    """Report generation status."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class AudioFile:
    """
    Audio file model with analysis results.

    Attributes:
        id: Unique identifier
        filename: Original filename
        transcript: Transcribed text
        report_status: Report generation status
        report_version: Current report version
    """

    def __init__(
        self,
        id: int,
        filename: str,
        transcript: str | None = None,
        report_status: ReportStatus | None = None,
        report_version: int = 0,
    ):
        self.id = id
        self.filename = filename
        self.transcript = transcript
        self.report_status = report_status
        self.report_version = report_version

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "filename": self.filename,
            "transcript": self.transcript,
            "report_status": self.report_status.value if self.report_status else None,
            "report_version": self.report_version,
        }
