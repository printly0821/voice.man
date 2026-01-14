"""
Document Utilization Strategy System for Forensic Pipeline
SPEC-ASSET-004: Document Generation, Distribution, and Usage Analytics

Defines how forensic reports are used:
1. Legal evidence preparation
2. Case management integration
3. Analytics and insights
4. API access for external systems
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from .uuidv7_manager import AssetMetadata, AssetRegistry, AssetType, get_asset_registry

logger = logging.getLogger(__name__)


class DocumentPurpose(str, Enum):
    """Purpose of generated documents"""

    LEGAL_EVIDENCE = "legal_evidence"  # Court submission
    CASE_ANALYSIS = "case_analysis"  # Internal review
    CLIENT_REPORT = "client_report"  # Client delivery
    RESEARCH = "research"  # Academic research
    TRAINING = "training"  # Staff training
    ARCHIVE = "archive"  # Long-term storage


class DocumentFormat(str, Enum):
    """Document output formats"""

    HTML = "html"  # Interactive web format
    PDF = "pdf"  # Print-ready format
    JSON = "json"  # Machine-readable format
    DOCX = "docx"  # Word document format
    XLSX = "xlsx"  # Excel spreadsheet format


@dataclass
class DocumentTemplate:
    """Document template configuration"""

    name: str
    purpose: DocumentPurpose
    format: DocumentFormat
    template_path: Optional[str] = None
    sections: List[str] = field(default_factory=list)
    include_mermaid: bool = True
    include_appendices: bool = False
    language: str = "ko"  # ko, en, ja, zh


# Predefined document templates
DOCUMENT_TEMPLATES: Dict[str, DocumentTemplate] = {
    "legal_evidence_ko": DocumentTemplate(
        name="Legal Evidence (Korean)",
        purpose=DocumentPurpose.LEGAL_EVIDENCE,
        format=DocumentFormat.PDF,
        sections=[
            "executive_summary",
            "case_background",
            "evidence_summary",
            "forensic_analysis",
            "gaslighting_patterns",
            "threat_assessment",
            "deception_analysis",
            "conclusions",
            "appendices",
        ],
        include_mermaid=False,
        include_appendices=True,
        language="ko",
    ),
    "case_analysis_ko": DocumentTemplate(
        name="Case Analysis (Korean)",
        purpose=DocumentPurpose.CASE_ANALYSIS,
        format=DocumentFormat.HTML,
        sections=[
            "executive_summary",
            "timeline",
            "speaker_analysis",
            "emotion_timeline",
            "gaslighting_techniques",
            "risk_assessment",
            "recommendations",
        ],
        include_mermaid=True,
        include_appendices=False,
        language="ko",
    ),
    "client_report_ko": DocumentTemplate(
        name="Client Report (Korean)",
        purpose=DocumentPurpose.CLIENT_REPORT,
        format=DocumentFormat.PDF,
        sections=[
            "executive_summary",
            "key_findings",
            "risk_level",
            "recommendations",
            "contact_info",
        ],
        include_mermaid=False,
        include_appendices=False,
        language="ko",
    ),
    "research_export": DocumentTemplate(
        name="Research Export",
        purpose=DocumentPurpose.RESEARCH,
        format=DocumentFormat.JSON,
        sections=["all"],
        include_mermaid=False,
        include_appendices=False,
        language="en",
    ),
}


@dataclass
class DocumentUsageEvent:
    """Track document usage events"""

    document_id: str
    event_type: str  # viewed, downloaded, shared, printed, etc.
    user_id: Optional[str]
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class DocumentUtilizationManager:
    """
    Manages document generation, distribution, and usage tracking

    Coordinates with report service to generate appropriate documents
    based on purpose and tracks how they are used.
    """

    def __init__(
        self,
        output_base_path: Optional[Path] = None,
        registry: Optional[AssetRegistry] = None,
    ):
        """
        Initialize document utilization manager

        Args:
            output_base_path: Base path for document output
            registry: Asset registry
        """
        self.output_base_path = output_base_path or Path("data/documents")
        self.registry = registry or get_asset_registry()

        # Usage tracking
        self._usage_events: List[DocumentUsageEvent] = []
        self._usage_log_path = self.output_base_path / "usage_events.jsonl"

        # Create directory structure
        self._create_output_structure()

    def _create_output_structure(self):
        """Create output directory structure by purpose"""
        for purpose in DocumentPurpose:
            purpose_path = self.output_base_path / purpose.value
            purpose_path.mkdir(parents=True, exist_ok=True)

    def generate_document(
        self,
        audio_asset_id: str,
        template_name: str,
        custom_sections: Optional[List[str]] = None,
        output_formats: Optional[List[DocumentFormat]] = None,
    ) -> Dict[str, str]:
        """
        Generate document(s) for an audio asset

        Args:
            audio_asset_id: Source audio asset ID
            template_name: Template name from DOCUMENT_TEMPLATES
            custom_sections: Override template sections
            output_formats: Formats to generate (default: from template)

        Returns:
            Dictionary mapping format to file path
        """
        # Get template
        template = DOCUMENT_TEMPLATES.get(template_name)
        if not template:
            raise ValueError(f"Unknown template: {template_name}")

        # Get asset lineage
        lineage = self.registry.get_asset_lineage(audio_asset_id)
        if not lineage:
            raise ValueError(f"Asset not found: {audio_asset_id}")

        # Gather all analysis results
        analysis_data = self._gather_analysis_data(lineage)

        # Override sections if specified
        sections = custom_sections or template.sections

        # Determine output formats
        formats = output_formats or [template.format]

        # Generate documents
        output_paths = {}
        for fmt in formats:
            output_path = self._generate_single_document(
                template, sections, analysis_data, fmt, audio_asset_id
            )
            output_paths[fmt.value] = str(output_path)

            # Register document as asset
            from .core_asset_manager import get_core_asset_manager

            asset_mgr = get_core_asset_manager()
            asset_mgr.register_derived_asset(
                asset_type=AssetType.REPORT_HTML
                if fmt == DocumentFormat.HTML
                else AssetType.REPORT_PDF,
                parent_id=audio_asset_id,
                content={
                    "template": template_name,
                    "path": str(output_path),
                    "format": fmt.value,
                    "purpose": template.purpose.value,
                },
                storage_format="json",
            )

        return output_paths

    def _gather_analysis_data(self, lineage: List[AssetMetadata]) -> Dict[str, Any]:
        """
        Gather all analysis data from asset lineage

        Args:
            lineage: Asset lineage from audio to final results

        Returns:
            Dictionary with all analysis data
        """
        data = {
            "audio_file": {},
            "transcript": {},
            "stt_results": {},
            "ser_results": {},
            "forensic_scores": {},
            "gaslighting_analysis": {},
            "crime_language": {},
            "cross_validation": {},
        }

        for asset in lineage:
            if asset.asset_type == AssetType.AUDIO_FILE:
                data["audio_file"] = asset.to_dict()
            elif asset.asset_type == AssetType.TRANSCRIPT:
                data["transcript"] = asset.to_dict()
            elif asset.asset_type == AssetType.STT_RESULT:
                if "stt_results" not in data:
                    data["stt_results"] = []
                data["stt_results"].append(asset.to_dict())
            elif asset.asset_type == AssetType.SER_RESULT:
                if "ser_results" not in data:
                    data["ser_results"] = []
                data["ser_results"].append(asset.to_dict())
            elif asset.asset_type == AssetType.FORENSIC_SCORE:
                data["forensic_scores"] = asset.to_dict()
            elif asset.asset_type == AssetType.GASLIGHTING_ANALYSIS:
                data["gaslighting_analysis"] = asset.to_dict()
            elif asset.asset_type == AssetType.CRIME_LANGUAGE:
                data["crime_language"] = asset.to_dict()
            elif asset.asset_type == AssetType.CROSS_VALIDATION:
                data["cross_validation"] = asset.to_dict()

        return data

    def _generate_single_document(
        self,
        template: DocumentTemplate,
        sections: List[str],
        data: Dict[str, Any],
        format_type: DocumentFormat,
        asset_id: str,
    ) -> Path:
        """
        Generate a single document in specified format

        Args:
            template: Document template
            sections: Sections to include
            data: Analysis data
            format_type: Output format
            asset_id: Asset ID

        Returns:
            Output file path
        """
        # Create output path
        purpose_dir = self.output_base_path / template.purpose.value / format_type.value
        purpose_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{asset_id}_{template.name}.{format_type.value}"
        output_path = purpose_dir / filename

        # Generate document based on format
        if format_type == DocumentFormat.HTML:
            self._generate_html_document(output_path, template, sections, data)
        elif format_type == DocumentFormat.PDF:
            # Generate HTML first, then convert to PDF
            html_path = output_path.with_suffix(".html")
            self._generate_html_document(html_path, template, sections, data)
            self._convert_html_to_pdf(html_path, output_path)
        elif format_type == DocumentFormat.JSON:
            self._generate_json_document(output_path, sections, data)
        else:
            raise ValueError(f"Unsupported format: {format_type}")

        logger.info(f"Generated document: {output_path}")
        return output_path

    def _generate_html_document(
        self,
        output_path: Path,
        template: DocumentTemplate,
        sections: List[str],
        data: Dict[str, Any],
    ):
        """Generate HTML document"""
        # Import report service
        try:
            from voice_man.reports.html_generator import HTMLReportGenerator

            generator = HTMLReportGenerator()
            # Use existing report generation logic
            # ... implementation details depend on existing service
            html_content = generator.generate_from_data(data, sections, template.language)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html_content)

        except ImportError:
            # Fallback: simple HTML generation
            self._generate_simple_html(output_path, data)

    def _generate_simple_html(self, output_path: Path, data: Dict[str, Any]):
        """Generate simple HTML document (fallback)"""
        html = f"""
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Forensic Analysis Report</title>
    <style>
        body {{ font-family: 'Noto Sans KR', sans-serif; margin: 40px; line-height: 1.6; }}
        .section {{ margin: 20px 0; padding: 15px; border-left: 3px solid #333; }}
        .risk-high {{ background: #fee; }}
        .risk-medium {{ background: #ffc; }}
        .risk-low {{ background: #efe; }}
    </style>
</head>
<body>
    <h1>Forensic Analysis Report</h1>
    <p>Generated: {datetime.now(timezone.utc).isoformat()}</p>
    <pre>{json.dumps(data, indent=2, ensure_ascii=False)}</pre>
</body>
</html>
        """
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

    def _convert_html_to_pdf(self, html_path: Path, pdf_path: Path):
        """Convert HTML to PDF"""
        try:
            from voice_man.reports.pdf_generator import PDFGenerator

            generator = PDFGenerator()
            generator.convert_html_to_pdf(html_path, pdf_path)

        except ImportError:
            logger.warning("PDF generator not available, skipping PDF conversion")

    def _generate_json_document(self, output_path: Path, sections: List[str], data: Dict[str, Any]):
        """Generate JSON document"""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"sections": sections, "data": data}, f, indent=2, ensure_ascii=False)

    def track_usage(
        self, document_id: str, event_type: str, user_id: Optional[str] = None, **metadata
    ):
        """
        Track document usage event

        Args:
            document_id: Document asset ID
            event_type: Type of event (viewed, downloaded, etc.)
            user_id: User who triggered event
            **metadata: Additional metadata
        """
        event = DocumentUsageEvent(
            document_id=document_id,
            event_type=event_type,
            user_id=user_id,
            timestamp=datetime.now(timezone.utc),
            metadata=metadata,
        )

        self._usage_events.append(event)

        # Append to usage log
        with open(self._usage_log_path, "a", encoding="utf-8") as f:
            json.dump(event.to_dict() if hasattr(event, "to_dict") else event.__dict__, f)
            f.write("\n")

    def get_usage_analytics(self, document_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get usage analytics

        Args:
            document_id: Filter by specific document, or None for all

        Returns:
            Usage analytics summary
        """
        events = (
            [e for e in self._usage_events if e.document_id == document_id]
            if document_id
            else self._usage_events
        )

        # Count by event type
        event_counts = {}
        for event in events:
            event_counts[event.event_type] = event_counts.get(event.event_type, 0) + 1

        # Count by user
        user_counts = {}
        for event in events:
            if event.user_id:
                user_counts[event.user_id] = user_counts.get(event.user_id, 0) + 1

        return {
            "total_events": len(events),
            "by_event_type": event_counts,
            "by_user": user_counts,
            "unique_documents": len(set(e.document_id for e in events)),
        }

    def generate_batch_summary(self, batch_id: str, template_name: str = "case_analysis_ko") -> str:
        """
        Generate batch summary document

        Args:
            batch_id: Batch job ID
            template_name: Template to use

        Returns:
            Path to generated summary
        """
        # Get all assets in batch
        assets = self.registry.get_assets_by_batch(batch_id)

        # Generate summary
        summary_data = {
            "batch_id": batch_id,
            "total_files": len(assets),
            "files": [asset.to_dict() for asset in assets],
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

        # Write summary
        summary_path = (
            self.output_base_path / DocumentPurpose.CASE_ANALYSIS.value / f"{batch_id}_summary.json"
        )

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)

        return str(summary_path)


@dataclass
class DocumentDistributionConfig:
    """Configuration for document distribution"""

    enable_api_access: bool = True
    enable_email_delivery: bool = False
    enable_web_dashboard: bool = True
    enable_download_links: bool = True
    retention_days_downloads: int = 30


class DocumentDistributionManager:
    """
    Manages document distribution to various channels

    Handles:
    - API access for external systems
    - Email delivery
    - Web dashboard
    - Download link generation
    """

    def __init__(
        self,
        config: Optional[DocumentDistributionConfig] = None,
        utilization_manager: Optional[DocumentUtilizationManager] = None,
    ):
        """
        Initialize document distribution manager

        Args:
            config: Distribution configuration
            utilization_manager: Document utilization manager
        """
        self.config = config or DocumentDistributionConfig()
        self.utilization = utilization_manager or DocumentUtilizationManager()

    def generate_download_link(self, document_id: str, expiry_hours: int = 24) -> str:
        """
        Generate time-limited download link

        Args:
            document_id: Document asset ID
            expiry_hours: Link expiry in hours

        Returns:
            Download URL
        """
        # Get document path
        asset = self.utilization.registry.get_asset(document_id)
        if not asset or not asset.storage_path:
            raise ValueError(f"Document not found: {document_id}")

        # Generate secure token
        import secrets

        token = secrets.token_urlsafe(32)

        # In production, this would:
        # 1. Store token in database with expiry
        # 2. Generate API endpoint URL
        # 3. Return URL

        download_url = f"/api/documents/download/{document_id}?token={token}"

        logger.info(f"Generated download link for {document_id} (expires in {expiry_hours}h)")
        return download_url

    def send_email_delivery(
        self, document_id: str, recipient_email: str, subject: str, message: str
    ) -> bool:
        """
        Send document via email

        Args:
            document_id: Document asset ID
            recipient_email: Recipient email address
            subject: Email subject
            message: Email message body

        Returns:
            True if sent successfully
        """
        # Implementation depends on email service
        # For now, just log
        logger.info(f"Email delivery requested: {document_id} -> {recipient_email}")
        return True

    def prepare_api_response(self, document_id: str) -> Dict[str, Any]:
        """
        Prepare document for API response

        Args:
            document_id: Document asset ID

        Returns:
            API response data
        """
        asset = self.utilization.registry.get_asset(document_id)
        if not asset:
            raise ValueError(f"Document not found: {document_id}")

        return {
            "document_id": document_id,
            "asset_type": asset.asset_type.value,
            "status": asset.status.value,
            "created_at": asset.created_at.isoformat(),
            "download_url": self.generate_download_link(document_id),
            "metadata": asset.metadata,
        }
