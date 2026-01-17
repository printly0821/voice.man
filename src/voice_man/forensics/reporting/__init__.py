"""
Legal report generation modules.

Implements:
- Title page generation
- Executive summary
- Methodology documentation
- Reproduction guide
- Expert testimony support
"""

from voice_man.forensics.reporting.legal_report_generator import LegalReportGenerator, generate_legal_report

__all__ = [
    "LegalReportGenerator",
    "generate_legal_report",
]
