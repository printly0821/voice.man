"""
Forensic Report Generation Modules

Provides HTML and PDF report generation for forensic analysis results.
"""

from voice_man.reports.html_generator import ForensicHTMLGenerator
from voice_man.reports.pdf_generator import ForensicPDFGenerator

__all__ = [
    "ForensicHTMLGenerator",
    "ForensicPDFGenerator",
]
