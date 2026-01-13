"""
Forensic PDF Report Generator

Converts HTML forensic reports to PDF with proper Korean font support.
Uses Playwright for accurate HTML-to-PDF conversion with JavaScript rendering
(essential for Mermaid diagrams).
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from voice_man.reports.html_generator import ForensicHTMLGenerator

logger = logging.getLogger(__name__)


class ForensicPDFGenerator:
    """
    Generate PDF reports from forensic analysis data.

    Features:
    - Korean font support via Playwright
    - Mermaid diagram rendering (JavaScript-dependent)
    - Professional layout for legal documentation
    - Batch processing support
    """

    def __init__(
        self,
        html_generator: Optional[ForensicHTMLGenerator] = None,
        output_dir: Optional[Path] = None,
    ):
        """
        Initialize the PDF generator.

        Args:
            html_generator: HTML generator instance (creates default if None)
            output_dir: Default output directory for PDFs
        """
        self.html_generator = html_generator or ForensicHTMLGenerator()
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()

        # Playwright browser instance (lazy loaded)
        self._playwright = None
        self._browser = None

    async def generate_from_json(
        self,
        json_path: str,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Generate PDF report from forensic analysis JSON file.

        Args:
            json_path: Path to forensic analysis JSON file
            output_path: Optional output PDF file path

        Returns:
            Path to generated PDF file

        Raises:
            FileNotFoundError: If JSON file not found
            RuntimeError: If PDF generation fails
        """
        json_file = Path(json_path)
        if not json_file.exists():
            raise FileNotFoundError(f"Forensic JSON file not found: {json_path}")

        # Read JSON data
        import json

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        return await self.generate(data, output_path or str(json_file.with_suffix(".pdf")))

    async def generate(
        self,
        data: Dict[str, Any],
        output_path: Optional[str] = None,
    ) -> str:
        """
        Generate PDF report from forensic analysis data.

        Args:
            data: Forensic analysis data dictionary
            output_path: Optional output PDF file path

        Returns:
            Path to generated PDF file

        Raises:
            RuntimeError: If PDF generation fails
        """
        # Generate HTML first
        html_content = self.html_generator.generate(data)

        # Determine output path
        if output_path:
            pdf_path = Path(output_path)
        else:
            analysis_id = data.get("analysis_id", "unknown")
            pdf_path = self.output_dir / f"forensic_report_{analysis_id}.pdf"

        pdf_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert HTML to PDF
        await self._html_to_pdf(html_content, str(pdf_path))

        return str(pdf_path)

    async def _html_to_pdf(self, html_content: str, pdf_path: str) -> None:
        """
        Convert HTML content to PDF using Playwright.

        Args:
            html_content: HTML string to convert
            pdf_path: Output PDF file path

        Raises:
            ImportError: If Playwright is not installed
            RuntimeError: If PDF generation fails
        """
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            raise ImportError(
                "Playwright is required for PDF generation. "
                "Install with: uv add playwright && uv run playwright install chromium"
            )

        try:
            # Initialize Playwright
            async with async_playwright() as playwright:
                browser = await playwright.chromium.launch()

                # Create page
                page = await browser.new_page()

                # Set content and wait for Mermaid rendering
                await page.set_content(html_content, wait_until="networkidle")

                # Wait for Mermaid diagrams to render
                try:
                    await page.wait_for_selector(".mermaid > svg", timeout=5000)
                except Exception:
                    # Mermaid might not be present or already rendered
                    pass

                # Generate PDF
                await page.pdf(
                    path=str(pdf_path),
                    format="A4",
                    print_background=True,
                    margin={
                        "top": "0.5cm",
                        "right": "0.5cm",
                        "bottom": "0.5cm",
                        "left": "0.5cm",
                    },
                )

                # Close page
                await page.close()
                await browser.close()

        except Exception as e:
            raise RuntimeError(f"PDF generation failed: {e}")

    async def generate_batch(
        self,
        json_paths: List[str],
        output_dir: Optional[str] = None,
    ) -> List[str]:
        """
        Generate PDF reports from multiple JSON files.

        Args:
            json_paths: List of paths to forensic JSON files
            output_dir: Optional output directory for PDFs

        Returns:
            List of paths to generated PDF files
        """
        if output_dir:
            out_dir = Path(output_dir)
        else:
            out_dir = self.output_dir

        out_dir.mkdir(parents=True, exist_ok=True)

        pdf_paths = []
        for json_path in json_paths:
            try:
                json_file = Path(json_path)
                pdf_path = out_dir / json_file.with_suffix(".pdf").name
                result = await self.generate_from_json(str(json_file), str(pdf_path))
                pdf_paths.append(result)
            except Exception as e:
                logger.warning("Failed to generate PDF for %s: %s", json_path, e)

        return pdf_paths

    async def close(self) -> None:
        """Close browser instance and cleanup resources."""
        if self._browser:
            await self._browser.close()
            self._browser = None
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


class SimplePDFGenerator:
    """
    Fallback PDF generator using WeasyPrint when Playwright is unavailable.

    Note: WeasyPrint does NOT render JavaScript (Mermaid diagrams).
    Use this only when diagram rendering is not required.
    """

    def __init__(
        self,
        html_generator: Optional[ForensicHTMLGenerator] = None,
        output_dir: Optional[Path] = None,
    ):
        """
        Initialize the simple PDF generator.

        Args:
            html_generator: HTML generator instance
            output_dir: Default output directory for PDFs
        """
        self.html_generator = html_generator or ForensicHTMLGenerator()
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()

    def generate_from_json(
        self,
        json_path: str,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Generate PDF report from forensic analysis JSON file.

        Warning: This method does NOT render Mermaid diagrams.

        Args:
            json_path: Path to forensic analysis JSON file
            output_path: Optional output PDF file path

        Returns:
            Path to generated PDF file
        """
        import json

        json_file = Path(json_path)
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        return self.generate(data, output_path or str(json_file.with_suffix(".pdf")))

    def generate(
        self,
        data: Dict[str, Any],
        output_path: Optional[str] = None,
    ) -> str:
        """
        Generate PDF report from forensic analysis data.

        Warning: This method does NOT render Mermaid diagrams.

        Args:
            data: Forensic analysis data dictionary
            output_path: Optional output PDF file path

        Returns:
            Path to generated PDF file
        """
        try:
            import weasyprint
        except ImportError:
            raise ImportError("WeasyPrint is required. Install with: uv add weasyprint")

        # Generate HTML
        html_content = self.html_generator.generate(data)

        # Determine output path
        if output_path:
            pdf_path = Path(output_path)
        else:
            analysis_id = data.get("analysis_id", "unknown")
            pdf_path = self.output_dir / f"forensic_report_{analysis_id}.pdf"

        pdf_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to PDF
        css = weasyprint.CSS(
            string="""
            @page {
                size: A4;
                margin: 1cm;
            }
            """
        )

        doc = weasyprint.HTML(string=html_content)
        doc.write_pdf(str(pdf_path), stylesheets=[css])

        return str(pdf_path)
