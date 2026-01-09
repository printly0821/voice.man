#!/usr/bin/env python3
"""
Forensic Report Generation CLI

Generate HTML/PDF reports from forensic analysis JSON results.

Usage:
    python scripts/generate_forensic_report.py <json_file> [options]

Examples:
    # Generate HTML only
    python scripts/generate_forensic_report.py results/forensic_result.json

    # Generate PDF with Mermaid diagrams
    python scripts/generate_forensic_report.py results/forensic_result.json --pdf

    # Generate both HTML and PDF
    python scripts/generate_forensic_report.py results/forensic_result.json --both

    # Batch process multiple files
    python scripts/generate_forensic_report.py results/*.json --batch
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from voice_man.reports.html_generator import ForensicHTMLGenerator
from voice_man.reports.pdf_generator import ForensicPDFGenerator


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate forensic analysis reports from JSON results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "input",
        help="Path to forensic JSON file or directory for batch processing",
    )

    parser.add_argument(
        "-o",
        "--output",
        help="Output file path (default: <input>.html or <input>.pdf)",
    )

    parser.add_argument(
        "--pdf",
        action="store_true",
        help="Generate PDF (requires Playwright)",
    )

    parser.add_argument(
        "--html",
        action="store_true",
        help="Generate HTML (default if no format specified)",
    )

    parser.add_argument(
        "--both",
        action="store_true",
        help="Generate both HTML and PDF",
    )

    parser.add_argument(
        "--batch",
        action="store_true",
        help="Batch process all JSON files in input directory",
    )

    parser.add_argument(
        "--output-dir",
        help="Output directory for batch processing",
    )

    parser.add_argument(
        "--simple-pdf",
        action="store_true",
        help="Use WeasyPrint for PDF (no JavaScript/Mermaid rendering)",
    )

    return parser.parse_args()


def generate_html(json_path: str, output_path: str | None = None) -> str:
    """
    Generate HTML report from forensic JSON.

    Args:
        json_path: Path to forensic JSON file
        output_path: Optional output HTML path

    Returns:
        Path to generated HTML file
    """
    print(f"Generating HTML from {json_path}...")
    generator = ForensicHTMLGenerator()
    result_path = generator.generate_from_json(json_path, output_path)
    print(f"HTML report saved to: {result_path}")
    return result_path


async def generate_pdf(json_path: str, output_path: str | None = None, simple: bool = False) -> str:
    """
    Generate PDF report from forensic JSON.

    Args:
        json_path: Path to forensic JSON file
        output_path: Optional output PDF path
        simple: Use WeasyPrint instead of Playwright

    Returns:
        Path to generated PDF file
    """
    print(f"Generating PDF from {json_path}...")

    if simple:
        from voice_man.reports.pdf_generator import SimplePDFGenerator

        generator = SimplePDFGenerator()
        result_path = generator.generate_from_json(json_path, output_path)
    else:
        generator = ForensicPDFGenerator()
        result_path = await generator.generate_from_json(json_path, output_path)
        await generator.close()

    print(f"PDF report saved to: {result_path}")
    return result_path


async def process_file(args: argparse.Namespace, json_path: Path) -> None:
    """
    Process a single JSON file.

    Args:
        args: Parsed command line arguments
        json_path: Path to JSON file
    """
    if not json_path.exists():
        print(f"Error: File not found: {json_path}", file=sys.stderr)
        return

    # Determine output paths
    base_output = args.output
    if not base_output:
        base_output = json_path.with_suffix("")

    if args.both:
        # Generate both
        html_output = str(base_output) + ".html"
        pdf_output = str(base_output) + ".pdf"

        generate_html(str(json_path), html_output)

        if args.simple_pdf:
            from voice_man.reports.pdf_generator import SimplePDFGenerator

            generator = SimplePDFGenerator()
            pdf_path = generator.generate_from_json(str(json_path), pdf_output)
            print(f"PDF report saved to: {pdf_path}")
        else:
            generator = ForensicPDFGenerator()
            pdf_path = await generator.generate_from_json(str(json_path), pdf_output)
            print(f"PDF report saved to: {pdf_path}")
            await generator.close()

    elif args.pdf:
        # PDF only
        if args.simple_pdf:
            from voice_man.reports.pdf_generator import SimplePDFGenerator

            generator = SimplePDFGenerator()
            pdf_path = generator.generate_from_json(str(json_path), args.output)
            print(f"PDF report saved to: {pdf_path}")
        else:
            generator = ForensicPDFGenerator()
            pdf_path = await generator.generate_from_json(str(json_path), args.output)
            print(f"PDF report saved to: {pdf_path}")
            await generator.close()

    else:
        # HTML only (default)
        generate_html(str(json_path), args.output)


async def process_batch(args: argparse.Namespace) -> None:
    """
    Process multiple JSON files in batch.

    Args:
        args: Parsed command line arguments
    """
    input_path = Path(args.input)

    if input_path.is_file():
        json_files = [input_path]
    elif input_path.is_dir():
        json_files = sorted(input_path.glob("*.json"))
        if not json_files:
            print(f"No JSON files found in {input_path}", file=sys.stderr)
            return
    else:
        print(f"Error: Path not found: {input_path}", file=sys.stderr)
        return

    print(f"Processing {len(json_files)} file(s)...")

    output_dir = Path(args.output_dir) if args.output_dir else None

    if args.pdf or args.both:
        # Use PDF generator for batch
        if args.simple_pdf:
            from voice_man.reports.pdf_generator import SimplePDFGenerator

            generator = SimplePDFGenerator(output_dir)

            for json_file in json_files:
                try:
                    output_path = None
                    if output_dir:
                        output_path = str(output_dir / json_file.with_suffix(".pdf").name)
                    generator.generate_from_json(str(json_file), output_path)
                except Exception as e:
                    print(f"Error processing {json_file}: {e}")
        else:
            generator = ForensicPDFGenerator(output_dir=output_dir)

            json_paths = [str(f) for f in json_files]
            await generator.generate_batch(json_paths, args.output_dir)
            await generator.close()

    else:
        # HTML only
        generator = ForensicHTMLGenerator()

        for json_file in json_files:
            try:
                output_path = None
                if output_dir:
                    output_path = str(output_dir / json_file.with_suffix(".html").name)
                generator.generate_from_json(str(json_file), output_path)
            except Exception as e:
                print(f"Error processing {json_file}: {e}")

    print("Batch processing complete.")


async def main() -> int:
    """
    Main entry point.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    args = parse_args()

    try:
        if args.batch:
            await process_batch(args)
        else:
            await process_file(args, Path(args.input))

        return 0

    except KeyboardInterrupt:
        print("\nOperation cancelled.", file=sys.stderr)
        return 1

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
