#!/usr/bin/env python3
"""
Batch HTML to PDF Converter

Convert transcription HTML reports to PDF using Playwright.
"""

import asyncio
from pathlib import Path
from typing import List

from playwright.async_api import async_playwright


async def convert_html_to_pdf(html_path: Path, pdf_path: Path) -> bool:
    """Convert a single HTML file to PDF.

    Args:
        html_path: Path to HTML file
        pdf_path: Path to output PDF file

    Returns:
        True if successful, False otherwise
    """
    try:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch()
            page = await browser.new_page()

            # Load HTML file
            await page.goto(f"file://{html_path.absolute()}", wait_until="networkidle")

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

            await browser.close()
            return True

    except Exception as e:
        print(f"Error converting {html_path.name}: {e}")
        return False


async def batch_convert(
    html_dir: Path,
    output_dir: Path,
    pattern: str = "*_transcript.html",
) -> None:
    """Batch convert HTML files to PDF.

    Args:
        html_dir: Directory containing HTML files
        output_dir: Directory for PDF output
        pattern: Glob pattern for HTML files
    """
    html_files = sorted(html_dir.glob(pattern))
    total = len(html_files)

    print(f"Found {total} HTML files to convert")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert in smaller batches to avoid memory issues
    batch_size = 10
    success_count = 0

    for i in range(0, total, batch_size):
        batch = html_files[i : i + batch_size]
        batch_end = min(i + batch_size, total)

        print(f"\nBatch {i // batch_size + 1}: Converting files {i + 1}-{batch_end}...")

        for html_path in batch:
            pdf_path = output_dir / f"{html_path.stem}.pdf"

            if await convert_html_to_pdf(html_path, pdf_path):
                success_count += 1
                print(f"  [{i + html_files.index(html_path) + 1}/{total}] {html_path.name} -> PDF")
            else:
                print(f"  [{i + html_files.index(html_path) + 1}/{total}] FAILED: {html_path.name}")

    print(f"\n{'=' * 50}")
    print("PDF 변환 완료")
    print(f"성공: {success_count}/{total}")
    print(f"출력 디렉터리: {output_dir.absolute()}")
    print(f"{'=' * 50}")


async def main() -> int:
    """Main entry point.

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    html_dir = Path("/home/innojini/dev/voice.man/ref/call/results/transcripts")
    output_dir = Path("/home/innojini/dev/voice.man/ref/call/results/transcripts")

    await batch_convert(html_dir, output_dir)

    return 0


if __name__ == "__main__":
    asyncio.run(main())
