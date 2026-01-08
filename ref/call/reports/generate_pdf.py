#!/usr/bin/env python3
"""
Generate PDF from HTML comprehensive analysis report.
"""

import sys
from pathlib import Path


def generate_pdf() -> None:
    """Generate PDF from HTML report using wkhtmltopdf or WeasyPrint."""

    html_path = Path("/Users/innojini/Dev/voice.man/ref/call/reports/comprehensive_analysis_report.html")
    pdf_path = Path("/Users/innojini/Dev/voice.man/ref/call/reports/comprehensive_analysis_report.pdf")

    try:
        # Try WeasyPrint first
        from weasyprint import HTML

        print("WeasyPrint로 PDF 변환 중...")
        HTML(filename=str(html_path)).write_pdf(str(pdf_path))
        print(f"PDF 생성 완료: {pdf_path}")

    except ImportError:
        print("WeasyPrint가 설치되지 않았습니다.")
        print("설치 명령어: pip install weasyprint")
        print("\n또는 wkhtmltopdf를 사용하세요:")
        print(f"wkhtmltopdf {html_path} {pdf_path}")

        # Alternative: Try pyppeteer (Chrome headless)
        try:
            import asyncio

            from pyppeteer import launch

            async def convert():
                browser = await launch(headless=True)
                page = await browser.newPage()
                await page.goto(f"file://{html_path.absolute()}")
                await page.pdf({
                    "path": str(pdf_path),
                    "format": "A4",
                    "printBackground": True
                })
                await browser.close()

            print("Pyppeteer로 PDF 변환 중...")
            asyncio.get_event_loop().run_until_complete(convert())
            print(f"PDF 생성 완료: {pdf_path}")

        except ImportError:
            print("Pyppeteer도 설치되지 않았습니다.")
            print("설치 명령어: pip install pyppeteer")
            sys.exit(1)

if __name__ == "__main__":
    generate_pdf()
