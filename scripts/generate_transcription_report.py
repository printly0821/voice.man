#!/usr/bin/env python3
"""
Transcription Report Generator

Generate HTML/PDF transcription reports from E2E batch test results.

Usage:
    python scripts/generate_transcription_report.py --input results/e2e_test_report.json
    python scripts/generate_transcription_report.py --input results/e2e_test_report.json --pdf
    python scripts/generate_transcription_report.py --input results/e2e_test_report.json --output-dir reports/
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import jinja2

# Add project root to path
project_root = Path(__file__).parent.parent


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate transcription reports from E2E test results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to E2E test report JSON file",
    )

    parser.add_argument(
        "--output-dir",
        default="reports/transcription",
        help="Output directory for reports (default: reports/transcription)",
    )

    parser.add_argument(
        "--pdf",
        action="store_true",
        help="Also generate PDF reports (requires Playwright)",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of reports to generate (for testing)",
    )

    return parser.parse_args()


def generate_html(
    file_result: Dict[str, Any],
    idx: int,
    total: int,
    template: jinja2.Template,
) -> str:
    """Generate HTML report for a single file.

    Args:
        file_result: Single file result from E2E test
        idx: File index
        total: Total number of files
        template: Jinja2 template

    Returns:
        Generated HTML string
    """
    file_path = Path(file_result["file_path"])
    file_name = file_path.name

    # Format processing time
    proc_time = file_result.get("processing_time_seconds", 0)
    time_str = f"{proc_time:.2f}초" if proc_time else "N/A"

    # Extract speaker information
    speakers = file_result.get("speakers", [])
    speaker_count = len([s for s in speakers if s != "UNKNOWN"])

    # Format segments with speaker info
    segments = file_result.get("segments", [])
    formatted_segments = []
    for seg in segments:
        speaker = seg.get("speaker", "UNKNOWN")
        start = seg.get("start", 0)
        end = seg.get("end", 0)
        text = seg.get("text", "")

        # Format timestamp as MM:SS
        start_min, start_sec = divmod(int(start), 60)
        end_min, end_sec = divmod(int(end), 60)

        formatted_segments.append(
            {
                "speaker": speaker,
                "start_time": f"{start_min:02d}:{start_sec:02d}",
                "end_time": f"{end_min:02d}:{end_sec:02d}",
                "start_raw": start,
                "end_raw": end,
                "text": text.strip(),
            }
        )

    # Group by speaker for summary
    speaker_segments: Dict[str, List[Dict]] = {}
    for seg in formatted_segments:
        spk = seg["speaker"]
        if spk not in speaker_segments:
            speaker_segments[spk] = []
        speaker_segments[spk].append(seg)

    # Calculate speaking time per speaker
    speaker_stats = []
    for speaker, segs in speaker_segments.items():
        if speaker == "UNKNOWN":
            continue
        total_time = sum(s["end_raw"] - s["start_raw"] for s in segs)
        word_count = sum(len(s["text"].split()) for s in segs)
        speaker_stats.append(
            {
                "speaker": speaker,
                "segment_count": len(segs),
                "speaking_time": f"{total_time:.1f}초",
                "word_count": word_count,
            }
        )

    # Sort by speaking time
    speaker_stats.sort(key=lambda x: x["speaking_time"], reverse=True)

    # Render template
    html = template.render(
        file_name=file_name,
        file_path=str(file_path),
        index=idx,
        total=total,
        status=file_result.get("status", "unknown"),
        processing_time=time_str,
        speaker_count=speaker_count,
        speakers=speakers,
        segments=formatted_segments,
        speaker_stats=speaker_stats,
        full_text=file_result.get("transcript_text", ""),
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )

    return html


def create_template() -> jinja2.Template:
    """Create Jinja2 template for HTML report.

    Returns:
        Jinja2 Template object
    """
    template_str = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>전사 보고서 - {{ file_name }}</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Noto Sans KR', -apple-system, BlinkMacSystemFont, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
            padding: 20px;
        }

        .container {
            max-width: 900px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }

        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
        }

        .header h1 {
            font-size: 24px;
            margin-bottom: 10px;
        }

        .header .meta {
            font-size: 14px;
            opacity: 0.9;
        }

        .content {
            padding: 30px;
        }

        .section {
            margin-bottom: 30px;
        }

        .section-title {
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #667eea;
            color: #667eea;
        }

        .info-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }

        .info-item {
            background: #f8f9fa;
            padding: 12px;
            border-radius: 6px;
        }

        .info-label {
            font-size: 12px;
            color: #666;
            margin-bottom: 4px;
        }

        .info-value {
            font-size: 16px;
            font-weight: 600;
            color: #333;
        }

        .speaker-stats {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }

        .speaker-stat {
            background: #f0f4ff;
            padding: 10px 15px;
            border-radius: 20px;
            font-size: 14px;
        }

        .speaker-stat .name {
            font-weight: 600;
            color: #667eea;
        }

        .segments {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
        }

        .segment {
            margin-bottom: 20px;
            padding-bottom: 20px;
            border-bottom: 1px solid #e0e0e0;
        }

        .segment:last-child {
            border-bottom: none;
            margin-bottom: 0;
            padding-bottom: 0;
        }

        .segment-header {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 8px;
        }

        .speaker-badge {
            background: #667eea;
            color: white;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
        }

        .speaker-badge.SPEAKER_00 {
            background: #667eea;
        }

        .speaker-badge.SPEAKER_01 {
            background: #10b981;
        }

        .speaker-badge.SPEAKER_02 {
            background: #f59e0b;
        }

        .timestamp {
            font-size: 12px;
            color: #666;
        }

        .segment-text {
            font-size: 15px;
            line-height: 1.8;
            padding-left: 10px;
        }

        .full-text {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            font-size: 14px;
            line-height: 1.8;
            white-space: pre-wrap;
            max-height: 300px;
            overflow-y: auto;
        }

        .status-success {
            color: #10b981;
            font-weight: 600;
        }

        .status-failed {
            color: #ef4444;
            font-weight: 600;
        }

        .footer {
            text-align: center;
            padding: 20px;
            font-size: 12px;
            color: #666;
            background: #f8f9fa;
        }

        @media print {
            body {
                background: white;
                padding: 0;
            }
            .container {
                box-shadow: none;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>전사 보고서</h1>
            <div class="meta">
                파일: {{ file_name }} | 보고서 #{{ index }}/{{ total }}
            </div>
        </div>

        <div class="content">
            <!-- File Information -->
            <div class="section">
                <div class="section-title">파일 정보</div>
                <div class="info-grid">
                    <div class="info-item">
                        <div class="info-label">파일명</div>
                        <div class="info-value">{{ file_name }}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">상태</div>
                        <div class="info-value {% if status == 'success' %}status-success{% else %}status-failed{% endif %}">
                            {{ '성공' if status == 'success' else '실패' }}
                        </div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">화자 수</div>
                        <div class="info-value">{{ speaker_count }}명</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">처리 시간</div>
                        <div class="info-value">{{ processing_time }}</div>
                    </div>
                </div>
            </div>

            {% if speaker_stats %}
            <!-- Speaker Statistics -->
            <div class="section">
                <div class="section-title">화자 통계</div>
                <div class="speaker-stats">
                    {% for stat in speaker_stats %}
                    <div class="speaker-stat">
                        <span class="name">{{ stat.speaker }}</span>:
                        {{ stat.segment_count }}구간,
                        {{ stat.speaking_time }},
                        {{ stat.word_count }}단어
                    </div>
                    {% endfor %}
                </div>
            </div>
            {% endif %}

            {% if segments %}
            <!-- Transcript Segments -->
            <div class="section">
                <div class="section-title">전사 내용 (화자별)</div>
                <div class="segments">
                    {% for segment in segments %}
                    <div class="segment">
                        <div class="segment-header">
                            <span class="speaker-badge {{ segment.speaker }}">{{ segment.speaker }}</span>
                            <span class="timestamp">{{ segment.start_time }} - {{ segment.end_time }}</span>
                        </div>
                        <div class="segment-text">{{ segment.text }}</div>
                    </div>
                    {% endfor %}
                </div>
            </div>
            {% endif %}

            {% if full_text %}
            <!-- Full Text -->
            <div class="section">
                <div class="section-title">전체 전사 텍스트</div>
                <div class="full-text">{{ full_text }}</div>
            </div>
            {% endif %}
        </div>

        <div class="footer">
            생성일: {{ generated_at }} | VoiceMan E2E 전사 시스템
        </div>
    </div>
</body>
</html>
"""
    return jinja2.Template(template_str)


async def generate_pdf_from_html(html_path: Path, pdf_path: Path) -> None:
    """Generate PDF from HTML using Playwright.

    Args:
        html_path: Path to HTML file
        pdf_path: Path to output PDF file
    """
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        raise ImportError(
            "Playwright is required for PDF generation. "
            "Install with: uv add playwright && uv run playwright install chromium"
        )

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


async def main() -> int:
    """Main entry point.

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    args = parse_args()

    # Load E2E test report
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1

    print(f"Loading E2E test report from {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        e2e_data = json.load(f)

    file_results = e2e_data.get("result", {}).get("file_results", [])
    total_files = len(file_results)

    print(f"Found {total_files} files in E2E report")

    # Apply limit if specified
    if args.limit:
        file_results = file_results[: args.limit]
        print(f"Limited to {len(file_results)} files")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create template
    template = create_template()

    # Generate reports
    html_files = []
    pdf_files = []

    for idx, file_result in enumerate(file_results, 1):
        if file_result.get("status") != "success":
            print(f"[{idx}/{total_files}] Skipping failed file: {file_result.get('file_path')}")
            continue

        file_path = Path(file_result["file_path"])
        base_name = file_path.stem
        output_name = f"{base_name}_transcript"

        print(f"[{idx}/{len(file_results)}] Generating report for: {file_path.name}")

        # Generate HTML
        html_content = generate_html(file_result, idx, total_files, template)

        html_path = output_dir / f"{output_name}.html"
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        html_files.append(html_path)

        print(f"  -> HTML: {html_path}")

        # Generate PDF if requested
        if args.pdf:
            pdf_path = output_dir / f"{output_name}.pdf"
            try:
                await generate_pdf_from_html(html_path, pdf_path)
                pdf_files.append(pdf_path)
                print(f"  -> PDF: {pdf_path}")
            except Exception as e:
                print(f"  -> PDF generation failed: {e}")

    # Generate summary
    print("\n" + "=" * 50)
    print("전사 보고서 생성 완료")
    print("=" * 50)
    print(f"생성된 HTML 보고서: {len(html_files)}개")
    print(f"생성된 PDF 보고서: {len(pdf_files)}개")
    print(f"출력 디렉터리: {output_dir.absolute()}")
    print("=" * 50)

    # Generate index HTML
    index_html = f"""
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>전사 보고서 목록</title>
    <style>
        body {{ font-family: 'Noto Sans KR', sans-serif; padding: 40px; max-width: 1200px; margin: 0 auto; }}
        h1 {{ color: #667eea; }}
        .stats {{ display: flex; gap: 20px; margin: 20px 0; }}
        .stat {{ background: #f8f9fa; padding: 15px 25px; border-radius: 8px; }}
        .stat-label {{ font-size: 12px; color: #666; }}
        .stat-value {{ font-size: 24px; font-weight: 600; color: #333; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #e0e0e0; }}
        th {{ background: #f8f9fa; font-weight: 600; }}
        tr:hover {{ background: #f8f9fa; }}
        a {{ color: #667eea; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
    </style>
</head>
<body>
    <h1>전사 보고서 목록</h1>
    <div class="stats">
        <div class="stat">
            <div class="stat-label">총 파일</div>
            <div class="stat-value">{len(html_files)}</div>
        </div>
        <div class="stat">
            <div class="stat-label">생성일</div>
            <div class="stat-value">{datetime.now().strftime("%Y-%m-%d")}</div>
        </div>
    </div>
    <table>
        <thead>
            <tr>
                <th>#</th>
                <th>파일명</th>
                <th>HTML</th>
                {"<th>PDF</th>" if args.pdf else ""}
            </tr>
        </thead>
        <tbody>
"""

    for idx, html_path in enumerate(html_files, 1):
        file_name = html_path.stem.replace("_transcript", "")
        pdf_link = f'<td><a href="{html_path.stem}.pdf">PDF</a></td>' if args.pdf else ""
        index_html += f"""
            <tr>
                <td>{idx}</td>
                <td>{file_name}</td>
                <td><a href="{html_path.name}">HTML</a></td>
                {pdf_link}
            </tr>
"""

    index_html += """
        </tbody>
    </table>
</body>
</html>
"""

    index_path = output_dir / "index.html"
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(index_html)
    print(f"인덱스 파일: {index_path.absolute()}")

    return 0


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
