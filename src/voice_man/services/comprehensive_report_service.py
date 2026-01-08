"""
종합 보고서 생성 서비스

분석 결과(범죄 태깅, 가스라이팅, 감정 분석)를 통합하여
HTML 보고서를 생성하고 PDF로 변환하는 서비스
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from jinja2 import Template, Environment, FileSystemLoader

from voice_man.services.analysis_pipeline_service import AnalysisResult


logger = logging.getLogger(__name__)


class ComprehensiveReportService:
    """
    종합 보고서 생성 서비스

    기능:
    1. 분석 결과 수집 및 통합
    2. HTML 보고서 생성 (Jinja2)
    3. PDF 변환 (WeasyPrint)
    4. 보고서 저장
    """

    def __init__(
        self,
        template_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
    ):
        """
        초기화

        Args:
            template_dir: HTML 템플릿 디렉토리
            output_dir: 보고서 출력 디렉토리
        """
        if template_dir is None:
            template_dir = Path(__file__).parent.parent.parent / "templates"
        if output_dir is None:
            output_dir = Path("reports")

        self.template_dir = Path(template_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Jinja2 환경 설정
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            autoescape=True,
        )

    def collect_analysis_results(self, analysis_results: List[AnalysisResult]) -> Dict[str, Any]:
        """
        분석 결과 수집 및 통합

        Args:
            analysis_results: 분석 결과 리스트

        Returns:
            통합된 데이터 딕셔너리
        """
        summary = {
            "total_files": len(analysis_results),
            "successful": 0,
            "failed": 0,
            "partial": 0,
            "files": [],
        }

        for result in analysis_results:
            file_info = {
                "file_path": result.file_path,
                "status": result.status,
                "timestamp": result.timestamp,
            }

            # 성공한 경우에만 상세 정보 추가
            if result.status == "success":
                summary["successful"] += 1

                # STT 결과
                if result.transcription:
                    file_info["transcription"] = {
                        "text": result.transcription.get("text", ""),
                        "language": result.transcription.get("language", "unknown"),
                        "segments_count": len(result.transcription.get("segments", [])),
                    }

                # 범죄 태그
                if result.crime_tags:
                    crime_tags = []
                    for tag in result.crime_tags:
                        if hasattr(tag, "type"):
                            crime_tags.append(
                                {
                                    "type": tag.type.value
                                    if hasattr(tag.type, "value")
                                    else str(tag.type),
                                    "confidence": tag.confidence
                                    if hasattr(tag, "confidence")
                                    else 0.0,
                                    "keywords": tag.keywords if hasattr(tag, "keywords") else [],
                                }
                            )
                    file_info["crime_tags"] = crime_tags
                    file_info["crime_tags_count"] = len(crime_tags)

                # 가스라이팅 패턴
                if result.gaslighting_patterns:
                    patterns = []
                    for pattern in result.gaslighting_patterns:
                        if hasattr(pattern, "type"):
                            patterns.append(
                                {
                                    "type": pattern.type.value
                                    if hasattr(pattern.type, "value")
                                    else str(pattern.type),
                                    "confidence": pattern.confidence
                                    if hasattr(pattern, "confidence")
                                    else 0.0,
                                    "intensity": pattern.intensity
                                    if hasattr(pattern, "intensity")
                                    else 0.0,
                                }
                            )
                    file_info["gaslighting_patterns"] = patterns
                    file_info["gaslighting_patterns_count"] = len(patterns)

                # 감정 분석
                if result.emotions:
                    emotions = []
                    for emotion in result.emotions:
                        if hasattr(emotion, "primary_emotion"):
                            emotions.append(
                                {
                                    "primary_emotion": emotion.primary_emotion.value
                                    if hasattr(emotion.primary_emotion, "value")
                                    else str(emotion.primary_emotion),
                                    "intensity": emotion.intensity
                                    if hasattr(emotion, "intensity")
                                    else 0.0,
                                    "confidence": emotion.confidence
                                    if hasattr(emotion, "confidence")
                                    else 0.0,
                                }
                            )
                    file_info["emotions"] = emotions

            elif result.status == "failed":
                summary["failed"] += 1
                file_info["error"] = result.error

            elif result.status == "partial":
                summary["partial"] += 1
                if result.error:
                    file_info["error"] = result.error

            summary["files"].append(file_info)

        # 전체 통계 계산
        summary["success_rate"] = (
            summary["successful"] / summary["total_files"] if summary["total_files"] > 0 else 0.0
        )

        # 전체 범죄 태그 통계
        all_crime_tags = []
        for file_info in summary["files"]:
            if "crime_tags" in file_info:
                all_crime_tags.extend(file_info["crime_tags"])

        crime_type_counts: Dict[str, int] = {}
        for tag in all_crime_tags:
            crime_type = tag["type"]
            crime_type_counts[crime_type] = crime_type_counts.get(crime_type, 0) + 1

        summary["crime_statistics"] = {
            "total_tags": len(all_crime_tags),
            "by_type": crime_type_counts,
        }

        # 전체 가스라이팅 통계
        all_gaslighting = []
        for file_info in summary["files"]:
            if "gaslighting_patterns" in file_info:
                all_gaslighting.extend(file_info["gaslighting_patterns"])

        pattern_type_counts: Dict[str, int] = {}
        for pattern in all_gaslighting:
            pattern_type = pattern["type"]
            pattern_type_counts[pattern_type] = pattern_type_counts.get(pattern_type, 0) + 1

        summary["gaslighting_statistics"] = {
            "total_patterns": len(all_gaslighting),
            "by_type": pattern_type_counts,
        }

        return summary

    def generate_html_report(
        self, analysis_results: List[AnalysisResult], report_title: str = "음성 녹취 분석 보고서"
    ) -> str:
        """
        HTML 보고서 생성

        Args:
            analysis_results: 분석 결과 리스트
            report_title: 보고서 제목

        Returns:
            HTML 컨텐츠 (문자열)
        """
        # 데이터 수집
        data = self.collect_analysis_results(analysis_results)

        # 템플릿 컨텍스트 준비
        context = {
            "report_title": report_title,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "summary": data,
            "css_styles": self._get_default_css(),
        }

        # HTML 템플릿 생성 (간단한 내장 템플릿)
        html_template = Template(self._get_html_template())

        # 렌더링
        html_content = html_template.render(**context)

        return html_content

    def save_html_report(self, html_content: str, output_path: Path) -> None:
        """
        HTML 보고서 저장

        Args:
            html_content: HTML 컨텐츠
            output_path: 저장 경로
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info(f"HTML report saved to {output_path}")

    def convert_to_pdf(self, html_content: str, output_path: Path) -> None:
        """
        HTML을 PDF로 변환

        Args:
            html_content: HTML 컨텐츠
            output_path: PDF 출력 경로

        Note:
            WeasyPrint가 설치되어 있어야 함
            설치: pip install weasyprint
        """
        try:
            from weasyprint import HTML

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # PDF 변환
            HTML(string=html_content).write_pdf(str(output_path))

            logger.info(f"PDF report saved to {output_path}")

        except ImportError:
            logger.error("WeasyPrint not installed. Install with: pip install weasyprint")
            raise RuntimeError("WeasyPrint required for PDF conversion")
        except Exception as e:
            logger.error(f"PDF conversion failed: {e}")
            raise

    def generate_comprehensive_report(
        self,
        analysis_results: List[AnalysisResult],
        output_prefix: str = "report",
        generate_pdf: bool = True,
    ) -> Dict[str, str]:
        """
        종합 보고서 생성 (HTML + PDF)

        Args:
            analysis_results: 분석 결과 리스트
            output_prefix: 출력 파일 접두사
            generate_pdf: PDF 생성 여부

        Returns:
            생성된 파일 경로 딕셔너리 {"html": path, "pdf": path}
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # HTML 생성
        html_content = self.generate_html_report(analysis_results)

        # HTML 저장
        html_path = self.output_dir / f"{output_prefix}_{timestamp}.html"
        self.save_html_report(html_content, html_path)

        result_paths = {"html": str(html_path)}

        # PDF 변환
        if generate_pdf:
            pdf_path = self.output_dir / f"{output_prefix}_{timestamp}.pdf"
            self.convert_to_pdf(html_content, pdf_path)
            result_paths["pdf"] = str(pdf_path)

        return result_paths

    def _get_html_template(self) -> str:
        """기본 HTML 템플릿 반환"""
        return """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ report_title }}</title>
    <style>
        {{ css_styles }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{{ report_title }}</h1>
            <p class="timestamp">생성일시: {{ generated_at }}</p>
        </header>

        <section class="summary">
            <h2>분석 요약</h2>
            <div class="stats">
                <div class="stat-item">
                    <span class="stat-label">전체 파일:</span>
                    <span class="stat-value">{{ summary.total_files }}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">성공:</span>
                    <span class="stat-value success">{{ summary.successful }}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">실패:</span>
                    <span class="stat-value failed">{{ summary.failed }}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">부분 성공:</span>
                    <span class="stat-value partial">{{ summary.partial }}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">성공률:</span>
                    <span class="stat-value">{{ "%.1f%%"|format(summary.success_rate * 100) }}</span>
                </div>
            </div>
        </section>

        {% if summary.crime_statistics.total_tags > 0 %}
        <section class="crime-analysis">
            <h2>범죄 발언 분석</h2>
            <div class="statistics">
                <h3>전체 범죄 태그: {{ summary.crime_statistics.total_tags }}개</h3>
                <ul>
                {% for crime_type, count in summary.crime_statistics.by_type.items() %}
                    <li>{{ crime_type }}: {{ count }}개</li>
                {% endfor %}
                </ul>
            </div>
        </section>
        {% endif %}

        {% if summary.gaslighting_statistics.total_patterns > 0 %}
        <section class="gaslighting-analysis">
            <h2>가스라이팅 패턴 분석</h2>
            <div class="statistics">
                <h3>전체 패턴: {{ summary.gaslighting_statistics.total_patterns }}개</h3>
                <ul>
                {% for pattern_type, count in summary.gaslighting_statistics.by_type.items() %}
                    <li>{{ pattern_type }}: {{ count }}개</li>
                {% endfor %}
                </ul>
            </div>
        </section>
        {% endif %}

        <section class="file-details">
            <h2>파일별 상세 분석</h2>
            {% for file in summary.files %}
            <div class="file-item {{ file.status }}">
                <h3>{{ file.file_path }}</h3>
                <p class="status">상태: <span class="{{ file.status }}">{{ file.status }}</span></p>

                {% if file.transcription %}
                <div class="transcription">
                    <h4>STT 변환</h4>
                    <p><strong>언어:</strong> {{ file.transcription.language }}</p>
                    <p><strong>세그먼트 수:</strong> {{ file.transcription.segments_count }}</p>
                    <p><strong>텍스트:</strong> {{ file.transcription.text[:200] }}{% if file.transcription.text|length > 200 %}...{% endif %}</p>
                </div>
                {% endif %}

                {% if file.crime_tags %}
                <div class="crime-tags">
                    <h4>범죄 태그 ({{ file.crime_tags_count }}개)</h4>
                    <ul>
                    {% for tag in file.crime_tags %}
                        <li>
                            <span class="tag-type">{{ tag.type }}</span>
                            <span class="tag-confidence">신뢰도: {{ "%.1f%%"|format(tag.confidence * 100) }}</span>
                        </li>
                    {% endfor %}
                    </ul>
                </div>
                {% endif %}

                {% if file.gaslighting_patterns %}
                <div class="gaslighting-patterns">
                    <h4>가스라이팅 패턴 ({{ file.gaslighting_patterns_count }}개)</h4>
                    <ul>
                    {% for pattern in file.gaslighting_patterns %}
                        <li>
                            <span class="pattern-type">{{ pattern.type }}</span>
                            <span class="pattern-confidence">신뢰도: {{ "%.1f%%"|format(pattern.confidence * 100) }}</span>
                            <span class="pattern-intensity">강도: {{ "%.1f%%"|format(pattern.intensity * 100) }}</span>
                        </li>
                    {% endfor %}
                    </ul>
                </div>
                {% endif %}

                {% if file.emotions %}
                <div class="emotions">
                    <h4>감정 분석</h4>
                    <ul>
                    {% for emotion in file.emotions %}
                        <li>
                            <span class="emotion-type">{{ emotion.primary_emotion }}</span>
                            <span class="emotion-intensity">강도: {{ "%.1f%%"|format(emotion.intensity * 100) }}</span>
                        </li>
                    {% endfor %}
                    </ul>
                </div>
                {% endif %}

                {% if file.error %}
                <div class="error">
                    <h4>에러</h4>
                    <p>{{ file.error }}</p>
                </div>
                {% endif %}
            </div>
            {% endfor %}
        </section>

        <footer>
            <p>이 보고서는 음성 녹취 증거 분석 시스템에 의해 자동 생성되었습니다.</p>
        </footer>
    </div>
</body>
</html>
        """

    def _get_default_css(self) -> str:
        """기본 CSS 스타일 반환"""
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Noto Sans KR', -apple-system, BlinkMacSystemFont, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        header {
            border-bottom: 2px solid #007bff;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }

        header h1 {
            color: #007bff;
            font-size: 2em;
            margin-bottom: 10px;
        }

        .timestamp {
            color: #666;
            font-size: 0.9em;
        }

        section {
            margin-bottom: 40px;
        }

        h2 {
            color: #333;
            border-left: 4px solid #007bff;
            padding-left: 10px;
            margin-bottom: 20px;
        }

        h3 {
            color: #555;
            margin-bottom: 15px;
        }

        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }

        .stat-item {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            display: flex;
            justify-content: space-between;
        }

        .stat-label {
            font-weight: bold;
            color: #555;
        }

        .stat-value {
            font-weight: bold;
        }

        .stat-value.success {
            color: #28a745;
        }

        .stat-value.failed {
            color: #dc3545;
        }

        .stat-value.partial {
            color: #ffc107;
        }

        .file-item {
            background-color: #f8f9fa;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 5px;
            border-left: 4px solid #6c757d;
        }

        .file-item.success {
            border-left-color: #28a745;
        }

        .file-item.failed {
            border-left-color: #dc3545;
        }

        .file-item.partial {
            border-left-color: #ffc107;
        }

        .file-item h3 {
            word-break: break-all;
            margin-bottom: 10px;
        }

        .status {
            margin-bottom: 15px;
        }

        .status .success {
            color: #28a745;
            font-weight: bold;
        }

        .status .failed {
            color: #dc3545;
            font-weight: bold;
        }

        .status .partial {
            color: #ffc107;
            font-weight: bold;
        }

        .transcription, .crime-tags, .gaslighting-patterns, .emotions {
            margin-bottom: 15px;
            padding: 10px;
            background-color: white;
            border-radius: 3px;
        }

        .transcription h4, .crime-tags h4, .gaslighting-patterns h4, .emotions h4 {
            margin-bottom: 10px;
            color: #007bff;
        }

        ul {
            list-style: none;
            padding-left: 0;
        }

        li {
            padding: 5px 0;
            border-bottom: 1px solid #e9ecef;
        }

        li:last-child {
            border-bottom: none;
        }

        .tag-type, .pattern-type, .emotion-type {
            display: inline-block;
            padding: 3px 8px;
            background-color: #007bff;
            color: white;
            border-radius: 3px;
            margin-right: 10px;
            font-size: 0.9em;
        }

        .tag-confidence, .pattern-confidence, .pattern-intensity, .emotion-intensity {
            color: #666;
            font-size: 0.9em;
        }

        .error {
            background-color: #f8d7da;
            color: #721c24;
            padding: 10px;
            border-radius: 3px;
        }

        footer {
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #dee2e6;
            text-align: center;
            color: #6c757d;
        }

        @media print {
            body {
                background-color: white;
            }

            .container {
                box-shadow: none;
            }
        }
        """
