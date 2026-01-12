"""
포렌식 모니터링 HTML 보고서 생성 서비스

STT 정확도와 포렌식 분석 결과를 시각화된 HTML 보고서로 생성합니다.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class MonitoringReportService:
    """
    포렌식 모니터링 결과를 HTML 보고서로 생성하는 서비스
    """

    def __init__(self):
        """서비스 초기화"""
        self.template_dir = Path(__file__).parent.parent / "templates"

    def generate_html_report(
        self,
        results: List[Dict[str, Any]],
        output_path: str,
        title: str = "포렌식 모니터링 보고서",
    ) -> str:
        """
        결과 리스트를 HTML 보고서로 생성

        Args:
            results: 분석 결과 딕셔너리 리스트
            output_path: 출력 HTML 파일 경로
            title: 보고서 제목

        Returns:
            생성된 HTML 파일 경로
        """
        # 통계 계산
        summary = self._calculate_summary(results)

        # HTML 생성
        html_content = self._generate_html(results, summary, title)

        # 파일 저장
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(html_content, encoding="utf-8")

        return str(output_file)

    def _calculate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """결과 요약 통계 계산"""
        if not results:
            return {}

        total_files = len(results)
        total_duration = sum(r.get("duration_seconds", 0) for r in results)
        total_words = sum(r.get("stt_total_words", 0) for r in results)

        # 위험도 통계
        avg_risk = sum(r.get("overall_risk_score", 0) for r in results) / total_files
        high_risk = sum(1 for r in results if r.get("overall_risk_score", 0) >= 60)
        critical = sum(1 for r in results if r.get("overall_risk_score", 0) >= 80)

        # STT 신뢰도 통계
        avg_confidence = sum(r.get("stt_confidence_avg", 0) for r in results) / total_files
        grade_dist = {}
        for r in results:
            g = r.get("stt_confidence_grade", "N/A")
            grade_dist[g] = grade_dist.get(g, 0) + 1

        # WER/CER 통계 (참조 있는 파일만)
        results_with_ref = [r for r in results if r.get("stt_has_reference")]
        if results_with_ref:
            avg_wer = sum(r.get("stt_wer", 0) for r in results_with_ref if r.get("stt_wer")) / len(
                results_with_ref
            )
            avg_cer = sum(r.get("stt_cer", 0) for r in results_with_ref if r.get("stt_cer")) / len(
                results_with_ref
            )
        else:
            avg_wer = avg_cer = None

        # 카테고리 평균
        categories = {
            "가스라이팅": "gaslighting_score",
            "협박": "threat_score",
            "강요": "coercion_score",
            "사기": "deception_score",
            "감정조작": "emotional_score",
        }
        category_avg = {}
        for name, key in categories.items():
            category_avg[name] = sum(r.get(key, 0) for r in results) / total_files

        return {
            "total_files": total_files,
            "total_duration_minutes": total_duration / 60,
            "total_words": total_words,
            "avg_risk": avg_risk,
            "high_risk_count": high_risk,
            "critical_count": critical,
            "avg_confidence": avg_confidence,
            "grade_distribution": grade_dist,
            "avg_wer": avg_wer,
            "avg_cer": avg_cer,
            "category_averages": category_avg,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    def _generate_html(
        self, results: List[Dict[str, Any]], summary: Dict[str, Any], title: str
    ) -> str:
        """HTML 콘텐츠 생성"""

        # 등급별 색상 매핑
        grade_colors = {
            "A": "#10b981",  # 초록
            "B": "#22c55e",  # 연두
            "C": "#eab308",  # 노랑
            "D": "#f97316",  # 주황
            "F": "#ef4444",  # 빨강
            "N/A": "#6b7280",  # 회색
        }

        # 위험도 등급 색상
        risk_colors = {
            "낮음": "#10b981",
            "중간": "#eab308",
            "높음": "#f97316",
            "심각": "#ef4444",
        }

        html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            min-height: 100vh;
            padding: 20px;
            color: #e2e8f0;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        .header {{
            text-align: center;
            padding: 40px 20px;
            background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
            border-radius: 16px;
            margin-bottom: 30px;
            box-shadow: 0 10px 40px rgba(59, 130, 246, 0.3);
        }}
        .header h1 {{
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 10px;
            color: white;
        }}
        .header .meta {{
            color: rgba(255, 255, 255, 0.8);
            font-size: 1rem;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .summary-card {{
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
        }}
        .summary-card .value {{
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 5px;
        }}
        .summary-card .label {{
            color: #94a3b8;
            font-size: 0.875rem;
        }}
        .section {{
            background: rgba(255, 255, 255, 0.03);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 30px;
            margin-bottom: 30px;
        }}
        .section-title {{
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: 20px;
            color: #f8fafc;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .grade-distribution {{
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
        }}
        .grade-bar {{
            flex: 1;
            min-width: 100px;
        }}
        .grade-bar .bar {{
            height: 30px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 600;
            color: white;
            margin-bottom: 5px;
        }}
        .grade-bar .count {{
            text-align: center;
            color: #94a3b8;
            font-size: 0.875rem;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }}
        th {{
            background: rgba(59, 130, 246, 0.2);
            font-weight: 600;
            color: #f8fafc;
        }}
        tr:hover {{
            background: rgba(255, 255, 255, 0.05);
        }}
        .grade-badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.875rem;
            font-weight: 600;
        }}
        .risk-badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.875rem;
            font-weight: 600;
        }}
        .confidence-bar {{
            width: 100px;
            height: 8px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 4px;
            overflow: hidden;
            display: inline-block;
            vertical-align: middle;
            margin-left: 10px;
        }}
        .confidence-fill {{
            height: 100%;
            background: linear-gradient(90deg, #10b981, #22c55e);
            transition: width 0.3s;
        }}
        .detail-card {{
            background: rgba(0, 0, 0, 0.2);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 15px;
        }}
        .detail-card .file-name {{
            font-weight: 600;
            color: #f8fafc;
            margin-bottom: 10px;
        }}
        .detail-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
        }}
        .detail-item {{
            display: flex;
            flex-direction: column;
        }}
        .detail-item .label {{
            color: #94a3b8;
            font-size: 0.75rem;
            margin-bottom: 2px;
        }}
        .detail-item .value {{
            color: #f8fafc;
            font-weight: 500;
        }}
        .category-bars {{
            display: flex;
            gap: 10px;
            margin-top: 10px;
        }}
        .category-bar {{
            flex: 1;
            height: 20px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 4px;
            position: relative;
            overflow: hidden;
        }}
        .category-fill {{
            height: 100%;
            border-radius: 4px;
            transition: width 0.3s;
        }}
        .category-label {{
            position: absolute;
            left: 50%;
            top: 50%;
            transform: translate(-50%, -50%);
            font-size: 0.75rem;
            font-weight: 600;
            color: white;
            text-shadow: 0 1px 2px rgba(0, 0, 0, 0.5);
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- 헤더 -->
        <div class="header">
            <h1>{title}</h1>
            <div class="meta">
                생성일시: {summary.get("generated_at", "N/A")} |
                처리 파일: {summary.get("total_files", 0)}개 |
                총 길이: {summary.get("total_duration_minutes", 0):.1f}분
            </div>
        </div>

        <!-- 요약 통계 -->
        <div class="summary-grid">
            <div class="summary-card">
                <div class="value" style="color: #f97316;">{summary.get("avg_risk", 0):.1f}</div>
                <div class="label">평균 위험도 점수</div>
            </div>
            <div class="summary-card">
                <div class="value" style="color: #10b981;">{summary.get("avg_confidence", 0):.3f}</div>
                <div class="label">평균 STT 신뢰도</div>
            </div>
            <div class="summary-card">
                <div class="value" style="color: #3b82f6;">{summary.get("total_words", 0)}</div>
                <div class="label">총 단어 수</div>
            </div>
            <div class="summary-card">
                <div class="value" style="color: #ef4444;">{summary.get("high_risk_count", 0)}</div>
                <div class="label">고위험 파일 (≥60)</div>
            </div>
        </div>
"""

        # 신뢰도 등급 분포
        grade_dist = summary.get("grade_distribution", {})
        if grade_dist:
            html += """
        <!-- 신뢰도 등급 분포 -->
        <div class="section">
            <div class="section-title">📊 STT 신뢰도 등급 분포</div>
            <div class="grade-distribution">
"""
            for grade in ["A", "B", "C", "D", "F"]:
                count = grade_dist.get(grade, 0)
                color = grade_colors.get(grade, "#6b7280")
                html += f"""
                <div class="grade-bar">
                    <div class="bar" style="background: {color};">{grade}등급</div>
                    <div class="count">{count}개 파일</div>
                </div>
"""
            html += """
            </div>
        </div>
"""

        # 카테고리별 평균
        category_avg = summary.get("category_averages", {})
        if category_avg:
            html += """
        <!-- 카테고리별 평균 -->
        <div class="section">
            <div class="section-title">📈 카테고리별 평균 점수</div>
            <div class="category-bars">
"""
            colors = {
                "가스라이팅": "#a855f7",
                "협박": "#ef4444",
                "강요": "#f97316",
                "사기": "#eab308",
                "감정조작": "#ec4899",
            }
            for name, score in category_avg.items():
                color = colors.get(name, "#3b82f6")
                html += f"""
                <div class="category-bar">
                    <div class="category-fill" style="width: {score}%; background: {color};"></div>
                    <div class="category-label">{name}: {score:.1f}</div>
                </div>
"""
            html += """
            </div>
        </div>
"""

        # 파일별 상세 결과
        html += """
        <!-- 파일별 상세 결과 -->
        <div class="section">
            <div class="section-title">📁 파일별 분석 결과</div>
            <table>
                <thead>
                    <tr>
                        <th>파일명</th>
                        <th>길이</th>
                        <th>화자</th>
                        <th>위험도</th>
                        <th>신뢰도</th>
                        <th>단어수</th>
                        <th>WPM</th>
                    </tr>
                </thead>
                <tbody>
"""
        for r in results:
            filename = r.get("file_name", "")
            if len(filename) > 30:
                filename = filename[:27] + "..."

            risk_level = r.get("overall_risk_level", "N/A")
            risk_color = risk_colors.get(risk_level, "#6b7280")

            confidence_grade = r.get("stt_confidence_grade", "N/A")
            grade_color = grade_colors.get(confidence_grade, "#6b7280")

            confidence_avg = r.get("stt_confidence_avg", 0)

            html += f"""
                    <tr>
                        <td title="{r.get("file_name", "")}">{filename}</td>
                        <td>{r.get("duration_seconds", 0):.1f}초</td>
                        <td>{r.get("num_speakers", 0)}명</td>
                        <td>
                            <span class="risk-badge" style="background: {risk_color};">{risk_level}</span>
                            ({r.get("overall_risk_score", 0):.1f})
                        </td>
                        <td>
                            <span class="grade-badge" style="background: {grade_color};">{confidence_grade}</span>
                            {confidence_avg:.3f}
                            <div class="confidence-bar">
                                <div class="confidence-fill" style="width: {confidence_avg * 100}%;"></div>
                            </div>
                        </td>
                        <td>{r.get("stt_total_words", 0)}</td>
                        <td>{r.get("stt_words_per_minute", 0):.0f}</td>
                    </tr>
"""
        html += """
                </tbody>
            </table>
        </div>
"""

        # 상세 분석 카드
        html += """
        <!-- 상세 분석 -->
        <div class="section">
            <div class="section-title">🔍 상세 분석</div>
"""
        for r in results:
            filename = r.get("file_name", "")
            risk_level = r.get("overall_risk_level", "N/A")
            risk_color = risk_colors.get(risk_level, "#6b7280")

            confidence_grade = r.get("stt_confidence_grade", "N/A")
            grade_color = grade_colors.get(confidence_grade, "#6b7280")

            html += f"""
            <div class="detail-card">
                <div class="file-name">
                    {filename}
                    <span style="float: right;">
                        <span class="risk-badge" style="background: {risk_color};">{risk_level}</span>
                        <span class="grade-badge" style="background: {grade_color}; margin-left: 5px;">{confidence_grade}</span>
                    </span>
                </div>
                <div class="detail-grid">
                    <div class="detail-item">
                        <div class="label">위험도 점수</div>
                        <div class="value">{r.get("overall_risk_score", 0):.1f}/100</div>
                    </div>
                    <div class="detail-item">
                        <div class="label">STT 신뢰도</div>
                        <div class="value">{r.get("stt_confidence_avg", 0):.3f}</div>
                    </div>
                    <div class="detail-item">
                        <div class="label">총 단어</div>
                        <div class="value">{r.get("stt_total_words", 0)}</div>
                    </div>
                    <div class="detail-item">
                        <div class="label">고유 단어</div>
                        <div class="value">{r.get("stt_unique_words", 0)}</div>
                    </div>
                    <div class="detail-item">
                        <div class="label">말하기 속도</div>
                        <div class="value">{r.get("stt_words_per_minute", 0):.0f} WPM</div>
                    </div>
                    <div class="detail-item">
                        <div class="label">한글 비율</div>
                        <div class="value">{r.get("stt_korean_ratio", 0):.1%}</div>
                    </div>
                    <div class="detail-item">
                        <div class="label">저신뢰도 단어</div>
                        <div class="value">{r.get("stt_low_conf_words", 0)}개</div>
                    </div>
"""

            # WER/CER 추가 (있는 경우)
            if r.get("stt_wer"):
                html += f"""
                    <div class="detail-item">
                        <div class="label">WER</div>
                        <div class="value">{r.get("stt_wer", 0):.3f}</div>
                    </div>
"""
            if r.get("stt_cer"):
                html += f"""
                    <div class="detail-item">
                        <div class="label">CER</div>
                        <div class="value">{r.get("stt_cer", 0):.3f}</div>
                    </div>
"""
            html += """
                </div>
"""

            # 카테고리 점수 바
            html += """
                <div class="category-bars" style="margin-top: 15px;">
"""
            categories_kr = {
                "가스라이팅": ("gaslighting_score", "#a855f7"),
                "협박": ("threat_score", "#ef4444"),
                "강요": ("coercion_score", "#f97316"),
                "사기": ("deception_score", "#eab308"),
                "감정조작": ("emotional_score", "#ec4899"),
            }
            for name, (key, color) in categories_kr.items():
                score = r.get(key, 0)
                html += f"""
                    <div class="category-bar" style="height: 16px;">
                        <div class="category-fill" style="width: {score}%; background: {color};"></div>
                        <div class="category-label" style="font-size: 0.65rem;">{name}: {score:.0f}</div>
                    </div>
"""
            html += """
                </div>
            </div>
"""
        html += """
        </div>

        <!-- 푸터 -->
        <div style="text-align: center; color: #64748b; padding: 20px;">
            <p>Voice.Man 포렌식 모니터링 시스템 | STT 정확도 분석 포함</p>
        </div>
    </div>
</body>
</html>
"""
        return html

    def generate_json_report(self, results: List[Dict[str, Any]], output_path: str) -> str:
        """
        결과 리스트를 JSON 파일로 저장

        Args:
            results: 분석 결과 딕셔너리 리스트
            output_path: 출력 JSON 파일 경로

        Returns:
            저장된 JSON 파일 경로
        """
        summary = self._calculate_summary(results)

        output_data = {
            "summary": summary,
            "results": results,
        }

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        return str(output_file)
