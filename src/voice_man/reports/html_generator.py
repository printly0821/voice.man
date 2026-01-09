"""
Forensic HTML Report Generator

Generates comprehensive HTML reports from forensic analysis JSON results
with Mermaid diagram visualization for long-term case analysis.

Features:
- Executive summary with Korean language support
- Category scores with visual charts
- Transcript with speaker segmentation
- Timeline visualization with Mermaid
- Gaslighting pattern analysis
- Long-term perspective view
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class ForensicHTMLGenerator:
    """
    Generate HTML forensic analysis reports with embedded Mermaid diagrams.

    Produces publication-ready HTML reports suitable for legal evidence
    documentation with proper Korean language rendering.
    """

    # Risk level color mapping
    RISK_COLORS = {
        "critical": "#dc2626",  # Red
        "high": "#ea580c",  # Orange
        "moderate": "#ca8a04",  # Yellow/amber
        "low": "#2563eb",  # Blue
        "minimal": "#16a34a",  # Green
    }

    # Risk level display names in Korean
    RISK_NAMES_KO = {
        "critical": "심각",
        "high": "높음",
        "moderate": "중간",
        "low": "낮음",
        "minimal": "최소",
    }

    # Category names in Korean
    CATEGORY_NAMES_KO = {
        "gaslighting": "가스라이팅",
        "threat": "협박",
        "coercion": "강압",
        "deception": "기만",
        "emotional_manipulation": "감정 조종",
    }

    def __init__(self, template_dir: Optional[Path] = None):
        """
        Initialize the HTML generator.

        Args:
            template_dir: Optional directory for custom HTML templates
        """
        self.template_dir = template_dir

    def generate_from_json(self, json_path: str, output_path: Optional[str] = None) -> str:
        """
        Generate HTML report from forensic analysis JSON file.

        Args:
            json_path: Path to forensic analysis JSON file
            output_path: Optional output HTML file path

        Returns:
            Generated HTML content as string

        Raises:
            FileNotFoundError: If JSON file not found
            ValueError: If JSON content is invalid
        """
        json_file = Path(json_path)
        if not json_file.exists():
            raise FileNotFoundError(f"Forensic JSON file not found: {json_path}")

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        return self.generate(data, output_path or str(json_file.with_suffix(".html")))

    def generate(self, data: Dict[str, Any], output_path: Optional[str] = None) -> str:
        """
        Generate HTML report from forensic analysis data.

        Args:
            data: Forensic analysis data dictionary
            output_path: Optional output HTML file path

        Returns:
            Generated HTML content as string
        """
        html_content = self._build_html(data)

        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(html_content)

        return html_content

    def _build_html(self, data: Dict[str, Any]) -> str:
        """
        Build complete HTML document from forensic data.

        Args:
            data: Forensic analysis data dictionary

        Returns:
            Complete HTML document as string
        """
        # Extract key information
        analysis_id = data.get("analysis_id", "unknown")
        analyzed_at = data.get("analyzed_at", "")
        audio_file = data.get("audio_file", "")
        audio_duration = data.get("audio_duration_seconds", 0)
        overall_score = data.get("overall_risk_score", 0)
        overall_level = data.get("overall_risk_level", "minimal")
        confidence = data.get("confidence_level", 0)
        summary = data.get("summary", "")
        recommendations = data.get("recommendations", [])
        flags = data.get("flags", [])
        category_scores = data.get("category_scores", [])
        gaslighting_analysis = data.get("gaslighting_analysis", {})
        threat_assessment = data.get("threat_assessment", {})
        deception_analysis = data.get("deception_analysis", {})
        transcript = data.get("transcript", {})

        # Format datetime
        if analyzed_at:
            try:
                dt = datetime.fromisoformat(analyzed_at.replace("Z", "+00:00"))
                formatted_date = dt.strftime("%Y년 %m월 %d일 %H:%M")
            except Exception:
                formatted_date = analyzed_at
        else:
            formatted_date = "알 수 없음"

        # Format duration
        minutes = int(audio_duration // 60)
        seconds = int(audio_duration % 60)
        duration_str = f"{minutes}분 {seconds}초"

        # Build HTML sections
        sections = []

        # Header section
        sections.append(self._build_header(analysis_id, formatted_date, audio_file, duration_str))

        # Executive summary
        sections.append(
            self._build_executive_summary(
                overall_score, overall_level, confidence, summary, recommendations, flags
            )
        )

        # Category scores
        sections.append(self._build_category_scores(category_scores))

        # Mermaid diagrams section
        sections.append(
            self._build_mermaid_diagrams(data, overall_score, overall_level, category_scores)
        )

        # Detailed analysis sections
        sections.append(self._build_gaslighting_analysis(gaslighting_analysis))
        sections.append(self._build_threat_assessment(threat_assessment))
        sections.append(self._build_deception_analysis(deception_analysis))

        # Transcript section
        if transcript:
            sections.append(self._build_transcript(transcript))

        # Long-term perspective section
        sections.append(self._build_long_term_perspective(data))

        # Footer
        sections.append(self._build_footer(analysis_id, formatted_date))

        # Combine all sections
        return self._wrap_html(sections, overall_level)

    def _wrap_html(self, sections: List[str], risk_level: str) -> str:
        """
        Wrap HTML sections in complete document structure.

        Args:
            sections: List of HTML section strings
            risk_level: Overall risk level for theming

        Returns:
            Complete HTML document
        """
        theme_color = self.RISK_COLORS.get(risk_level, self.RISK_COLORS["minimal"])

        html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>음성 포렌식 분석 보고서</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Apple SD Gothic Neo', 'Malgun Gothic', 'Nanum Gothic', -apple-system, BlinkMacSystemFont, sans-serif;
            line-height: 1.6;
            color: #1a1a1a;
            background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%);
            min-height: 100vh;
            padding: 20px;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }}

        .header {{
            background: linear-gradient(135deg, {theme_color} 0%, {self._darken_color(theme_color)} 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}

        .header h1 {{
            font-size: 28px;
            font-weight: 700;
            margin-bottom: 10px;
        }}

        .header .subtitle {{
            font-size: 14px;
            opacity: 0.9;
        }}

        .section {{
            padding: 30px 40px;
            border-bottom: 1px solid #eee;
        }}

        .section:last-child {{
            border-bottom: none;
        }}

        .section-title {{
            font-size: 20px;
            font-weight: 600;
            margin-bottom: 20px;
            color: #333;
            display: flex;
            align-items: center;
        }}

        .section-title::before {{
            content: '';
            width: 4px;
            height: 24px;
            background: {theme_color};
            margin-right: 12px;
            border-radius: 2px;
        }}

        .risk-badge {{
            display: inline-block;
            padding: 6px 16px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: 600;
            color: white;
            background: {theme_color};
        }}

        .score-display {{
            display: flex;
            align-items: baseline;
            gap: 8px;
            margin: 10px 0;
        }}

        .score-value {{
            font-size: 48px;
            font-weight: 700;
            color: {theme_color};
        }}

        .score-max {{
            font-size: 18px;
            color: #999;
        }}

        .category-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}

        .category-card {{
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            border-left: 4px solid #ddd;
        }}

        .category-card.critical {{ border-left-color: #dc2626; }}
        .category-card.high {{ border-left-color: #ea580c; }}
        .category-card.moderate {{ border-left-color: #ca8a04; }}
        .category-card.low {{ border-left-color: #2563eb; }}
        .category-card.minimal {{ border-left-color: #16a34a; }}

        .category-name {{
            font-size: 14px;
            color: #666;
            margin-bottom: 8px;
        }}

        .category-score {{
            font-size: 24px;
            font-weight: 600;
        }}

        .category-bar {{
            height: 6px;
            background: #e5e7eb;
            border-radius: 3px;
            margin-top: 10px;
            overflow: hidden;
        }}

        .category-bar-fill {{
            height: 100%;
            background: {theme_color};
            border-radius: 3px;
            transition: width 0.3s ease;
        }}

        .recommendation-list {{
            list-style: none;
            margin-top: 15px;
        }}

        .recommendation-list li {{
            padding: 12px 16px;
            background: #fffbeb;
            border-left: 4px solid #f59e0b;
            margin-bottom: 10px;
            border-radius: 4px;
        }}

        .recommendation-list li.critical {{
            background: #fef2f2;
            border-left-color: #dc2626;
        }}

        .transcript-container {{
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            max-height: 400px;
            overflow-y: auto;
        }}

        .transcript-segment {{
            margin-bottom: 15px;
            padding: 12px;
            background: white;
            border-radius: 6px;
        }}

        .speaker-label {{
            display: inline-block;
            padding: 4px 10px;
            background: #e5e7eb;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
            margin-bottom: 8px;
        }}

        .diagram-container {{
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
            text-align: center;
            overflow-x: auto;
        }}

        .flag-container {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 15px;
        }}

        .flag {{
            padding: 6px 12px;
            background: #fef3c7;
            border: 1px solid #f59e0b;
            border-radius: 6px;
            font-size: 13px;
            color: #92400e;
        }}

        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}

        .metric-box {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}

        .metric-label {{
            font-size: 12px;
            color: #666;
            margin-bottom: 5px;
        }}

        .metric-value {{
            font-size: 20px;
            font-weight: 600;
            color: {theme_color};
        }}

        .timeline-view {{
            position: relative;
            padding: 20px 0;
        }}

        .timeline-item {{
            display: flex;
            gap: 20px;
            margin-bottom: 20px;
        }}

        .timeline-time {{
            min-width: 80px;
            font-size: 14px;
            color: #666;
            text-align: right;
        }}

        .timeline-content {{
            flex: 1;
        }}

        .footer {{
            background: #1a1a1a;
            color: #999;
            padding: 30px 40px;
            text-align: center;
            font-size: 13px;
        }}

        @media print {{
            body {{ background: white; padding: 0; }}
            .container {{ box-shadow: none; border-radius: 0; }}
            .section {{ page-break-inside: avoid; }}
        }}

        .mermaid {{
            background: white;
        }}
    </style>
</head>
<body>
    <div class="container">
        {"".join(sections)}
    </div>

    <script>
        mermaid.initialize({{
            startOnLoad: true,
            theme: 'default',
            themeVariables: {{
                primaryColor: '{theme_color}',
                primaryTextColor: '#fff',
                primaryBorderColor: '{theme_color}',
                lineColor: '#666',
                secondaryColor: '#f8f9fa',
                tertiaryColor: '#eee'
            }}
        }});
    </script>
</body>
</html>"""
        return html

    def _darken_color(self, hex_color: str, factor: float = 0.8) -> str:
        """Darken a hex color by a factor."""
        hex_color = hex_color.lstrip("#")
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        r, g, b = int(r * factor), int(g * factor), int(b * factor)
        return f"#{r:02x}{g:02x}{b:02x}"

    def _build_header(self, analysis_id: str, date: str, audio_file: str, duration: str) -> str:
        """Build header section."""
        audio_name = Path(audio_file).name if audio_file else "알 수 없음"

        return f"""
        <div class="header">
            <h1>음성 포렌식 분석 보고서</h1>
            <p class="subtitle">Voice Forensic Analysis Report</p>
            <div style="margin-top: 20px; font-size: 14px;">
                <p><strong>분석 ID:</strong> {analysis_id}</p>
                <p><strong>분석 일시:</strong> {date}</p>
                <p><strong>오디오 파일:</strong> {audio_name}</p>
                <p><strong>녹음 길이:</strong> {duration}</p>
            </div>
        </div>
        """

    def _build_executive_summary(
        self,
        score: float,
        level: str,
        confidence: float,
        summary: str,
        recommendations: List[str],
        flags: List[str],
    ) -> str:
        """Build executive summary section."""
        level_ko = self.RISK_NAMES_KO.get(level, level)
        level_color = self.RISK_COLORS.get(level, self.RISK_COLORS["minimal"])

        rec_items = ""
        for rec in recommendations:
            priority = (
                "critical" if any(word in rec for word in ["즉시", "전문", "신고", "안전"]) else ""
            )
            rec_items += f'<li class="{priority}">{rec}</li>'

        flags_html = ""
        if flags:
            flags_html = "<div class='flag-container'>"
            for flag in flags:
                flag_display = flag.replace("_", " ").title()
                flags_html += f'<span class="flag">{flag_display}</span>'
            flags_html += "</div>"

        return f"""
        <div class="section">
            <h2 class="section-title">요약 (Executive Summary)</h2>

            <div style="display: flex; justify-content: space-between; align-items: start; flex-wrap: wrap; gap: 20px;">
                <div>
                    <span class="risk-badge" style="background: {level_color}; font-size: 18px;">
                        위험도: {level_ko}
                    </span>
                    <div class="score-display">
                        <span class="score-value">{score:.1f}</span>
                        <span class="score-max">/ 100</span>
                    </div>
                    <p style="color: #666; font-size: 14px;">
                        신뢰도: {confidence * 100:.1f}%
                    </p>
                </div>

                <div style="flex: 1; min-width: 300px;">
                    <p style="font-size: 16px; line-height: 1.8;">{summary}</p>
                </div>
            </div>

            {flags_html}

            <h3 style="margin-top: 30px; margin-bottom: 15px; font-size: 16px;">권장 사항</h3>
            <ul class="recommendation-list">
                {rec_items}
            </ul>
        </div>
        """

    def _build_category_scores(self, category_scores: List[Dict]) -> str:
        """Build category scores section."""
        cards = ""

        for cat in category_scores:
            category = cat.get("category", "")
            score = cat.get("score", 0)
            confidence = cat.get("confidence", 0)
            evidence_count = cat.get("evidence_count", 0)
            key_indicators = cat.get("key_indicators", [])

            cat_name = self.CATEGORY_NAMES_KO.get(category, category.title())
            cat_level = self._get_level_from_score(score)

            # Determine indicators text
            indicators_text = ""
            if key_indicators:
                indicators_ko = [self._translate_indicator(ind) for ind in key_indicators[:3]]
                indicators_text = f"<p style='font-size: 12px; color: #666; margin-top: 8px;'>주요 지표: {', '.join(indicators_ko)}</p>"

            cards += f"""
            <div class="category-card {cat_level}">
                <div class="category-name">{cat_name}</div>
                <div class="category-score">{score:.1f}%</div>
                <div class="category-bar">
                    <div class="category-bar-fill" style="width: {score}%"></div>
                </div>
                <p style="font-size: 12px; color: #999; margin-top: 8px;">
                    신뢰도: {confidence * 100:.0f}% | 증거: {evidence_count}건
                </p>
                {indicators_text}
            </div>
            """

        return f"""
        <div class="section">
            <h2 class="section-title">범주별 분석 (Category Analysis)</h2>
            <div class="category-grid">
                {cards}
            </div>
        </div>
        """

    def _build_mermaid_diagrams(
        self,
        data: Dict,
        overall_score: float,
        overall_level: str,
        category_scores: List[Dict],
    ) -> str:
        """Build Mermaid diagram visualizations."""
        risk_flow = self._generate_risk_flow_diagram(category_scores, overall_score, overall_level)
        timeline = self._generate_timeline_diagram(data)
        speaker_interaction = self._generate_speaker_interaction_diagram(data)

        return f"""
        <div class="section">
            <h2 class="section-title">시각화 분석 (Visualization)</h2>

            <h3 style="font-size: 16px; margin: 20px 0 10px;">위험도 흐름 (Risk Flow)</h3>
            <div class="diagram-container">
                <pre class="mermaid">
{risk_flow}
                </pre>
            </div>

            <h3 style="font-size: 16px; margin: 20px 0 10px;">대화 타임라인 (Conversation Timeline)</h3>
            <div class="diagram-container">
                <pre class="mermaid">
{timeline}
                </pre>
            </div>

            <h3 style="font-size: 16px; margin: 20px 0 10px;">화자 상호작용 (Speaker Interaction)</h3>
            <div class="diagram-container">
                <pre class="mermaid">
{speaker_interaction}
                </pre>
            </div>
        </div>
        """

    def _generate_risk_flow_diagram(
        self, category_scores: List[Dict], overall_score: float, overall_level: str
    ) -> str:
        """Generate Mermaid flowchart for risk assessment."""
        lines = ["graph TD"]

        # Add nodes for each category
        for i, cat in enumerate(category_scores):
            category = cat.get("category", "unknown")
            score = cat.get("score", 0)
            cat_name = self.CATEGORY_NAMES_KO.get(category, category.title())

            # Style based on score
            if score >= 70:
                style = "fill:#dc2626,stroke:#b91c1c,color:#fff"
            elif score >= 50:
                style = "fill:#ea580c,stroke:#c2410c,color:#fff"
            elif score >= 30:
                style = "fill:#ca8a04,stroke:#a16207,color:#fff"
            elif score >= 10:
                style = "fill:#2563eb,stroke:#1d4ed8,color:#fff"
            else:
                style = "fill:#16a34a,stroke:#15803d,color:#fff"

            lines.append(f'    Cat{i}["{cat_name}: {score:.1f}%"]:::cat{i}')
            lines.append(f"    style Cat{i} {style}")

        # Add overall risk node
        lines.append(
            f'    Overall["최종 위험도: {overall_score:.1f}% ({self.RISK_NAMES_KO.get(overall_level, overall_level)})"]:::overall'
        )

        # Connect categories to overall
        for i in range(len(category_scores)):
            weight = category_scores[i].get("score", 0) / 4
            lines.append(f"    Cat{i} -->|{weight:.1f}%| Overall")

        return "\n".join(lines)

    def _generate_timeline_diagram(self, data: Dict) -> str:
        """Generate Mermaid timeline/gantt chart for conversation."""
        transcript = data.get("transcript", {})
        text = transcript.get("text", "")
        duration = data.get("audio_duration_seconds", 0)

        lines = ["gantt", "    title 통화 분석 타임라인", "    dateFormat X", "    axisFormat %s"]

        # Create time sections
        if duration > 0:
            section_duration = duration / 4
            sections = ["초반부", "중반부 1", "중반부 2", "후반부"]

            for i, section in enumerate(sections):
                start = int(i * section_duration)
                end = int((i + 1) * section_duration)
                lines.append(f"    section {section}")
                lines.append(f"    대화 구간{i + 1} : {start}, {end}")

        return "\n".join(lines)

    def _generate_speaker_interaction_diagram(self, data: Dict) -> str:
        """Generate Mermaid graph for speaker interaction patterns."""
        transcript = data.get("transcript", {})
        speakers = transcript.get("speakers", [])

        if not speakers:
            speakers = ["SPEAKER_01", "SPEAKER_02"]

        lines = ["graph LR"]

        # Add speaker nodes
        for i, speaker in enumerate(speakers):
            if speaker == "UNKNOWN":
                continue
            display_name = speaker.replace("SPEAKER_", "화자 ")
            lines.append(f'    Speaker{i}["{display_name}"]')

        # Add interaction
        if len(speakers) >= 2:
            valid_speakers = [s for s in speakers if s != "UNKNOWN"]
            if len(valid_speakers) >= 2:
                lines.append("    Speaker0 <---> Speaker1")

        return "\n".join(lines)

    def _build_gaslighting_analysis(self, analysis: Dict) -> str:
        """Build gaslighting analysis section."""
        intensity = analysis.get("intensity_score", 0)
        patterns = analysis.get("patterns_detected", [])
        techniques = analysis.get("manipulation_techniques", [])
        victim_impact = analysis.get("victim_impact_level", "minimal")

        impact_ko = self.RISK_NAMES_KO.get(victim_impact, victim_impact)

        patterns_html = ""
        if patterns:
            patterns_ko = [self._translate_gaslighting_pattern(p) for p in patterns]
            patterns_html = f"<p>감지된 패턴: {', '.join(patterns_ko)}</p>"

        techniques_html = ""
        if techniques:
            techniques_ko = [self._translate_technique(t) for t in techniques]
            techniques_html = f"<p>사용된 기술: {', '.join(techniques_ko)}</p>"

        return f"""
        <div class="section">
            <h2 class="section-title">가스라이팅 분석 (Gaslighting Analysis)</h2>

            <div class="metrics-grid">
                <div class="metric-box">
                    <div class="metric-label">강도 점수</div>
                    <div class="metric-value">{intensity:.1f}%</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">피해자 영향</div>
                    <div class="metric-value">{impact_ko}</div>
                </div>
            </div>

            <div style="margin-top: 20px;">
                {patterns_html}
                {techniques_html}
            </div>
        </div>
        """

    def _build_threat_assessment(self, assessment: Dict) -> str:
        """Build threat assessment section."""
        threat_level = assessment.get("threat_level", "minimal")
        threat_types = assessment.get("threat_types", [])
        immediacy = assessment.get("immediacy", "long-term")
        specificity = assessment.get("specificity", "vague")
        credibility = assessment.get("credibility_score", 0)

        level_ko = self.RISK_NAMES_KO.get(threat_level, threat_level)

        immediacy_ko = {
            "immediate": "즉각적",
            "near-term": "단기적",
            "long-term": "장기적",
        }.get(immediacy, immediacy)

        specificity_ko = {
            "detailed": "상세함",
            "specific": "구체적",
            "vague": "모호함",
        }.get(specificity, specificity)

        types_html = ""
        if threat_types:
            types_ko = [self._translate_threat_type(t) for t in threat_types]
            types_html = f"<p>위협 유형: {', '.join(types_ko)}</p>"

        return f"""
        <div class="section">
            <h2 class="section-title">위협 평가 (Threat Assessment)</h2>

            <div class="metrics-grid">
                <div class="metric-box">
                    <div class="metric-label">위협 수준</div>
                    <div class="metric-value">{level_ko}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">긴급성</div>
                    <div class="metric-value">{immediacy_ko}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">구체성</div>
                    <div class="metric-value">{specificity_ko}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">신뢰도</div>
                    <div class="metric-value">{credibility * 100:.0f}%</div>
                </div>
            </div>

            <div style="margin-top: 20px;">
                {types_html}
            </div>
        </div>
        """

    def _build_deception_analysis(self, analysis: Dict) -> str:
        """Build deception analysis section."""
        probability = analysis.get("deception_probability", 0)
        voice_text_consistency = analysis.get("voice_text_consistency", 0)
        emotional_authenticity = analysis.get("emotional_authenticity", 0)
        markers_count = analysis.get("linguistic_markers_count", 0)
        behavioral_indicators = analysis.get("behavioral_indicators", [])

        indicators_html = ""
        if behavioral_indicators:
            indicators_ko = [self._translate_indicator(ind) for ind in behavioral_indicators]
            indicators_html = f"<p>행동 지표: {', '.join(indicators_ko)}</p>"

        return f"""
        <div class="section">
            <h2 class="section-title">기만 분석 (Deception Analysis)</h2>

            <div class="metrics-grid">
                <div class="metric-box">
                    <div class="metric-label">기만 확률</div>
                    <div class="metric-value">{probability * 100:.1f}%</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">음성-텍스트 일관성</div>
                    <div class="metric-value">{voice_text_consistency * 100:.1f}%</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">감정 진정성</div>
                    <div class="metric-value">{emotional_authenticity * 100:.1f}%</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">언어적 지표</div>
                    <div class="metric-value">{markers_count}개</div>
                </div>
            </div>

            <div style="margin-top: 20px;">
                {indicators_html}
            </div>
        </div>
        """

    def _build_transcript(self, transcript: Dict) -> str:
        """Build transcript section."""
        text = transcript.get("text", "")
        speakers = transcript.get("speakers", [])
        num_segments = transcript.get("num_segments", 0)

        # Simple display of full text
        # In production, would parse segments with timestamps and speaker labels
        return f"""
        <div class="section">
            <h2 class="section-title">대화 기록 (Transcript)</h2>

            <p style="color: #666; margin-bottom: 15px;">
                화자: {", ".join(speakers)} | 구간 수: {num_segments}
            </p>

            <div class="transcript-container">
                <p style="white-space: pre-wrap; line-height: 1.8;">{text}</p>
            </div>
        </div>
        """

    def _build_long_term_perspective(self, data: Dict) -> str:
        """Build long-term perspective section for case analysis."""
        overall_level = data.get("overall_risk_level", "minimal")
        gaslighting = data.get("gaslighting_analysis", {})
        threat = data.get("threat_assessment", {})

        # Generate long-term recommendations
        long_term_considerations = []

        if gaslighting.get("intensity_score", 0) >= 30:
            long_term_considerations.append(
                {
                    "category": "가스라이팅 장기 관찰",
                    "items": [
                        "반복적인 부정 패턴 기록 유지",
                        "자가 인식 저하 징후 모니터링",
                        "독립적인 현실 확인 확보",
                    ],
                }
            )

        if threat.get("threat_level") in ["high", "critical", "moderate"]:
            long_term_considerations.append(
                {
                    "category": "안전 계획 수립",
                    "items": [
                        "비상 연락망 구축",
                        "증거 자료 안전한 곳 보관",
                        "법적 지원 옵션 사전 파악",
                    ],
                }
            )

        long_term_considerations.append(
            {
                "category": "일반적 권장사항",
                "items": [
                    "정기적인 심리 상태 점검",
                    "신뢰할 수 있는 지원 시스템 유지",
                    "필요시 전문가 상담",
                ],
            }
        )

        sections_html = ""
        for consideration in long_term_considerations:
            items_html = "".join(f"<li>{item}</li>" for item in consideration["items"])
            sections_html += f"""
            <div style="margin-bottom: 20px;">
                <h4 style="font-size: 15px; margin-bottom: 10px; color: #333;">
                    {consideration["category"]}
                </h4>
                <ul style="padding-left: 20px; color: #555;">
                    {items_html}
                </ul>
            </div>
            """

        return f"""
        <div class="section">
            <h2 class="section-title">장기적 관점 (Long-Term Perspective)</h2>

            <p style="color: #666; margin-bottom: 20px;">
                단일 통화 분석 결과를 바탕으로 한 장기적 관점에서의 권장사항입니다.
                이것은 전문적인 법적 또는 심리적 조언을 대체하지 않습니다.
            </p>

            {sections_html}

            <div style="background: #fef3c7; border-left: 4px solid #f59e0b; padding: 15px; margin-top: 20px; border-radius: 4px;">
                <p style="font-size: 14px; color: #92400e;">
                    <strong>면책 조항:</strong> 이 보고서는 자동화된 시스템에 의해 생성된 분석 결과입니다.
                    법적 절차나 의사결정을 위해서는 반드시 자격을 갖춘 전문가의 검토가 필요합니다.
                </p>
            </div>
        </div>
        """

    def _build_footer(self, analysis_id: str, date: str) -> str:
        """Build footer section."""
        return f"""
        <div class="footer">
            <p>음성 포렌식 분석 시스템 (Voice Forensic Analysis System)</p>
            <p style="margin-top: 10px;">분석 ID: {analysis_id} | 생성일: {date}</p>
            <p style="margin-top: 5px; font-size: 11px; color: #666;">
                이 보고서는 자동화된 분석 시스템에 의해 생성되었습니다.
            </p>
        </div>
        """

    def _get_level_from_score(self, score: float) -> str:
        """Map score to risk level string."""
        if score >= 80:
            return "critical"
        elif score >= 60:
            return "high"
        elif score >= 40:
            return "moderate"
        elif score >= 20:
            return "low"
        else:
            return "minimal"

    def _translate_indicator(self, indicator: str) -> str:
        """Translate indicator to Korean."""
        translations = {
            "hedging language": "모호한 언어",
            "emotional incongruence": "감정 불일치",
            "voice-text inconsistency": "음성-텍스트 불일치",
            "distancing language": "거리두기 언어",
            "elevated stress": "높은 스트레스",
        }
        return translations.get(indicator.lower(), indicator)

    def _translate_gaslighting_pattern(self, pattern: str) -> str:
        """Translate gaslighting pattern to Korean."""
        translations = {
            "denial": "부정",
            "countering": "반박",
            "trivializing": "축소",
            "blame_shifting": "책임 전가",
        }
        return translations.get(pattern.lower(), pattern)

    def _translate_technique(self, technique: str) -> str:
        """Translate technique to Korean."""
        translations = {
            "reality distortion": "현실 왜곡",
            "emotional coercion": "감정적 강압",
        }
        return translations.get(technique.lower(), technique)

    def _translate_threat_type(self, threat_type: str) -> str:
        """Translate threat type to Korean."""
        translations = {
            "direct": "직접적",
            "conditional": "조건부",
            "veiled": "음담패술",
            "coercive": "강압적",
        }
        return translations.get(threat_type.lower(), threat_type)
