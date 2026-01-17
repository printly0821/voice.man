"""
Legal Report Generator for Forensic Evidence Submission

법정 제출용 포렌식 오디오 분석 보고서 생성 시스템을 구현합니다.

TAG: [FORENSIC-EVIDENCE-001]
Reference: SPEC-FORENSIC-EVIDENCE-001
Compliance: ISO/IEC 17025, Korean Criminal Procedure Law Article 313(2)(3)
"""

import base64
import hashlib
import io
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import jinja2
from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML, CSS

logger = logging.getLogger(__name__)


class LegalReportGenerator:
    """
    법정 제출용 포렌식 오디오 분석 보고서 생성기

    한국 법정 제출에 적합한 포맷으로 PDF 보고서를 생성합니다.
    """

    def __init__(
        self, template_dir: Optional[str] = None, output_dir: Optional[str] = None
    ):
        """
        법정 보고서 생성기 초기화

        Args:
            template_dir: Jinja2 템플릿 디렉토리
            output_dir: PDF 출력 디렉토리
        """
        # 템플릿 디렉토리 설정
        if template_dir is None:
            template_dir = Path(__file__).parent / "templates"
        self.template_dir = Path(template_dir)

        # 출력 디렉토리 설정
        if output_dir is None:
            output_dir = Path("/tmp/forensic_reports")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Jinja2 환경 설정
        self.env = Environment(
            loader=FileSystemLoader(self.template_dir),
            autoescape=True,
        )

        # 필터 등록
        self.env.filters["korean_date_format"] = self._korean_date_format
        self.env.filters["format_number"] = self._format_number

        logger.info(
            f"LegalReportGenerator initialized: templates={self.template_dir}, "
            f"output={self.output_dir}"
        )

    def _korean_date_format(self, value, format_str="%Y년 %m월 %d일 %H시%M분%S초"):
        """한국어 날짜 형식으로 변환"""
        if value is None:
            return ""
        if isinstance(value, str):
            try:
                dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except:
                return value
        else:
            dt = value

        return dt.strftime(format_str)

    def _format_number(self, value, decimal_places=0):
        """숫자를 천 단위 구분 포맷으로 변환 (한국어)"""
        if value is None:
            return ""
        try:
            return f"{value:,.{decimal_places}f}"
        except (TypeError, ValueError):
            return str(value)

    def _prepare_analysis_results(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        분석 결과를 템플릿에 맞게 구조화

        Args:
            analysis_results: 원본 분석 결과

        Returns:
            Dict: 구조화된 분석 결과 (scores 리스트 포함)
        """
        # scores 리스트가 없으면 생성
        if "scores" not in analysis_results:
            scores = []

            # forensic_score에서 score 카드 생성
            forensic_score = analysis_results.get("forensic_score")
            if forensic_score is not None:
                scores.append({
                    "title": "진정성 점수",
                    "value": f"{forensic_score}/100"
                })

            # crime_detected에서 score 카드 생성
            crime_detected = analysis_results.get("crime_detected")
            if crime_detected is not None:
                score_value = "검출됨" if crime_detected else "미검출"
                scores.append({
                    "title": "범죄어 발견",
                    "value": score_value
                })

            # emotional_state에서 score 카드 생성
            emotional_state = analysis_results.get("emotional_state")
            if emotional_state:
                scores.append({
                    "title": "감정 상태",
                    "value": emotional_state
                })

            # 기존 scores가 있으면 병합
            analysis_results = dict(analysis_results)  # 원본 수정 방지
            analysis_results["scores"] = scores

        return analysis_results

    def generate_forensic_report(
        self,
        evidence_data: Dict[str, Any],
        analysis_results: Dict[str, Any],
        custody_chain: List[Dict[str, Any]],
        personnel_info: Dict[str, Any],
        template_name: str = "legal_report_full.html",
    ) -> bytes:
        """
        포렌식 오디오 분석 보고서 PDF 생성

        Args:
            evidence_data: 증거 기본 정보
            analysis_results: 분석 결과 (스코어, 패턴, 감정 등)
            custody_chain: Chain of Custody 이력
            personnel_info: 분석자 정보
            template_name: 사용할 템플릿 파일명

        Returns:
            bytes: PDF 파일의 바이너리 데이터
        """
        # 템플릿 렌더링
        template = self.env.get_template(template_name)

        # 컨텍스트 데이터 준비
        context = {
            # 보고서 식별자
            "report_id": str(uuid4()),
            "generation_date": datetime.now(timezone.utc),
            "generation_date_korean": datetime.now(timezone.utc).strftime("%Y년 %m월 %d일 %H시%M분%S초"),
            "evidence_number": evidence_data.get("evidence_number", ""),
            "case_number": evidence_data.get("case_number", ""),
            "original_filename": evidence_data.get("original_filename", ""),
            "file_hash": evidence_data.get("file_hash", ""),
            "file_size": evidence_data.get("file_size", 0),
            "audio_duration": evidence_data.get("audio_duration", 0),
            # 분석자 정보
            "analyst_name": personnel_info.get("name", ""),
            "analyst_credential": personnel_info.get("credential", ""),
            "analyst_organization": personnel_info.get("organization", ""),
            "analyst_contact": personnel_info.get("contact", ""),
            # 분석 결과 (scores 리스트 구조화)
            "analysis_results": self._prepare_analysis_results(analysis_results),
            # Chain of Custody
            "custody_chain": custody_chain,
            # 법적 고지사항
            "legal_disclaimer": "본 보고서는 디지털 증거의 무결성과 진정성을 입증하기 위하여 작성되었습니다.",
            "compliance_standards": [
                "한국 형사소송법 제313조(2)(3)",
                "ISO/IEC 27037 - 디지털 증거 수집/보존",
                "ISO/IEC 17025 - 포렌식 실험실 인정",
                "NIST SP 800-86 - 디지털 포렌식 가이드",
            ],
            # 서명 정보
            "has_digital_signature": evidence_data.get("digital_signature") is not None,
            "has_timestamp": evidence_data.get("timestamp_rfc3161") is not None,
            "digital_signature": evidence_data.get("digital_signature", ""),
            "timestamp_rfc3161": evidence_data.get("timestamp_rfc3161", ""),
        }

        # HTML 렌더링
        html_content = template.render(**context)

        # WeasyPrint를 사용하여 PDF 생성
        pdf_bytes = self._html_to_pdf(html_content)

        return pdf_bytes

    def _html_to_pdf(self, html_content: str, css_string: str = None) -> bytes:
        """
        HTML 내용을 PDF로 변환

        Args:
            html_content: HTML 문자열
            css_string: CSS 스타일 문자열 (선택)

        Returns:
            bytes: PDF 바이너리 데이터
        """
        # CSS 설정
        if css_string is None:
            css_string = self._get_default_css()

        # WeasyPrint를 사용하여 PDF 생성
        pdf_bytes = HTML(string=html_content).write_pdf(
            stylesheets=[CSS(string=css_string)]
        )

        return pdf_bytes

    def _get_default_css(self) -> str:
        """
        기본 CSS 스타일 (한국어 폰트 지원)
        """
        return """
        @page {
            size: A4;
            margin: 2.5cm 2cm 2cm 2cm;

            @top-center {
                content: "비밀 (CONFIDENTIAL) - 법정 제출용";
                font-size: 9pt;
                color: #666;
            }

            @bottom-center {
                content: "Page " counter(page) " of " counter(pages);
                font-size: 9pt;
                color: #666;
            }
        }

        body {
            font-family: 'Noto Sans CJK KR', 'NanumGothic', sans-serif;
            line-height: 1.6;
            color: #333;
        }

        h1, h2, h3 {
            color: #333;
            margin-top: 1.5em;
            margin-bottom: 0.8em;
        }

        h1 { font-size: 24pt; }
        h2 { font-size: 18pt; border-bottom: 2px solid #333; padding-bottom: 5px; }
        h3 { font-size: 16pt; }

        .hash-display {
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            word-break: break-all;
            background-color: #f5f5f5;
            padding: 8px;
            border-radius: 4px;
        }
        """

    def save_report(
        self,
        evidence_data: Dict[str, Any],
        analysis_results: Dict[str, Any],
        custody_chain: List[Dict[str, Any]],
        personnel_info: Dict[str, Any],
        output_filename: Optional[str] = None,
    ) -> str:
        """
        보고서 생성 및 저장

        Args:
            evidence_data: 증거 기본 정보
            analysis_results: 분석 결과
            custody_chain: Chain of Custody 이력
            personnel_info: 분석자 정보
            output_filename: 출력 파일명 (선택, 자동 생성됨)

        Returns:
            str: 저장된 PDF 파일 경로
        """
        # PDF 생성
        pdf_bytes = self.generate_forensic_report(
            evidence_data=evidence_data,
            analysis_results=analysis_results,
            custody_chain=custody_chain,
            personnel_info=personnel_info,
        )

        # 파일명 생성
        if output_filename is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            output_filename = f"{evidence_data.get('evidence_number', timestamp)}_report.pdf"

        # PDF 파일 저장
        output_path = self.output_dir / output_filename
        with open(output_path, "wb") as f:
            f.write(pdf_bytes)

        # 메타데이터 저장
        report_id = str(uuid4())
        self.save_report_metadata(
            report_id=report_id,
            file_path=str(output_path),
            metadata={
                "evidence_number": evidence_data.get("evidence_number"),
                "case_number": evidence_data.get("case_number"),
                "original_filename": evidence_data.get("original_filename"),
                "file_hash": evidence_data.get("file_hash"),
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "analysis_summary": {
                    "forensic_score": analysis_results.get("forensic_score"),
                    "crime_detected": analysis_results.get("crime_detected"),
                    "emotional_state": analysis_results.get("emotional_state"),
                },
            },
        )

        logger.info(f"Report saved: {output_path}")
        logger.info(f"Metadata saved: {report_id}_metadata.json")

        return str(output_path)

    def save_report_metadata(
        self,
        report_id: str,
        file_path: str,
        metadata: Dict[str, Any],
    ) -> str:
        """
        보고서 메타데이터 저장

        Args:
            report_id: 보고서 ID
            file_path: PDF 파일 경로
            metadata: 보고서 메타데이터

        Returns:
            str: 메타데이터 파일 경로
        """
        metadata_file = self.output_dir / f"{report_id}_metadata.json"

        metadata["report_id"] = report_id
        metadata["file_path"] = file_path
        metadata["created_at"] = datetime.now(timezone.utc).isoformat()

        import json

        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        logger.info(f"Report metadata saved: {metadata_file}")
        return str(metadata_file)


# Convenience function for quick report generation
def generate_legal_report(
    evidence_data: Dict[str, Any],
    analysis_results: Dict[str, Any],
    custody_chain: List[Dict[str, Any]],
    personnel_info: Dict[str, Any],
) -> bytes:
    """
    법정 보고서 생성 편의 함수

    Args:
        evidence_data: 증거 기본 정보
        analysis_results: 분석 결과
        custody_chain: Chain of Custody 이력
        personnel_info: 분석자 정보

    Returns:
        bytes: HTML/PDF 데이터
    """
    generator = LegalReportGenerator()
    return generator.generate_forensic_report(
        evidence_data=evidence_data,
        analysis_results=analysis_results,
        custody_chain=custody_chain,
        personnel_info=personnel_info,
    )
