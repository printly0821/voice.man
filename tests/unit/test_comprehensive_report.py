"""
종합 보고서 생성 서비스 테스트

TASK-005: HTML 보고서 생성 및 PDF 변환 테스트
"""

import pytest
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch

from voice_man.services.comprehensive_report_service import ComprehensiveReportService
from voice_man.services.analysis_pipeline_service import AnalysisResult


class TestComprehensiveReportService:
    """종합 보고서 서비스 테스트"""

    @pytest.fixture
    def service(self, tmp_path):
        """서비스 인스턴스 fixture"""
        return ComprehensiveReportService(output_dir=tmp_path)

    @pytest.fixture
    def sample_analysis_results(self):
        """샘플 분석 결과 fixture"""
        return [
            AnalysisResult(
                file_path="/test/audio1.m4a",
                status="success",
                transcription={"text": "안녕하세요 테스트입니다", "language": "ko", "segments": []},
                crime_tags=[],
                gaslighting_patterns=[],
                emotions=[],
            ),
            AnalysisResult(
                file_path="/test/audio2.m4a",
                status="failed",
                error="Transcription failed",
            ),
        ]

    def test_initialization(self, service, tmp_path):
        """초기화 테스트"""
        assert service.output_dir == tmp_path
        assert service.jinja_env is not None
        # template_dir은 존재하지 않을 수 있음 (내장 템플릿 사용)
        assert service.template_dir is not None

    def test_collect_analysis_results_success(self, service, sample_analysis_results):
        """분석 결과 수집 테스트"""
        summary = service.collect_analysis_results(sample_analysis_results)

        assert summary["total_files"] == 2
        assert summary["successful"] == 1
        assert summary["failed"] == 1
        assert summary["partial"] == 0
        assert len(summary["files"]) == 2
        assert summary["success_rate"] == 0.5

    def test_collect_analysis_results_with_crime_tags(self, service):
        """범죄 태그 포함 결과 수집 테스트"""
        # Mock crime tag
        mock_tag = Mock()
        mock_tag.type.value = "협박"
        mock_tag.confidence = 0.85
        mock_tag.keywords = ["죽여버린다"]

        result = AnalysisResult(
            file_path="/test/audio.m4a",
            status="success",
            transcription={"text": "죽여버린다", "language": "ko", "segments": []},
            crime_tags=[mock_tag],
            gaslighting_patterns=[],
            emotions=[],
        )

        summary = service.collect_analysis_results([result])

        assert summary["successful"] == 1
        assert summary["crime_statistics"]["total_tags"] == 1
        assert "협박" in summary["crime_statistics"]["by_type"]

    def test_collect_analysis_results_with_gaslighting(self, service):
        """가스라이팅 패턴 포함 결과 수집 테스트"""
        # Mock gaslighting pattern
        mock_pattern = Mock()
        mock_pattern.type.value = "부정"
        mock_pattern.confidence = 0.8
        mock_pattern.intensity = 0.6

        result = AnalysisResult(
            file_path="/test/audio.m4a",
            status="success",
            transcription={"text": "그런 적 없어", "language": "ko", "segments": []},
            crime_tags=[],
            gaslighting_patterns=[mock_pattern],
            emotions=[],
        )

        summary = service.collect_analysis_results([result])

        assert summary["gaslighting_statistics"]["total_patterns"] == 1
        assert "부정" in summary["gaslighting_statistics"]["by_type"]

    def test_generate_html_report(self, service, sample_analysis_results):
        """HTML 보고서 생성 테스트"""
        html_content = service.generate_html_report(sample_analysis_results)

        assert isinstance(html_content, str)
        assert len(html_content) > 0
        assert "<!DOCTYPE html>" in html_content
        assert "음성 녹취 분석 보고서" in html_content
        assert "전체 파일:" in html_content
        assert "성공:" in html_content

    def test_save_html_report(self, service, tmp_path):
        """HTML 보고서 저장 테스트"""
        html_content = "<html><body>Test Report</body></html>"
        output_path = tmp_path / "test_report.html"

        service.save_html_report(html_content, output_path)

        assert output_path.exists()
        with open(output_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert content == html_content

    def test_convert_to_pdf_success(self, service, tmp_path):
        """PDF 변환 성공 테스트"""
        # Mock WeasyPrint HTML
        with patch("weasyprint.HTML") as mock_html_class:
            mock_html = Mock()
            mock_html_class.return_value = mock_html

            html_content = "<html><body>Test</body></html>"
            output_path = tmp_path / "test_report.pdf"

            service.convert_to_pdf(html_content, output_path)

            # WeasyPrint가 호출되었는지 확인
            mock_html_class.assert_called_once_with(string=html_content)
            mock_html.write_pdf.assert_called_once_with(str(output_path))

    def test_convert_to_pdf_no_weasyprint(self, service, tmp_path):
        """WeasyPrint 미설치 테스트"""
        with patch.dict("sys.modules", {"weasyprint": None}):
            # Import를 제거하여 WeasyPrint가 없는 것처럼 동작
            import sys

            weasyprint_backup = sys.modules.get("weasyprint")
            if "weasyprint" in sys.modules:
                del sys.modules["weasyprint"]

            try:
                html_content = "<html><body>Test</body></html>"
                output_path = tmp_path / "test_report.pdf"

                with pytest.raises(RuntimeError, match="WeasyPrint required"):
                    service.convert_to_pdf(html_content, output_path)
            finally:
                # 복구
                if weasyprint_backup:
                    sys.modules["weasyprint"] = weasyprint_backup

    def test_generate_comprehensive_report_html_only(
        self, service, sample_analysis_results, tmp_path
    ):
        """HTML만 생성하는 종합 보고서 테스트"""
        result_paths = service.generate_comprehensive_report(
            sample_analysis_results,
            output_prefix="test_report",
            generate_pdf=False,
        )

        assert "html" in result_paths
        assert "pdf" not in result_paths

        html_path = Path(result_paths["html"])
        assert html_path.exists()
        assert html_path.name.startswith("test_report_")
        assert html_path.suffix == ".html"

        # HTML 내용 확인
        with open(html_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "<!DOCTYPE html>" in content

    def test_generate_comprehensive_report_with_pdf(
        self, service, sample_analysis_results, tmp_path
    ):
        """HTML + PDF 생성 테스트"""
        # Mock WeasyPrint
        with patch("weasyprint.HTML") as mock_html_class:
            mock_html = Mock()
            mock_html_class.return_value = mock_html

            result_paths = service.generate_comprehensive_report(
                sample_analysis_results,
                output_prefix="test_report",
                generate_pdf=True,
            )

            assert "html" in result_paths
            assert "pdf" in result_paths

            html_path = Path(result_paths["html"])
            pdf_path = Path(result_paths["pdf"])

            assert html_path.exists()
            assert pdf_path.name.startswith("test_report_")
            assert pdf_path.suffix == ".pdf"

    def test_get_html_template(self, service):
        """HTML 템플릿 조회 테스트"""
        template = service._get_html_template()

        assert isinstance(template, str)
        assert "{{ report_title }}" in template
        assert "{{ summary.total_files }}" in template
        assert "{% for file in summary.files %}" in template

    def test_get_default_css(self, service):
        """기본 CSS 조회 테스트"""
        css = service._get_default_css()

        assert isinstance(css, str)
        assert ".container" in css
        assert "header" in css
        assert ".file-item" in css

    def test_html_template_includes_crime_section(self, service):
        """HTML 템플릿에 범죄 분석 섹션 포함 테스트"""
        # 범죄 태그가 있는 결과 생성
        mock_tag = Mock()
        mock_tag.type.value = "협박"
        mock_tag.confidence = 0.85
        mock_tag.keywords = ["죽여버린다"]

        result = AnalysisResult(
            file_path="/test/audio.m4a",
            status="success",
            transcription={"text": "죽여버린다", "language": "ko", "segments": []},
            crime_tags=[mock_tag],
            gaslighting_patterns=[],
            emotions=[],
        )

        html_content = service.generate_html_report([result])

        assert "범죄 발언 분석" in html_content
        assert "협박" in html_content

    def test_html_template_includes_gaslighting_section(self, service):
        """HTML 템플릿에 가스라이팅 섹션 포함 테스트"""
        # 가스라이팅 패턴이 있는 결과 생성
        mock_pattern = Mock()
        mock_pattern.type.value = "부정"
        mock_pattern.confidence = 0.8
        mock_pattern.intensity = 0.6

        result = AnalysisResult(
            file_path="/test/audio.m4a",
            status="success",
            transcription={"text": "그런 적 없어", "language": "ko", "segments": []},
            crime_tags=[],
            gaslighting_patterns=[mock_pattern],
            emotions=[],
        )

        html_content = service.generate_html_report([result])

        assert "가스라이팅 패턴 분석" in html_content
        assert "부정" in html_content


class TestIntegration:
    """통합 테스트"""

    @pytest.fixture
    def service(self, tmp_path):
        return ComprehensiveReportService(output_dir=tmp_path)

    def test_full_report_generation_workflow(self, service, tmp_path):
        """전체 보고서 생성 워크플로우 테스트"""
        # Mock WeasyPrint
        with patch("weasyprint.HTML") as mock_html_class:
            mock_html = Mock()
            mock_html_class.return_value = mock_html

            # 복합적인 분석 결과 생성
            mock_tag = Mock()
            mock_tag.type.value = "협박"
            mock_tag.confidence = 0.85
            mock_tag.keywords = ["죽여버린다"]

            mock_pattern = Mock()
            mock_pattern.type.value = "부정"
            mock_pattern.confidence = 0.8
            mock_pattern.intensity = 0.6

            mock_emotion = Mock()
            mock_emotion.primary_emotion.value = "anger"
            mock_emotion.intensity = 0.7
            mock_emotion.confidence = 0.75

            results = [
                AnalysisResult(
                    file_path="/test/audio1.m4a",
                    status="success",
                    transcription={
                        "text": "죽여버린다 그런 적 없어",
                        "language": "ko",
                        "segments": [],
                    },
                    crime_tags=[mock_tag],
                    gaslighting_patterns=[mock_pattern],
                    emotions=[mock_emotion],
                ),
                AnalysisResult(
                    file_path="/test/audio2.m4a",
                    status="success",
                    transcription={"text": "안녕하세요", "language": "ko", "segments": []},
                    crime_tags=[],
                    gaslighting_patterns=[],
                    emotions=[],
                ),
            ]

            # 보고서 생성
            result_paths = service.generate_comprehensive_report(
                results,
                output_prefix="integration_test",
                generate_pdf=True,
            )

            # 검증
            assert "html" in result_paths
            assert "pdf" in result_paths

            # HTML 파일 확인
            html_path = Path(result_paths["html"])
            assert html_path.exists()

            with open(html_path, "r", encoding="utf-8") as f:
                html_content = f.read()

            # HTML 내용에 모든 섹션이 포함되어 있는지 확인
            assert "전체 파일:" in html_content
            assert "성공:" in html_content
            assert "범죄 발언 분석" in html_content
            assert "가스라이팅 패턴 분석" in html_content
            assert "협박" in html_content
            assert "부정" in html_content
