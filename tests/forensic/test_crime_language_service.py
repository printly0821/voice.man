"""
범죄 언어 분석 서비스 테스트
SPEC-FORENSIC-001 Phase 2-A: Crime Language Analysis Service

TDD RED Phase: CrimeLanguageAnalysisService 클래스 테스트
"""

import pytest
from typing import List, Dict


class TestCrimeLanguageAnalysisService:
    """범죄 언어 분석 서비스 테스트"""

    def test_service_initialization(self):
        """서비스 초기화 테스트"""
        from voice_man.services.forensic.crime_language_service import (
            CrimeLanguageAnalysisService,
        )

        service = CrimeLanguageAnalysisService()
        assert service is not None

    def test_detect_gaslighting_denial_pattern(self):
        """가스라이팅 부정 패턴 감지 테스트"""
        from voice_man.services.forensic.crime_language_service import (
            CrimeLanguageAnalysisService,
        )
        from voice_man.models.forensic.crime_language import GaslightingType

        service = CrimeLanguageAnalysisService()
        text = "그런 적 없어, 네가 잘못 기억하는 거야"

        matches = service.detect_gaslighting(text)

        assert len(matches) >= 1
        assert any(m.type == GaslightingType.DENIAL for m in matches)

    def test_detect_gaslighting_blame_shifting_pattern(self):
        """가스라이팅 책임전가 패턴 감지 테스트"""
        from voice_man.services.forensic.crime_language_service import (
            CrimeLanguageAnalysisService,
        )
        from voice_man.models.forensic.crime_language import GaslightingType

        service = CrimeLanguageAnalysisService()
        text = "네가 그렇게 만든 거야, 네 탓이야"

        matches = service.detect_gaslighting(text)

        assert len(matches) >= 1
        assert any(m.type == GaslightingType.BLAME_SHIFTING for m in matches)

    def test_detect_gaslighting_trivializing_pattern(self):
        """가스라이팅 최소화 패턴 감지 테스트"""
        from voice_man.services.forensic.crime_language_service import (
            CrimeLanguageAnalysisService,
        )
        from voice_man.models.forensic.crime_language import GaslightingType

        service = CrimeLanguageAnalysisService()
        text = "별거 아닌데 왜 그래, 너무 예민해"

        matches = service.detect_gaslighting(text)

        assert len(matches) >= 1
        assert any(m.type == GaslightingType.TRIVIALIZING for m in matches)

    def test_detect_threat_direct_pattern(self):
        """직접적 협박 패턴 감지 테스트"""
        from voice_man.services.forensic.crime_language_service import (
            CrimeLanguageAnalysisService,
        )
        from voice_man.models.forensic.crime_language import ThreatType

        service = CrimeLanguageAnalysisService()
        text = "두고 봐, 가만 안 둬"

        matches = service.detect_threats(text)

        assert len(matches) >= 1
        assert any(m.type == ThreatType.DIRECT for m in matches)

    def test_detect_threat_economic_pattern(self):
        """경제적 협박 패턴 감지 테스트"""
        from voice_man.services.forensic.crime_language_service import (
            CrimeLanguageAnalysisService,
        )
        from voice_man.models.forensic.crime_language import ThreatType

        service = CrimeLanguageAnalysisService()
        text = "돈 한 푼 못 받아, 다 빼앗을 거야"

        matches = service.detect_threats(text)

        assert len(matches) >= 1
        assert any(m.type == ThreatType.ECONOMIC for m in matches)

    def test_detect_coercion_emotional_pattern(self):
        """감정적 강압 패턴 감지 테스트"""
        from voice_man.services.forensic.crime_language_service import (
            CrimeLanguageAnalysisService,
        )
        from voice_man.models.forensic.crime_language import CoercionType

        service = CrimeLanguageAnalysisService()
        text = "나를 사랑하면 이렇게 안 하지, 나 없으면 어떻게 할 거야"

        matches = service.detect_coercion(text)

        assert len(matches) >= 1
        assert any(m.type == CoercionType.EMOTIONAL for m in matches)

    def test_detect_coercion_isolation_pattern(self):
        """고립화 패턴 감지 테스트"""
        from voice_man.services.forensic.crime_language_service import (
            CrimeLanguageAnalysisService,
        )
        from voice_man.models.forensic.crime_language import CoercionType

        service = CrimeLanguageAnalysisService()
        text = "그 친구 만나지 마, 그 사람들이 널 이용하는 거야"

        matches = service.detect_coercion(text)

        assert len(matches) >= 1
        assert any(m.type == CoercionType.ISOLATION for m in matches)

    def test_analyze_deception_markers(self):
        """기만 언어 지표 분석 테스트"""
        from voice_man.services.forensic.crime_language_service import (
            CrimeLanguageAnalysisService,
        )

        service = CrimeLanguageAnalysisService()
        text = "아마 그랬을 수도 있는데, 글쎄 정확히는 모르겠어. 그 사람이 그랬어."

        analysis = service.analyze_deception(text)

        assert analysis.hedging_score > 0
        assert analysis.distancing_score > 0
        assert 0 <= analysis.overall_deception_score <= 1.0

    def test_analyze_empty_text(self):
        """빈 텍스트 분석 테스트"""
        from voice_man.services.forensic.crime_language_service import (
            CrimeLanguageAnalysisService,
        )

        service = CrimeLanguageAnalysisService()

        gaslighting = service.detect_gaslighting("")
        threats = service.detect_threats("")
        coercion = service.detect_coercion("")
        deception = service.analyze_deception("")

        assert len(gaslighting) == 0
        assert len(threats) == 0
        assert len(coercion) == 0
        assert deception.overall_deception_score == 0.0

    def test_analyze_neutral_text(self):
        """중립적 텍스트 분석 테스트"""
        from voice_man.services.forensic.crime_language_service import (
            CrimeLanguageAnalysisService,
        )

        service = CrimeLanguageAnalysisService()
        text = "오늘 날씨가 좋네요. 점심 뭐 먹을까요?"

        gaslighting = service.detect_gaslighting(text)
        threats = service.detect_threats(text)
        coercion = service.detect_coercion(text)

        assert len(gaslighting) == 0
        assert len(threats) == 0
        assert len(coercion) == 0

    def test_comprehensive_analysis(self):
        """종합 분석 테스트"""
        from voice_man.services.forensic.crime_language_service import (
            CrimeLanguageAnalysisService,
        )
        from voice_man.models.forensic.crime_language import CrimeLanguageScore

        service = CrimeLanguageAnalysisService()
        text = """
        그런 적 없어, 네가 잘못 기억하는 거야.
        네가 그렇게 만든 거야.
        별거 아닌데 왜 그래.
        두고 봐.
        나를 사랑하면 이렇게 안 하지.
        """

        result = service.analyze_comprehensive(text)

        assert isinstance(result, CrimeLanguageScore)
        assert result.gaslighting_count >= 2
        assert result.threat_count >= 1
        assert result.coercion_count >= 1
        assert result.overall_risk_score > 0

    def test_calculate_risk_level(self):
        """위험 수준 계산 테스트"""
        from voice_man.services.forensic.crime_language_service import (
            CrimeLanguageAnalysisService,
        )

        service = CrimeLanguageAnalysisService()

        # Low risk
        assert service._calculate_risk_level(0.1) == "낮음"
        # Medium risk
        assert service._calculate_risk_level(0.4) == "중간"
        # High risk
        assert service._calculate_risk_level(0.7) == "높음"
        # Very high risk
        assert service._calculate_risk_level(0.9) == "매우 높음"

    def test_match_positions_are_correct(self):
        """매칭 위치 정확성 테스트"""
        from voice_man.services.forensic.crime_language_service import (
            CrimeLanguageAnalysisService,
        )

        service = CrimeLanguageAnalysisService()
        text = "처음에는 괜찮았는데 그런 적 없어 라고 했어"

        matches = service.detect_gaslighting(text)

        if matches:
            match = matches[0]
            # Verify the matched pattern is at the correct position
            extracted = text[match.start_position : match.end_position]
            assert match.matched_pattern in text
            assert extracted == match.matched_pattern

    def test_confidence_calculation(self):
        """신뢰도 계산 테스트"""
        from voice_man.services.forensic.crime_language_service import (
            CrimeLanguageAnalysisService,
        )

        service = CrimeLanguageAnalysisService()

        # Exact match should have higher confidence
        exact_text = "그런 적 없어"
        matches = service.detect_gaslighting(exact_text)

        if matches:
            assert matches[0].confidence >= 0.8

    def test_speaker_and_timestamp_assignment(self):
        """화자 및 타임스탬프 할당 테스트"""
        from voice_man.services.forensic.crime_language_service import (
            CrimeLanguageAnalysisService,
        )

        service = CrimeLanguageAnalysisService()
        text = "그런 적 없어"

        matches = service.detect_gaslighting(text, speaker="SPEAKER_01", timestamp="00:01:30")

        if matches:
            assert matches[0].speaker == "SPEAKER_01"
            assert matches[0].timestamp == "00:01:30"


class TestCrimeLanguageConversationAnalysis:
    """대화 분석 테스트"""

    def test_analyze_conversation(self):
        """대화 분석 테스트"""
        from voice_man.services.forensic.crime_language_service import (
            CrimeLanguageAnalysisService,
        )

        service = CrimeLanguageAnalysisService()
        conversation = [
            {"speaker": "A", "text": "그런 적 없어, 네가 잘못 기억하는 거야", "time": "00:00:10"},
            {"speaker": "B", "text": "아니, 분명히 그랬잖아", "time": "00:00:15"},
            {"speaker": "A", "text": "네가 그렇게 만든 거야, 네 탓이야", "time": "00:00:20"},
            {"speaker": "B", "text": "왜 내 탓이야?", "time": "00:00:25"},
            {"speaker": "A", "text": "두고 봐, 가만 안 둬", "time": "00:00:30"},
        ]

        result = service.analyze_conversation(conversation)

        assert result.gaslighting_count >= 2
        assert result.threat_count >= 1
        assert result.overall_risk_score > 0
        assert result.risk_level in ["낮음", "중간", "높음", "매우 높음"]

    def test_analyze_conversation_with_speaker_breakdown(self):
        """화자별 분석 결과 테스트"""
        from voice_man.services.forensic.crime_language_service import (
            CrimeLanguageAnalysisService,
        )

        service = CrimeLanguageAnalysisService()
        conversation = [
            {"speaker": "A", "text": "그런 적 없어", "time": "00:00:10"},
            {"speaker": "A", "text": "네 탓이야", "time": "00:00:20"},
            {"speaker": "B", "text": "아니야", "time": "00:00:15"},
        ]

        result = service.analyze_conversation_by_speaker(conversation)

        assert "A" in result
        assert result["A"]["gaslighting_count"] >= 2
        assert "B" in result
        assert result["B"]["gaslighting_count"] == 0


class TestCrimeLanguageReportGeneration:
    """보고서 생성 테스트"""

    def test_generate_analysis_report(self):
        """분석 보고서 생성 테스트"""
        from voice_man.services.forensic.crime_language_service import (
            CrimeLanguageAnalysisService,
        )

        service = CrimeLanguageAnalysisService()
        text = "그런 적 없어, 네가 그렇게 만든 거야. 두고 봐."

        report = service.generate_analysis_report(text)

        assert "gaslighting_matches" in report
        assert "threat_matches" in report
        assert "coercion_matches" in report
        assert "deception_analysis" in report
        assert "crime_language_score" in report
        assert "recommendations" in report

    def test_generate_recommendations(self):
        """권고사항 생성 테스트"""
        from voice_man.services.forensic.crime_language_service import (
            CrimeLanguageAnalysisService,
        )

        service = CrimeLanguageAnalysisService()

        # High risk scenario
        recommendations = service._generate_recommendations(
            gaslighting_count=5, threat_count=3, coercion_count=2, risk_level="매우 높음"
        )

        assert len(recommendations) > 0
        assert any("전문가" in r or "상담" in r for r in recommendations)

    def test_export_report_as_dict(self):
        """보고서 딕셔너리 내보내기 테스트"""
        from voice_man.services.forensic.crime_language_service import (
            CrimeLanguageAnalysisService,
        )

        service = CrimeLanguageAnalysisService()
        text = "그런 적 없어"

        report = service.generate_analysis_report(text)

        # Should be serializable to JSON
        import json

        json_str = json.dumps(report, ensure_ascii=False)
        assert len(json_str) > 0
