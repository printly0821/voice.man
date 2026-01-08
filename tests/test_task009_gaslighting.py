"""
TASK-009: 가스라이팅 패턴 감지 시스템 테스트

테스트 케이스:
1. 부정(Denial) 패턴 감지
2. 전가(Blame-shifting) 패턴 감지
3. 축소(Minimizing) 패턴 감지
4. 혼란(Confusion) 유발 패턴 감지
5. 패턴 빈도 및 강도 분석
6. 대화 전체 패턴 추적
7. 복합 패턴 분석
"""

import pytest
from voice_man.services.gaslighting_service import (
    GaslightingService,
    GaslightingPattern,
    GaslightingPatternType,
    GaslightingAnalysisResult,
)


class TestGaslightingPatternDetection:
    """가스라이팅 패턴 감지 테스트"""

    def test_detect_denial_pattern(self):
        """부정(Denial) 패턴 감지"""
        service = GaslightingService()
        text = "그런 적 없어. 내가 언제 그랬어?"

        patterns = service.detect_patterns(text)

        assert len(patterns) > 0
        assert any(p.type == GaslightingPatternType.DENIAL for p in patterns)
        denial_pattern = next(p for p in patterns if p.type == GaslightingPatternType.DENIAL)
        assert denial_pattern.confidence >= 0.7

    def test_detect_denial_multiple_occurrences(self):
        """부정 패턴 다발 감지"""
        service = GaslightingService()

        test_cases = [
            "그런 적 없어",
            "네가 잘못 기억하는 거야",
            "내가 언제 그랬어",
            "상상한 거겠지",
            "그건 없었던 일이야",
        ]

        for text in test_cases:
            patterns = service.detect_patterns(text)
            assert any(p.type == GaslightingPatternType.DENIAL for p in patterns)

    def test_detect_blame_shifting_pattern(self):
        """전가(Blame-shifting) 패턴 감지"""
        service = GaslightingService()
        text = "네 때문에 이렇게 됐잖아. 네가 먼저 그랬어."

        patterns = service.detect_patterns(text)

        assert len(patterns) > 0
        assert any(p.type == GaslightingPatternType.BLAME_SHIFTING for p in patterns)

    def test_detect_minimizing_pattern(self):
        """축소(Minimizing) 패턴 감지"""
        service = GaslightingService()
        text = "별거 아닌데 왜 그래? 과민반응하지 마."

        patterns = service.detect_patterns(text)

        assert len(patterns) > 0
        assert any(p.type == GaslightingPatternType.MINIMIZING for p in patterns)

    def test_detect_confusion_pattern(self):
        """혼란(Confusion) 유발 패턴 감지"""
        service = GaslightingService()
        text = "아까는 그렇다고 했는데, 지금은 말이 안 되네. 뭐가 진짜야?"

        patterns = service.detect_patterns(text)

        assert len(patterns) > 0
        assert any(p.type == GaslightingPatternType.CONFUSION for p in patterns)


class TestPatternFrequencyAnalysis:
    """패턴 빈도 및 강도 분석 테스트"""

    def test_count_pattern_frequency(self):
        """패턴 빈도 계산"""
        service = GaslightingService()

        conversation = [
            {"speaker": "Speaker A", "text": "그런 적 없어", "time": "00:01:00"},
            {"speaker": "Speaker A", "text": "네가 잘못 기억하는 거야", "time": "00:03:00"},
            {"speaker": "Speaker A", "text": "내가 언제 그랬어", "time": "00:05:00"},
        ]

        result = service.analyze_conversation(conversation)

        denial_count = sum(1 for p in result.patterns if p.type == GaslightingPatternType.DENIAL)
        assert denial_count == 3
        assert result.pattern_frequency[GaslightingPatternType.DENIAL] == 3

    def test_calculate_pattern_intensity(self):
        """패턴 강도 분석"""
        service = GaslightingService()

        # 강한 부정
        strong_text = "그런 적 절대 없어! 넌 계속 거짓말만 해!"
        patterns_strong = service.detect_patterns(strong_text)

        # 약한 부정
        weak_text = "그런 건 기억 안 나네"
        patterns_weak = service.detect_patterns(weak_text)

        strong_denial = next(
            (p for p in patterns_strong if p.type == GaslightingPatternType.DENIAL), None
        )
        weak_denial = next(
            (p for p in patterns_weak if p.type == GaslightingPatternType.DENIAL), None
        )

        if strong_denial and weak_denial:
            assert strong_denial.intensity > weak_denial.intensity


class TestConversationTracking:
    """대화 전체 패턴 추적 테스트"""

    def test_track_patterns_across_conversation(self):
        """대화 전체 패턴 추적"""
        service = GaslightingService()

        conversation = [
            {"speaker": "Speaker A", "text": "그런 적 없어", "time": "00:01:00"},
            {"speaker": "Speaker B", "text": "있었는데?", "time": "00:02:00"},
            {"speaker": "Speaker A", "text": "네가 잘못 기억하는 거야", "time": "00:03:00"},
            {"speaker": "Speaker A", "text": "네 때문에 이렇게 됐잖아", "time": "00:04:00"},
            {"speaker": "Speaker B", "text": "뭐라고?", "time": "00:05:00"},
            {"speaker": "Speaker A", "text": "별거 아닌데 왜 그래", "time": "00:06:00"},
        ]

        result = service.analyze_conversation(conversation)

        # Speaker A의 패턴 확인
        speaker_a_patterns = [p for p in result.patterns if p.speaker == "Speaker A"]
        assert len(speaker_a_patterns) >= 3

        # 다양한 패턴 유형 확인
        pattern_types = set(p.type for p in speaker_a_patterns)
        assert GaslightingPatternType.DENIAL in pattern_types
        assert (
            GaslightingPatternType.BLAME_SHIFTING in pattern_types
            or GaslightingPatternType.MINIMIZING in pattern_types
        )

    def test_group_patterns_by_speaker(self):
        """화자별 패턴 그룹화"""
        service = GaslightingService()

        conversation = [
            {"speaker": "Speaker A", "text": "그런 적 없어", "time": "00:01:00"},
            {"speaker": "Speaker B", "text": "말도 안 돼", "time": "00:02:00"},
            {"speaker": "Speaker A", "text": "네가 잘못 기억해", "time": "00:03:00"},
            {"speaker": "Speaker B", "text": "거짓말하지 마", "time": "00:04:00"},
        ]

        result = service.analyze_conversation(conversation)

        # 화자별 패턴 그룹화 확인
        assert "Speaker A" in result.speaker_patterns
        assert "Speaker B" in result.speaker_patterns


class TestComplexPatternAnalysis:
    """복합 패턴 분석 테스트"""

    def test_detect_multiple_patterns_in_single_utterance(self):
        """단일 발언 내 복수 패턴 감지"""
        service = GaslightingService()
        text = "그런 적 없어(부정). 네 때문에 그런 거야(전가)."

        patterns = service.detect_patterns(text)

        # 최소 2개 패턴 감지
        assert len(patterns) >= 2

    def test_comprehensive_pattern_analysis(self):
        """종합 패턴 분석 (부정 + 전가 + 축소 + 혼란)"""
        service = GaslightingService()

        conversation = [
            {"speaker": "Speaker A", "text": "그런 적 없어", "time": "00:01:00"},  # 부정
            {"speaker": "Speaker A", "text": "네 때문에 이렇게 됐어", "time": "00:02:00"},  # 전가
            {"speaker": "Speaker A", "text": "별거 아닌데", "time": "00:03:00"},  # 축소
            {
                "speaker": "Speaker A",
                "text": "아까는 그랬는데 지금은 말이 안 되네",
                "time": "00:04:00",
            },  # 혼란
        ]

        result = service.analyze_conversation(conversation)

        # 모든 패턴 유형 감지 확인
        pattern_types = set(p.type for p in result.patterns)
        assert GaslightingPatternType.DENIAL in pattern_types
        assert GaslightingPatternType.BLAME_SHIFTING in pattern_types
        assert GaslightingPatternType.MINIMIZING in pattern_types
        assert GaslightingPatternType.CONFUSION in pattern_types

    def test_assess_overall_risk_level(self):
        """종합 위험도 평가"""
        service = GaslightingService()

        # 높은 위험도 대화
        high_risk_conversation = [
            {"speaker": "Speaker A", "text": "그런 적 없어", "time": "00:01:00"},
            {"speaker": "Speaker A", "text": "네가 잘못 기억하는 거야", "time": "00:02:00"},
            {"speaker": "Speaker A", "text": "네 때문에 이렇게 됐어", "time": "00:03:00"},
            {"speaker": "Speaker A", "text": "별거 아닌데", "time": "00:04:00"},
            {"speaker": "Speaker A", "text": "아까는 그랬는데 지금은 달라", "time": "00:05:00"},
        ]

        result = service.analyze_conversation(high_risk_conversation)

        assert result.risk_level in ["낮음", "중간", "높음", "매우 높음"]
        assert result.risk_level in ["높음", "매우 높음"]

    def test_generate_recommendations(self):
        """권고 조치 생성"""
        service = GaslightingService()

        conversation = [
            {"speaker": "Speaker A", "text": "그런 적 없어", "time": "00:01:00"},
            {"speaker": "Speaker A", "text": "네가 잘못 기억하는 거야", "time": "00:02:00"},
        ]

        result = service.analyze_conversation(conversation)

        # 권고 조치 포함 확인
        assert result.recommendations is not None
        assert len(result.recommendations) > 0


class TestEdgeCases:
    """엣지 케이스 테스트"""

    def test_empty_conversation(self):
        """빈 대화 처리"""
        service = GaslightingService()
        result = service.analyze_conversation([])

        assert len(result.patterns) == 0
        assert result.risk_level == "낮음"

    def test_no_gaslighting_patterns(self):
        """가스라이팅 패턴이 없는 일반 대화"""
        service = GaslightingService()

        normal_conversation = [
            {"speaker": "Speaker A", "text": "안녕하세요", "time": "00:01:00"},
            {"speaker": "Speaker B", "text": "네, 안녕하세요", "time": "00:02:00"},
            {"speaker": "Speaker A", "text": "오늘 날씨가 좋네요", "time": "00:03:00"},
        ]

        result = service.analyze_conversation(normal_conversation)

        # 패턴이 거의 감지되지 않음
        assert len(result.patterns) == 0 or all(p.confidence < 0.5 for p in result.patterns)

    def test_mixed_speakers_with_different_patterns(self):
        """다양한 화자와 패턴 혼합"""
        service = GaslightingService()

        conversation = [
            {"speaker": "Speaker A", "text": "그런 적 없어", "time": "00:01:00"},
            {"speaker": "Speaker B", "text": "말도 안 돼", "time": "00:02:00"},
            {"speaker": "Speaker A", "text": "네 때문에 그래", "time": "00:03:00"},
            {"speaker": "Speaker B", "text": "거짓말하지 마", "time": "00:04:00"},
            {"speaker": "Speaker A", "text": "별거 아닌데", "time": "00:00:05"},
        ]

        result = service.analyze_conversation(conversation)

        # 각 화자의 패턴 분리 확인
        assert "Speaker A" in result.speaker_patterns
        assert "Speaker B" in result.speaker_patterns


class TestPatternMatching:
    """패턴 매칭 테스트"""

    def test_case_variations(self):
        """다양한 표현 변형 매칭"""
        service = GaslightingService()

        variations = [
            "그런 적 없어",
            "그런 거 없었는데",
            "기억 안 나",
            "상상한 거지",
        ]

        for text in variations:
            patterns = service.detect_patterns(text)
            # 부정 패턴이 감지되어야 함
            assert len(patterns) > 0

    def test_pattern_with_modals(self):
        """조동사 포함 패턴 매칭"""
        service = GaslightingService()

        text = "그럴 리가 있어? 있을 수가 없지"
        patterns = service.detect_patterns(text)

        assert len(patterns) > 0
