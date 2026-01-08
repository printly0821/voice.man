"""
TASK-008: 범죄 발언 태깅 시스템 테스트

테스트 케이스:
1. 협박 발언 감지 (죽여버린다, 가만 안 둔다, 불질러버린다)
2. 공갈 발언 감지 (돈 안 주면, 까불면 알지, 퍼뜨려버린다)
3. 사기 발언 감지 (무조건 된다, 손해 없다, 내가 보장한다)
4. 모욕 발언 감지 (비속어, 인격 비하 표현)
5. 복합 범죄 발언 감지 (협박 + 모욕)
6. 맥락 기반 분석 (Claude API)
7. 신뢰도 점수 산출 (0.0 ~ 1.0)
8. 범죄 키워드 사전 로드
"""

import pytest
from pathlib import Path
from voice_man.services.crime_tagging_service import CrimeTaggingService
from voice_man.models.crime_tag import CrimeTag, CrimeType


class TestCrimeKeywordDictionary:
    """범죄 키워드 사전 테스트"""

    def test_load_crime_keywords_json(self):
        """범죄 키워드 사전 파일 로드 성공"""
        # Given: crime_keywords.json 파일이 존재한다
        service = CrimeTaggingService()

        # When: 키워드 사전을 로드한다
        keywords = service.load_keywords()

        # Then: 모든 범죄 유형의 키워드가 포함되어 있다
        assert "협박" in keywords
        assert "공갈" in keywords
        assert "사기" in keywords
        assert "모욕" in keywords
        assert len(keywords["협박"]) > 0
        assert len(keywords["공갈"]) > 0
        assert len(keywords["사기"]) > 0
        assert len(keywords["모욕"]) > 0

    def test_keywords_contain_required_threats(self):
        """필수 협박 키워드 포함 검증"""
        service = CrimeTaggingService()
        keywords = service.load_keywords()

        # 필수 협박 키워드
        threat_keywords = ["죽여버린다", "가만 안 둔다", "불질러버린다"]
        for keyword in threat_keywords:
            assert any(keyword in k for k in keywords["협박"])

    def test_keywords_contain_required_intimidation(self):
        """필수 공갈 키워드 포함 검증"""
        service = CrimeTaggingService()
        keywords = service.load_keywords()

        # 필수 공갈 키워드
        intimidation_keywords = ["돈 안 주면", "까불면 알지", "퍼뜨려버린다"]
        for keyword in intimidation_keywords:
            assert any(keyword in k for k in keywords["공갈"])


class TestThreatDetection:
    """협박 발언 감지 테스트"""

    def test_detect_threat_kill(self):
        """ "죽여버린다" 협박 감지"""
        service = CrimeTaggingService()
        text = "너 지금 가서 죽여버린다 진짜"

        tags = service.detect_crime(text)

        assert len(tags) > 0
        assert any(t.type == CrimeType.THREAT for t in tags)
        threat_tag = next(t for t in tags if t.type == CrimeType.THREAT)
        assert threat_tag.confidence >= 0.8
        assert "죽여버린다" in threat_tag.keywords
        assert threat_tag.legal_reference == "형법 제283조"

    def test_detect_threat_wont_leave_alone(self):
        """ "가만 안 둔다" 협박 감지"""
        service = CrimeTaggingService()
        text = "네놈을 가만 안 둔다. 기다려라"

        tags = service.detect_crime(text)

        assert len(tags) > 0
        assert any(t.type == CrimeType.THREAT for t in tags)
        threat_tag = next(t for t in tags if t.type == CrimeType.THREAT)
        assert threat_tag.confidence >= 0.8
        assert any("가만 안" in kw for kw in threat_tag.keywords)

    def test_detect_threat_shoot(self):
        """ "불질러버린다" 협박 감지"""
        service = CrimeTaggingService()
        text = "지금 당장 불질러버린다"

        tags = service.detect_crime(text)

        assert len(tags) > 0
        assert any(t.type == CrimeType.THREAT for t in tags)
        threat_tag = next(t for t in tags if t.type == CrimeType.THREAT)
        assert threat_tag.confidence >= 0.8


class TestIntimidationDetection:
    """공갈 발언 감지 테스트"""

    def test_detect_intimidation_money_threat(self):
        """ "돈 안 주면" 공갈 감지"""
        service = CrimeTaggingService()
        text = "돈 안 주면 너 사진 퍼뜨려버린다"

        tags = service.detect_crime(text)

        assert len(tags) > 0
        assert any(t.type == CrimeType.INTIMIDATION for t in tags)
        intimidation_tag = next(t for t in tags if t.type == CrimeType.INTIMIDATION)
        assert intimidation_tag.confidence >= 0.7
        assert intimidation_tag.legal_reference == "형법 제350조"

    def test_detect_intimidation_you_know(self):
        """ "까불면 알지" 공갈 감지"""
        service = CrimeTaggingService()
        text = "까불면 알지? 조용히 해"

        tags = service.detect_crime(text)

        assert len(tags) > 0
        assert any(t.type == CrimeType.INTIMIDATION for t in tags)

    def test_detect_intimidation_spread(self):
        """ "퍼뜨려버린다" 공갈 감지"""
        service = CrimeTaggingService()
        text = "네 이야기 다 퍼뜨려버린다"

        tags = service.detect_crime(text)

        assert len(tags) > 0
        assert any(t.type == CrimeType.INTIMIDATION for t in tags)


class TestFraudDetection:
    """사기 발언 감지 테스트"""

    def test_detect_fraud_guaranteed(self):
        """ "무조건 된다" 사기 감지"""
        service = CrimeTaggingService()
        text = "이건 무조건 된다. 내가 보장한다"

        tags = service.detect_crime(text)

        assert len(tags) > 0
        assert any(t.type == CrimeType.FRAUD for t in tags)
        fraud_tag = next(t for t in tags if t.type == CrimeType.FRAUD)
        assert fraud_tag.confidence >= 0.7
        assert fraud_tag.legal_reference == "형법 제347조"

    def test_detect_fraud_no_loss(self):
        """ "손해 없다" 사기 감지"""
        service = CrimeTaggingService()
        text = "절대 손해 없다. 100% 확실해"

        tags = service.detect_crime(text)

        assert len(tags) > 0
        assert any(t.type == CrimeType.FRAUD for t in tags)


class TestInsultDetection:
    """모욕 발언 감지 테스트"""

    def test_detect_insult_profanity(self):
        """비속어 포함 모욕 감지"""
        service = CrimeTaggingService()
        text = "니는 정말 멍청이냐"  # 예시 텍스트

        tags = service.detect_crime(text)

        assert len(tags) > 0
        assert any(t.type == CrimeType.INSULT for t in tags)
        insult_tag = next(t for t in tags if t.type == CrimeType.INSULT)
        assert insult_tag.confidence >= 0.7
        assert insult_tag.legal_reference == "형법 제311조"


class TestComplexCrimeDetection:
    """복합 범죄 발언 감지 테스트"""

    def test_detect_threat_and_insult(self):
        """협박 + 모욕 복합 감지"""
        service = CrimeTaggingService()
        text = "죽여버린다, 멍청한 놈아"

        tags = service.detect_crime(text)

        assert len(tags) >= 2
        assert any(t.type == CrimeType.THREAT for t in tags)
        assert any(t.type == CrimeType.INSULT for t in tags)

        # 각 태그의 신뢰도 점수 확인
        threat_tag = next(t for t in tags if t.type == CrimeType.THREAT)
        insult_tag = next(t for t in tags if t.type == CrimeType.INSULT)
        assert 0.0 <= threat_tag.confidence <= 1.0
        assert 0.0 <= insult_tag.confidence <= 1.0


class TestConfidenceScoreCalculation:
    """신뢰도 점수 산출 테스트"""

    def test_confidence_score_range(self):
        """신뢰도 점수가 0.0 ~ 1.0 범위인지 확인"""
        service = CrimeTaggingService()
        text = "죽여버린다"

        tags = service.detect_crime(text)

        for tag in tags:
            assert 0.0 <= tag.confidence <= 1.0

    def test_high_confidence_for_explicit_threat(self):
        """명시적 협박에 대한 높은 신뢰도 점수"""
        service = CrimeTaggingService()
        text = "지금 당장 너를 죽여버린다"

        tags = service.detect_crime(text)
        threat_tag = next(t for t in tags if t.type == CrimeType.THREAT)

        assert threat_tag.confidence >= 0.8

    def test_low_confidence_for_ambiguous_text(self):
        """모호한 텍스트에 대한 낮은 신뢰도 점수"""
        service = CrimeTaggingService()
        text = "그게 다 죽인다"  # 맥락에 따라 다름

        tags = service.detect_crime(text)

        if len(tags) > 0:
            threat_tag = next((t for t in tags if t.type == CrimeType.THREAT), None)
            if threat_tag:
                assert threat_tag.confidence < 0.8


class TestContextBasedAnalysis:
    """맥락 기반 분석 테스트 (Claude API 통합)"""

    @pytest.mark.asyncio
    async def test_context_analysis_with_claude_api(self):
        """Claude API를 사용한 맥락 기반 분석"""
        service = CrimeTaggingService(use_claude_api=True)

        # 농담/비유적 맥락
        text = "아이고, 이거 죽인다~ 배부워"  # 음식 맛 표현
        context = "음식 맛에 대해 이야기하는 상황"

        tags = await service.detect_crime_with_context(text, context)

        # 맥락을 고려하여 낮은 신뢰도 또는 "검토 필요" 상태
        if len(tags) > 0:
            threat_tag = next((t for t in tags if t.type == CrimeType.THREAT), None)
            if threat_tag:
                assert threat_tag.confidence < 0.5 or threat_tag.needs_review

    @pytest.mark.asyncio
    async def test_real_threat_with_context(self):
        """실제 협박 맥락 분석"""
        service = CrimeTaggingService(use_claude_api=True)

        text = "가만 안 둔다. 조심해"
        context = "화난 목소리로 위협적인 상황"

        tags = await service.detect_crime_with_context(text, context)

        assert len(tags) > 0
        threat_tag = next(t for t in tags if t.type == CrimeType.THREAT)
        assert threat_tag.confidence >= 0.7


class TestRegexPatternMatching:
    """정규 표현식 기반 패턴 매칭 테스트"""

    def test_regex_matches_various_threat_patterns(self):
        """다양한 협박 패턴 정규 표현식 매칭"""
        service = CrimeTaggingService()

        test_cases = [
            "죽여버린다",
            "죽인다",
            "가만 안 둔다",
            "불질러버린다",
            "없애버린다",
        ]

        for text in test_cases:
            tags = service.detect_crime(text)
            assert len(tags) > 0
            assert any(t.type == CrimeType.THREAT for t in tags)

    def test_regex_case_insensitive(self):
        """대소문자 구분 없이 매칭 (한국어의 경우 형태소 분석)"""
        service = CrimeTaggingService()

        # 한국어는 대소문자가 없지만, 형태소 변형 테스트
        text = "죽일 거다!"
        tags = service.detect_crime(text)

        assert len(tags) > 0
        assert any(t.type == CrimeType.THREAT for t in tags)


class TestKoELECTRAModel:
    """KoELECTRA 모델 기반 분류 테스트"""

    @pytest.mark.skip(reason="Requires KoELECTRA model download")
    def test_koelectra_model_loads(self):
        """KoELECTRA-base-v3 모델 로드 성공"""
        service = CrimeTaggingService(use_model=True)

        assert service.model is not None
        assert service.tokenizer is not None

    @pytest.mark.skip(reason="Requires KoELECTRA model download")
    def test_koelectra_classifies_crime(self):
        """KoELECTRA 모델 범죄 분류"""
        service = CrimeTaggingService(use_model=True)

        text = "너를 죽여버린다"
        label = service.classify_with_model(text)

        assert label in ["협박", "공갈", "사기", "모욕", "정상"]


class TestLegalReference:
    """법적 참조 테스트"""

    def test_threat_legal_reference(self):
        """협박 법적 참조 (형법 제283조)"""
        service = CrimeTaggingService()
        text = "죽여버린다"

        tags = service.detect_crime(text)
        threat_tag = next(t for t in tags if t.type == CrimeType.THREAT)

        assert threat_tag.legal_reference == "형법 제283조"

    def test_intimidation_legal_reference(self):
        """공갈 법적 참조 (형법 제350조)"""
        service = CrimeTaggingService()
        text = "돈 안 주면 퍼뜨려버린다"

        tags = service.detect_crime(text)
        intimidation_tag = next(t for t in tags if t.type == CrimeType.INTIMIDATION)

        assert intimidation_tag.legal_reference == "형법 제350조"

    def test_fraud_legal_reference(self):
        """사기 법적 참조 (형법 제347조)"""
        service = CrimeTaggingService()
        text = "무조건 된다. 내가 보장한다"

        tags = service.detect_crime(text)
        fraud_tag = next(t for t in tags if t.type == CrimeType.FRAUD)

        assert fraud_tag.legal_reference == "형법 제347조"

    def test_insult_legal_reference(self):
        """모욕 법적 참조 (형법 제311조)"""
        service = CrimeTaggingService()
        text = "멍청이"  # 예시

        tags = service.detect_crime(text)
        insult_tag = next((t for t in tags if t.type == CrimeType.INSULT), None)

        if insult_tag:
            assert insult_tag.legal_reference == "형법 제311조"


class TestEdgeCases:
    """엣지 케이스 테스트"""

    def test_empty_text(self):
        """빈 텍스트 처리"""
        service = CrimeTaggingService()
        tags = service.detect_crime("")

        assert len(tags) == 0

    def test_no_crime_text(self):
        """범죄 발언이 없는 일반 텍스트"""
        service = CrimeTaggingService()
        text = "안녕하세요, 오늘 날씨가 좋네요"

        tags = service.detect_crime(text)

        # 명확한 범죄 발언이 없으면 빈 리스트
        assert len(tags) == 0

    def test_very_long_text(self):
        """매우 긴 텍스트 처리"""
        service = CrimeTaggingService()
        long_text = "정상적인 대화입니다. " * 1000 + "죽여버린다"

        tags = service.detect_crime(long_text)

        # 긴 텍스트에서도 협박 감지
        assert len(tags) > 0
        assert any(t.type == CrimeType.THREAT for t in tags)

    def test_multiple_crimes_in_one_sentence(self):
        """한 문장 내 복수 범죄 발언"""
        service = CrimeTaggingService()
        text = "죽여버린다, 돈 안 주면 퍼뜨리고, 무조건 된다, 멍청아"

        tags = service.detect_crime(text)

        assert len(tags) >= 3  # 협박, 공갈, 사기, 모욕 중 최소 3개
