"""
범죄 언어 패턴 데이터 모델 테스트
SPEC-FORENSIC-001 Phase 2-A: Crime Language Pattern Database

TDD RED Phase: 먼저 실패하는 테스트를 작성합니다.
"""

import pytest
from pydantic import ValidationError


class TestGaslightingPatternModel:
    """가스라이팅 패턴 모델 테스트"""

    def test_gaslighting_type_enum_has_all_required_types(self):
        """가스라이팅 패턴 유형 열거형이 7개 유형을 모두 포함하는지 테스트"""
        from voice_man.models.forensic.crime_language import GaslightingType

        required_types = [
            "denial",
            "countering",
            "trivializing",
            "diverting",
            "blocking",
            "blame_shifting",
            "reality_distortion",
        ]

        for type_name in required_types:
            assert hasattr(GaslightingType, type_name.upper()), f"Missing type: {type_name}"

    def test_gaslighting_pattern_creation_with_valid_data(self):
        """유효한 데이터로 가스라이팅 패턴 생성 테스트"""
        from voice_man.models.forensic.crime_language import (
            GaslightingPattern,
            GaslightingType,
        )

        pattern = GaslightingPattern(
            type=GaslightingType.DENIAL,
            patterns_ko=["그런 적 없어", "내가 언제 그랬어", "네가 잘못 들은 거야"],
            patterns_en=["That never happened", "I never said that"],
            severity_weight=0.8,
            description_ko="상대방의 경험이나 기억을 부정하는 패턴",
            description_en="Pattern of denying the other person's experience or memory",
        )

        assert pattern.type == GaslightingType.DENIAL
        assert len(pattern.patterns_ko) >= 3
        assert 0.0 <= pattern.severity_weight <= 1.0

    def test_gaslighting_pattern_severity_weight_validation(self):
        """심각도 가중치 범위 검증 테스트"""
        from voice_man.models.forensic.crime_language import (
            GaslightingPattern,
            GaslightingType,
        )

        # 유효하지 않은 severity_weight (1.0 초과)
        with pytest.raises(ValidationError):
            GaslightingPattern(
                type=GaslightingType.DENIAL,
                patterns_ko=["그런 적 없어"],
                patterns_en=["That never happened"],
                severity_weight=1.5,  # Invalid: > 1.0
                description_ko="설명",
                description_en="description",
            )

        # 유효하지 않은 severity_weight (0.0 미만)
        with pytest.raises(ValidationError):
            GaslightingPattern(
                type=GaslightingType.DENIAL,
                patterns_ko=["그런 적 없어"],
                patterns_en=["That never happened"],
                severity_weight=-0.1,  # Invalid: < 0.0
                description_ko="설명",
                description_en="description",
            )


class TestThreatPatternModel:
    """협박 패턴 모델 테스트"""

    def test_threat_type_enum_has_all_required_types(self):
        """협박 패턴 유형 열거형이 5개 유형을 모두 포함하는지 테스트"""
        from voice_man.models.forensic.crime_language import ThreatType

        required_types = [
            "direct",
            "conditional",
            "veiled",
            "economic",
            "social",
        ]

        for type_name in required_types:
            assert hasattr(ThreatType, type_name.upper()), f"Missing type: {type_name}"

    def test_threat_pattern_creation_with_valid_data(self):
        """유효한 데이터로 협박 패턴 생성 테스트"""
        from voice_man.models.forensic.crime_language import ThreatPattern, ThreatType

        pattern = ThreatPattern(
            type=ThreatType.DIRECT,
            patterns_ko=["죽여버린다", "가만 안 둬", "두고 봐"],
            patterns_en=["I will kill you", "You will pay for this"],
            severity_weight=1.0,
            description_ko="직접적인 위협이나 해를 가하겠다는 표현",
            description_en="Direct expression of threat or intent to harm",
        )

        assert pattern.type == ThreatType.DIRECT
        assert len(pattern.patterns_ko) >= 3
        assert pattern.severity_weight == 1.0


class TestCoercionPatternModel:
    """강압 패턴 모델 테스트"""

    def test_coercion_type_enum_has_all_required_types(self):
        """강압 패턴 유형 열거형이 3개 유형을 모두 포함하는지 테스트"""
        from voice_man.models.forensic.crime_language import CoercionType

        required_types = [
            "emotional",
            "guilt_induction",
            "isolation",
        ]

        for type_name in required_types:
            assert hasattr(CoercionType, type_name.upper()), f"Missing type: {type_name}"

    def test_coercion_pattern_creation_with_valid_data(self):
        """유효한 데이터로 강압 패턴 생성 테스트"""
        from voice_man.models.forensic.crime_language import CoercionPattern, CoercionType

        pattern = CoercionPattern(
            type=CoercionType.EMOTIONAL,
            patterns_ko=["나 없으면 못 살잖아", "나만 믿어", "내가 아니면 누가"],
            patterns_en=["You can't live without me", "Trust only me"],
            severity_weight=0.7,
            description_ko="감정적으로 상대방을 조종하는 패턴",
            description_en="Pattern of emotionally manipulating the other person",
        )

        assert pattern.type == CoercionType.EMOTIONAL
        assert len(pattern.patterns_ko) >= 3


class TestDeceptionMarkerModel:
    """기만 언어 지표 모델 테스트"""

    def test_deception_category_enum_has_all_required_categories(self):
        """기만 지표 카테고리 열거형이 5개 카테고리를 모두 포함하는지 테스트"""
        from voice_man.models.forensic.crime_language import DeceptionCategory

        required_categories = [
            "hedging",
            "distancing",
            "negative_emotion",
            "exclusive",
            "cognitive_complexity",
        ]

        for category_name in required_categories:
            assert hasattr(DeceptionCategory, category_name.upper()), (
                f"Missing category: {category_name}"
            )

    def test_deception_marker_creation_with_valid_data(self):
        """유효한 데이터로 기만 지표 생성 테스트"""
        from voice_man.models.forensic.crime_language import (
            DeceptionMarker,
            DeceptionCategory,
        )

        marker = DeceptionMarker(
            category=DeceptionCategory.HEDGING,
            markers_ko=["아마", "글쎄", "그런 것 같아", "잘 모르겠는데"],
            markers_en=["maybe", "I think", "sort of"],
            description_ko="불확실성을 나타내는 회피어",
            description_en="Hedging words indicating uncertainty",
        )

        assert marker.category == DeceptionCategory.HEDGING
        assert len(marker.markers_ko) >= 3


class TestCrimeLanguageMatchModel:
    """범죄 언어 매칭 결과 모델 테스트"""

    def test_gaslighting_match_creation(self):
        """가스라이팅 매칭 결과 생성 테스트"""
        from voice_man.models.forensic.crime_language import (
            GaslightingMatch,
            GaslightingType,
        )

        match = GaslightingMatch(
            type=GaslightingType.DENIAL,
            matched_pattern="그런 적 없어",
            text="그런 적 없어, 네가 착각한 거야",
            start_position=0,
            end_position=6,
            confidence=0.85,
            severity_weight=0.8,
        )

        assert match.type == GaslightingType.DENIAL
        assert match.matched_pattern == "그런 적 없어"
        assert 0.0 <= match.confidence <= 1.0

    def test_threat_match_creation(self):
        """협박 매칭 결과 생성 테스트"""
        from voice_man.models.forensic.crime_language import ThreatMatch, ThreatType

        match = ThreatMatch(
            type=ThreatType.DIRECT,
            matched_pattern="죽여버린다",
            text="한번만 더 그러면 죽여버린다",
            start_position=10,
            end_position=15,
            confidence=0.95,
            severity_weight=1.0,
        )

        assert match.type == ThreatType.DIRECT
        assert match.severity_weight == 1.0

    def test_coercion_match_creation(self):
        """강압 매칭 결과 생성 테스트"""
        from voice_man.models.forensic.crime_language import CoercionMatch, CoercionType

        match = CoercionMatch(
            type=CoercionType.ISOLATION,
            matched_pattern="친구들 만나지 마",
            text="이제 친구들 만나지 마, 걔네가 널 이용하는 거야",
            start_position=3,
            end_position=12,
            confidence=0.9,
            severity_weight=0.85,
        )

        assert match.type == CoercionType.ISOLATION


class TestDeceptionAnalysisModel:
    """기만 분석 결과 모델 테스트"""

    def test_deception_analysis_creation(self):
        """기만 분석 결과 생성 테스트"""
        from voice_man.models.forensic.crime_language import (
            DeceptionAnalysis,
            DeceptionCategory,
            DeceptionMarkerMatch,
        )

        marker_matches = [
            DeceptionMarkerMatch(
                category=DeceptionCategory.HEDGING,
                marker="아마",
                count=3,
            ),
            DeceptionMarkerMatch(
                category=DeceptionCategory.DISTANCING,
                marker="그 사람",
                count=5,
            ),
        ]

        analysis = DeceptionAnalysis(
            marker_matches=marker_matches,
            hedging_score=0.6,
            distancing_score=0.7,
            negative_emotion_score=0.3,
            exclusive_score=0.2,
            cognitive_complexity_score=0.4,
            overall_deception_score=0.5,
        )

        assert len(analysis.marker_matches) == 2
        assert 0.0 <= analysis.overall_deception_score <= 1.0


class TestCrimeLanguageScoreModel:
    """범죄 언어 종합 점수 모델 테스트"""

    def test_crime_language_score_creation(self):
        """범죄 언어 종합 점수 생성 테스트"""
        from voice_man.models.forensic.crime_language import CrimeLanguageScore

        score = CrimeLanguageScore(
            gaslighting_score=0.7,
            threat_score=0.3,
            coercion_score=0.5,
            deception_score=0.4,
            overall_risk_score=0.55,
            risk_level="중간",
            gaslighting_count=5,
            threat_count=2,
            coercion_count=3,
        )

        assert score.overall_risk_score == 0.55
        assert score.risk_level in ["낮음", "중간", "높음", "매우 높음"]

    def test_crime_language_score_validation(self):
        """범죄 언어 점수 범위 검증 테스트"""
        from voice_man.models.forensic.crime_language import CrimeLanguageScore

        # 모든 점수가 0.0 ~ 1.0 범위 내에 있어야 함
        with pytest.raises(ValidationError):
            CrimeLanguageScore(
                gaslighting_score=1.5,  # Invalid
                threat_score=0.3,
                coercion_score=0.5,
                deception_score=0.4,
                overall_risk_score=0.55,
                risk_level="중간",
                gaslighting_count=5,
                threat_count=2,
                coercion_count=3,
            )
