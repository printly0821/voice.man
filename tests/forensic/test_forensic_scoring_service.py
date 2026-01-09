"""
Tests for Forensic Scoring Service
SPEC-FORENSIC-001 Phase 2-D: TDD tests for forensic scoring service

RED Phase: These tests define expected behavior for the forensic scoring service.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime

from voice_man.models.forensic.forensic_score import (
    RiskLevel,
    ForensicCategory,
    CategoryScore,
    DeceptionAnalysis,
    GaslightingAnalysis,
    ThreatAssessment,
    ForensicEvidence,
    ForensicScoreResult,
)
from voice_man.models.forensic.crime_language import (
    CrimeLanguageScore,
    GaslightingMatch,
    GaslightingType,
    ThreatMatch,
    ThreatType,
    CoercionMatch,
    CoercionType,
)
from voice_man.models.forensic.cross_validation import (
    CrossValidationResult,
    TextAnalysisResult,
    VoiceAnalysisResult,
    DiscrepancySeverity,
)
from voice_man.models.forensic.emotion_recognition import (
    MultiModelEmotionResult,
    EmotionAnalysisResult,
    EmotionDimensions,
    CategoricalEmotion,
    ForensicEmotionIndicators,
)
from voice_man.models.forensic.audio_features import StressFeatures
from voice_man.services.forensic.forensic_scoring_service import ForensicScoringService


class TestForensicScoringServiceInit:
    """Test ForensicScoringService initialization."""

    def test_service_initialization(self):
        """Test that service can be initialized with dependencies."""
        audio_service = MagicMock()
        stress_service = MagicMock()
        crime_service = MagicMock()
        ser_service = MagicMock()
        cross_validation_service = MagicMock()

        service = ForensicScoringService(
            audio_feature_service=audio_service,
            stress_analysis_service=stress_service,
            crime_language_service=crime_service,
            ser_service=ser_service,
            cross_validation_service=cross_validation_service,
        )

        assert service is not None


class TestGaslightingScoreCalculation:
    """Test gaslighting score calculation."""

    @pytest.fixture
    def service(self):
        """Create a ForensicScoringService with mocked dependencies."""
        return ForensicScoringService(
            audio_feature_service=MagicMock(),
            stress_analysis_service=MagicMock(),
            crime_language_service=MagicMock(),
            ser_service=MagicMock(),
            cross_validation_service=MagicMock(),
        )

    def test_calculate_gaslighting_score_high_pattern(self, service):
        """Test gaslighting score with high pattern detection."""
        crime_analysis = CrimeLanguageScore(
            gaslighting_score=0.8,
            threat_score=0.2,
            coercion_score=0.3,
            deception_score=0.4,
            overall_risk_score=0.5,
            risk_level="높음",
            gaslighting_count=5,
            threat_count=1,
            coercion_count=2,
        )

        cross_validation = CrossValidationResult(
            text_analysis=TextAnalysisResult(
                text="test",
                detected_sentiment="negative",
                sentiment_score=-0.5,
                detected_emotions=["anger"],
                crime_patterns_found=["gaslighting"],
                intensity_level=0.7,
            ),
            voice_analysis=VoiceAnalysisResult(
                dominant_emotion="angry",
                emotion_confidence=0.8,
                arousal=0.7,
                valence=0.3,
                stress_level=0.6,
            ),
            discrepancies=[],
            overall_consistency_score=0.6,
            deception_probability=0.4,
            risk_level=DiscrepancySeverity.MEDIUM,
            analysis_notes=[],
        )

        result = service.calculate_gaslighting_score(crime_analysis, cross_validation)

        assert isinstance(result, GaslightingAnalysis)
        assert result.intensity_score >= 60.0  # High gaslighting should result in high score
        assert result.victim_impact_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]

    def test_calculate_gaslighting_score_low_pattern(self, service):
        """Test gaslighting score with low pattern detection."""
        crime_analysis = CrimeLanguageScore(
            gaslighting_score=0.1,
            threat_score=0.1,
            coercion_score=0.1,
            deception_score=0.1,
            overall_risk_score=0.1,
            risk_level="낮음",
            gaslighting_count=0,
            threat_count=0,
            coercion_count=0,
        )

        cross_validation = CrossValidationResult(
            text_analysis=TextAnalysisResult(
                text="test",
                detected_sentiment="neutral",
                sentiment_score=0.0,
                detected_emotions=[],
                crime_patterns_found=[],
                intensity_level=0.2,
            ),
            voice_analysis=VoiceAnalysisResult(
                dominant_emotion="neutral",
                emotion_confidence=0.8,
                arousal=0.4,
                valence=0.5,
                stress_level=0.2,
            ),
            discrepancies=[],
            overall_consistency_score=0.9,
            deception_probability=0.1,
            risk_level=DiscrepancySeverity.LOW,
            analysis_notes=[],
        )

        result = service.calculate_gaslighting_score(crime_analysis, cross_validation)

        assert isinstance(result, GaslightingAnalysis)
        assert result.intensity_score <= 30.0  # Low gaslighting should result in low score
        assert result.victim_impact_level in [RiskLevel.MINIMAL, RiskLevel.LOW]


class TestDeceptionScoreCalculation:
    """Test deception score calculation."""

    @pytest.fixture
    def service(self):
        """Create a ForensicScoringService with mocked dependencies."""
        return ForensicScoringService(
            audio_feature_service=MagicMock(),
            stress_analysis_service=MagicMock(),
            crime_language_service=MagicMock(),
            ser_service=MagicMock(),
            cross_validation_service=MagicMock(),
        )

    def test_calculate_deception_score_high_inconsistency(self, service):
        """Test deception score with high voice-text inconsistency."""
        cross_validation = CrossValidationResult(
            text_analysis=TextAnalysisResult(
                text="I'm so happy!",
                detected_sentiment="positive",
                sentiment_score=0.8,
                detected_emotions=["joy"],
                crime_patterns_found=[],
                intensity_level=0.3,
            ),
            voice_analysis=VoiceAnalysisResult(
                dominant_emotion="sad",
                emotion_confidence=0.9,
                arousal=0.3,
                valence=0.2,
                stress_level=0.7,
            ),
            discrepancies=[],
            overall_consistency_score=0.2,  # Low consistency
            deception_probability=0.8,
            risk_level=DiscrepancySeverity.HIGH,
            analysis_notes=[],
        )

        emotion_result = MultiModelEmotionResult(
            primary_result=EmotionAnalysisResult(
                dimensions=EmotionDimensions(arousal=0.3, dominance=0.4, valence=0.2),
                model_used="test",
                processing_time_ms=100.0,
            ),
            ensemble_confidence=0.8,
            audio_duration_seconds=60.0,
        )

        crime_analysis = CrimeLanguageScore(
            gaslighting_score=0.2,
            threat_score=0.1,
            coercion_score=0.1,
            deception_score=0.7,
            overall_risk_score=0.3,
            risk_level="중간",
            gaslighting_count=1,
            threat_count=0,
            coercion_count=0,
        )

        result = service.calculate_deception_score(cross_validation, emotion_result, crime_analysis)

        assert isinstance(result, DeceptionAnalysis)
        assert result.deception_probability >= 0.6  # High deception expected
        assert result.voice_text_consistency <= 0.4  # Low consistency

    def test_calculate_deception_score_consistent(self, service):
        """Test deception score with consistent voice and text."""
        cross_validation = CrossValidationResult(
            text_analysis=TextAnalysisResult(
                text="I'm feeling sad today",
                detected_sentiment="negative",
                sentiment_score=-0.6,
                detected_emotions=["sadness"],
                crime_patterns_found=[],
                intensity_level=0.4,
            ),
            voice_analysis=VoiceAnalysisResult(
                dominant_emotion="sad",
                emotion_confidence=0.9,
                arousal=0.3,
                valence=0.2,
                stress_level=0.4,
            ),
            discrepancies=[],
            overall_consistency_score=0.9,  # High consistency
            deception_probability=0.1,
            risk_level=DiscrepancySeverity.LOW,
            analysis_notes=[],
        )

        emotion_result = MultiModelEmotionResult(
            primary_result=EmotionAnalysisResult(
                dimensions=EmotionDimensions(arousal=0.3, dominance=0.3, valence=0.2),
                model_used="test",
                processing_time_ms=100.0,
            ),
            ensemble_confidence=0.9,
            audio_duration_seconds=60.0,
        )

        crime_analysis = CrimeLanguageScore(
            gaslighting_score=0.0,
            threat_score=0.0,
            coercion_score=0.0,
            deception_score=0.1,
            overall_risk_score=0.05,
            risk_level="낮음",
            gaslighting_count=0,
            threat_count=0,
            coercion_count=0,
        )

        result = service.calculate_deception_score(cross_validation, emotion_result, crime_analysis)

        assert isinstance(result, DeceptionAnalysis)
        assert result.deception_probability <= 0.3  # Low deception expected
        assert result.voice_text_consistency >= 0.7  # High consistency


class TestThreatAssessment:
    """Test threat assessment calculation."""

    @pytest.fixture
    def service(self):
        """Create a ForensicScoringService with mocked dependencies."""
        return ForensicScoringService(
            audio_feature_service=MagicMock(),
            stress_analysis_service=MagicMock(),
            crime_language_service=MagicMock(),
            ser_service=MagicMock(),
            cross_validation_service=MagicMock(),
        )

    def test_assess_threat_level_high(self, service):
        """Test threat assessment with high threat patterns."""
        crime_analysis = CrimeLanguageScore(
            gaslighting_score=0.3,
            threat_score=0.9,
            coercion_score=0.6,
            deception_score=0.4,
            overall_risk_score=0.7,
            risk_level="매우 높음",
            gaslighting_count=2,
            threat_count=5,
            coercion_count=3,
        )

        stress_result = StressFeatures(
            shimmer_percent=8.0,
            hnr_db=15.0,
            formant_stability_score=45.0,
            stress_index=75.0,
            risk_level="high",
        )

        result = service.assess_threat_level(crime_analysis, stress_result)

        assert isinstance(result, ThreatAssessment)
        assert result.threat_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
        assert result.credibility_score >= 0.6

    def test_assess_threat_level_low(self, service):
        """Test threat assessment with minimal threat patterns."""
        crime_analysis = CrimeLanguageScore(
            gaslighting_score=0.1,
            threat_score=0.05,
            coercion_score=0.1,
            deception_score=0.1,
            overall_risk_score=0.1,
            risk_level="낮음",
            gaslighting_count=0,
            threat_count=0,
            coercion_count=0,
        )

        stress_result = StressFeatures(
            shimmer_percent=2.0,
            hnr_db=25.0,
            formant_stability_score=80.0,
            stress_index=20.0,
            risk_level="low",
        )

        result = service.assess_threat_level(crime_analysis, stress_result)

        assert isinstance(result, ThreatAssessment)
        assert result.threat_level in [RiskLevel.MINIMAL, RiskLevel.LOW]


class TestOverallScoreCalculation:
    """Test overall score calculation."""

    @pytest.fixture
    def service(self):
        """Create a ForensicScoringService with mocked dependencies."""
        return ForensicScoringService(
            audio_feature_service=MagicMock(),
            stress_analysis_service=MagicMock(),
            crime_language_service=MagicMock(),
            ser_service=MagicMock(),
            cross_validation_service=MagicMock(),
        )

    def test_calculate_overall_score_weights(self, service):
        """Test overall score calculation with correct weights."""
        category_scores = [
            CategoryScore(
                category=ForensicCategory.GASLIGHTING,
                score=80.0,  # 25% weight
                confidence=0.9,
                evidence_count=5,
                key_indicators=["denial"],
            ),
            CategoryScore(
                category=ForensicCategory.THREAT,
                score=60.0,  # 25% weight
                confidence=0.8,
                evidence_count=3,
                key_indicators=["direct threat"],
            ),
            CategoryScore(
                category=ForensicCategory.COERCION,
                score=40.0,  # 20% weight
                confidence=0.7,
                evidence_count=2,
                key_indicators=["emotional"],
            ),
            CategoryScore(
                category=ForensicCategory.DECEPTION,
                score=50.0,  # 20% weight
                confidence=0.8,
                evidence_count=4,
                key_indicators=["hedging"],
            ),
            CategoryScore(
                category=ForensicCategory.EMOTIONAL_MANIPULATION,
                score=30.0,  # 10% weight
                confidence=0.6,
                evidence_count=1,
                key_indicators=["guilt"],
            ),
        ]

        overall_score, risk_level = service.calculate_overall_score(category_scores)

        # Expected: 80*0.25 + 60*0.25 + 40*0.20 + 50*0.20 + 30*0.10 = 56.0
        assert 50.0 <= overall_score <= 62.0
        assert risk_level == RiskLevel.MODERATE

    def test_calculate_overall_score_critical(self, service):
        """Test overall score calculation for critical risk."""
        category_scores = [
            CategoryScore(
                category=ForensicCategory.GASLIGHTING,
                score=95.0,
                confidence=0.95,
                evidence_count=10,
                key_indicators=["all patterns"],
            ),
            CategoryScore(
                category=ForensicCategory.THREAT,
                score=90.0,
                confidence=0.9,
                evidence_count=8,
                key_indicators=["direct", "immediate"],
            ),
            CategoryScore(
                category=ForensicCategory.COERCION,
                score=85.0,
                confidence=0.85,
                evidence_count=6,
                key_indicators=["isolation"],
            ),
            CategoryScore(
                category=ForensicCategory.DECEPTION,
                score=88.0,
                confidence=0.88,
                evidence_count=7,
                key_indicators=["all markers"],
            ),
            CategoryScore(
                category=ForensicCategory.EMOTIONAL_MANIPULATION,
                score=80.0,
                confidence=0.8,
                evidence_count=5,
                key_indicators=["guilt", "shame"],
            ),
        ]

        overall_score, risk_level = service.calculate_overall_score(category_scores)

        assert overall_score >= 81.0
        assert risk_level == RiskLevel.CRITICAL

    def test_calculate_overall_score_minimal(self, service):
        """Test overall score calculation for minimal risk."""
        category_scores = [
            CategoryScore(
                category=ForensicCategory.GASLIGHTING,
                score=5.0,
                confidence=0.5,
                evidence_count=0,
                key_indicators=[],
            ),
            CategoryScore(
                category=ForensicCategory.THREAT,
                score=8.0,
                confidence=0.4,
                evidence_count=0,
                key_indicators=[],
            ),
            CategoryScore(
                category=ForensicCategory.COERCION,
                score=3.0,
                confidence=0.3,
                evidence_count=0,
                key_indicators=[],
            ),
            CategoryScore(
                category=ForensicCategory.DECEPTION,
                score=10.0,
                confidence=0.5,
                evidence_count=0,
                key_indicators=[],
            ),
            CategoryScore(
                category=ForensicCategory.EMOTIONAL_MANIPULATION,
                score=2.0,
                confidence=0.2,
                evidence_count=0,
                key_indicators=[],
            ),
        ]

        overall_score, risk_level = service.calculate_overall_score(category_scores)

        assert overall_score <= 20.0
        assert risk_level == RiskLevel.MINIMAL


class TestRiskLevelMapping:
    """Test risk level mapping from scores."""

    @pytest.fixture
    def service(self):
        """Create a ForensicScoringService with mocked dependencies."""
        return ForensicScoringService(
            audio_feature_service=MagicMock(),
            stress_analysis_service=MagicMock(),
            crime_language_service=MagicMock(),
            ser_service=MagicMock(),
            cross_validation_service=MagicMock(),
        )

    @pytest.mark.parametrize(
        "score,expected_level",
        [
            (0, RiskLevel.MINIMAL),
            (10, RiskLevel.MINIMAL),
            (20, RiskLevel.MINIMAL),
            (21, RiskLevel.LOW),
            (30, RiskLevel.LOW),
            (40, RiskLevel.LOW),
            (41, RiskLevel.MODERATE),
            (50, RiskLevel.MODERATE),
            (60, RiskLevel.MODERATE),
            (61, RiskLevel.HIGH),
            (70, RiskLevel.HIGH),
            (80, RiskLevel.HIGH),
            (81, RiskLevel.CRITICAL),
            (90, RiskLevel.CRITICAL),
            (100, RiskLevel.CRITICAL),
        ],
    )
    def test_score_to_risk_level_mapping(self, service, score, expected_level):
        """Test score to risk level mapping."""
        result = service._map_score_to_risk_level(score)
        assert result == expected_level


class TestEvidenceGeneration:
    """Test evidence item generation."""

    @pytest.fixture
    def service(self):
        """Create a ForensicScoringService with mocked dependencies."""
        return ForensicScoringService(
            audio_feature_service=MagicMock(),
            stress_analysis_service=MagicMock(),
            crime_language_service=MagicMock(),
            ser_service=MagicMock(),
            cross_validation_service=MagicMock(),
        )

    def test_generate_evidence_items(self, service):
        """Test evidence item generation from analysis results."""
        all_results = {
            "gaslighting_matches": [
                GaslightingMatch(
                    type=GaslightingType.DENIAL,
                    matched_pattern="그런 적 없어",
                    text="나는 그런 적 없어",
                    start_position=3,
                    end_position=10,
                    confidence=0.9,
                    severity_weight=0.8,
                ),
            ],
            "threat_matches": [
                ThreatMatch(
                    type=ThreatType.CONDITIONAL,
                    matched_pattern="그러면 가만 안 둬",
                    text="그러면 가만 안 둬",
                    start_position=0,
                    end_position=10,
                    confidence=0.85,
                    severity_weight=0.9,
                ),
            ],
            "audio_duration": 120.0,
        }

        evidence_items = service.generate_evidence_items(all_results)

        assert isinstance(evidence_items, list)
        assert len(evidence_items) >= 2
        for item in evidence_items:
            assert isinstance(item, ForensicEvidence)


class TestSummaryGeneration:
    """Test summary and recommendation generation."""

    @pytest.fixture
    def service(self):
        """Create a ForensicScoringService with mocked dependencies."""
        return ForensicScoringService(
            audio_feature_service=MagicMock(),
            stress_analysis_service=MagicMock(),
            crime_language_service=MagicMock(),
            ser_service=MagicMock(),
            cross_validation_service=MagicMock(),
        )

    def test_generate_summary_high_risk(self, service):
        """Test summary generation for high risk result."""
        score_result = MagicMock()
        score_result.overall_risk_level = RiskLevel.HIGH
        score_result.overall_risk_score = 75.0
        score_result.category_scores = [
            CategoryScore(
                category=ForensicCategory.GASLIGHTING,
                score=85.0,
                confidence=0.9,
                evidence_count=5,
                key_indicators=["denial", "blame shifting"],
            ),
            CategoryScore(
                category=ForensicCategory.THREAT,
                score=70.0,
                confidence=0.8,
                evidence_count=3,
                key_indicators=["conditional"],
            ),
        ]

        summary, recommendations = service.generate_summary_and_recommendations(score_result)

        assert isinstance(summary, str)
        assert len(summary) > 0
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        # High risk should include specific recommendations
        assert any("상담" in rec or "전문가" in rec or "법적" in rec for rec in recommendations)

    def test_generate_summary_low_risk(self, service):
        """Test summary generation for low risk result."""
        score_result = MagicMock()
        score_result.overall_risk_level = RiskLevel.LOW
        score_result.overall_risk_score = 25.0
        score_result.category_scores = [
            CategoryScore(
                category=ForensicCategory.GASLIGHTING,
                score=15.0,
                confidence=0.5,
                evidence_count=1,
                key_indicators=[],
            ),
        ]

        summary, recommendations = service.generate_summary_and_recommendations(score_result)

        assert isinstance(summary, str)
        assert isinstance(recommendations, list)


class TestFullAnalysis:
    """Test full forensic analysis workflow."""

    @pytest.fixture
    def service(self):
        """Create a ForensicScoringService with mocked dependencies."""
        audio_service = MagicMock()
        stress_service = MagicMock()
        crime_service = MagicMock()
        ser_service = MagicMock()
        cross_validation_service = MagicMock()

        return ForensicScoringService(
            audio_feature_service=audio_service,
            stress_analysis_service=stress_service,
            crime_language_service=crime_service,
            ser_service=ser_service,
            cross_validation_service=cross_validation_service,
        )

    @pytest.mark.asyncio
    async def test_analyze_returns_complete_result(self, service):
        """Test that analyze returns a complete ForensicScoreResult."""
        # Mock all the necessary services
        service._audio_feature_service.analyze_audio_features = MagicMock()
        service._stress_analysis_service.analyze_stress = MagicMock(
            return_value=StressFeatures(
                shimmer_percent=5.0,
                hnr_db=20.0,
                formant_stability_score=70.0,
                stress_index=45.0,
                risk_level="medium",
            )
        )
        service._crime_language_service.analyze_comprehensive = MagicMock(
            return_value=CrimeLanguageScore(
                gaslighting_score=0.5,
                threat_score=0.3,
                coercion_score=0.2,
                deception_score=0.4,
                overall_risk_score=0.4,
                risk_level="중간",
                gaslighting_count=3,
                threat_count=2,
                coercion_count=1,
            )
        )
        service._crime_language_service.detect_gaslighting = MagicMock(return_value=[])
        service._crime_language_service.detect_threats = MagicMock(return_value=[])
        service._crime_language_service.detect_coercion = MagicMock(return_value=[])

        service._ser_service.analyze_ensemble = MagicMock(
            return_value=MultiModelEmotionResult(
                primary_result=EmotionAnalysisResult(
                    dimensions=EmotionDimensions(arousal=0.5, dominance=0.5, valence=0.4),
                    model_used="test",
                    processing_time_ms=100.0,
                ),
                ensemble_confidence=0.8,
                audio_duration_seconds=60.0,
            )
        )
        service._ser_service.get_forensic_emotion_indicators = MagicMock(
            return_value=ForensicEmotionIndicators(
                high_arousal_low_valence=False,
                emotion_inconsistency_score=0.2,
                dominant_emotion="neutral",
                arousal_level="medium",
                stress_indicator=False,
                deception_indicator=False,
                confidence=0.8,
            )
        )

        service._cross_validation_service.cross_validate = MagicMock(
            return_value=CrossValidationResult(
                text_analysis=TextAnalysisResult(
                    text="test transcript",
                    detected_sentiment="neutral",
                    sentiment_score=0.0,
                    detected_emotions=[],
                    crime_patterns_found=[],
                    intensity_level=0.3,
                ),
                voice_analysis=VoiceAnalysisResult(
                    dominant_emotion="neutral",
                    emotion_confidence=0.8,
                    arousal=0.5,
                    valence=0.5,
                    stress_level=0.4,
                ),
                discrepancies=[],
                overall_consistency_score=0.8,
                deception_probability=0.2,
                risk_level=DiscrepancySeverity.LOW,
                analysis_notes=[],
            )
        )

        # Mock librosa for audio loading with proper numpy array
        import numpy as np
        import voice_man.services.forensic.forensic_scoring_service as scoring_module

        mock_audio = np.zeros(16000 * 60)  # 60 seconds of audio at 16kHz
        original_librosa = scoring_module.librosa

        # Create a mock librosa module
        mock_librosa = MagicMock()
        mock_librosa.load = MagicMock(return_value=(mock_audio, 16000))
        scoring_module.librosa = mock_librosa

        try:
            result = await service.analyze("/path/to/audio.wav", "test transcript")
        finally:
            scoring_module.librosa = original_librosa

        assert isinstance(result, ForensicScoreResult)
        assert result.analysis_id is not None
        assert isinstance(result.analyzed_at, datetime)
        assert 0 <= result.overall_risk_score <= 100
        # RiskLevel is serialized as string due to use_enum_values=True in ConfigDict
        assert result.overall_risk_level in [level.value for level in RiskLevel]
        assert isinstance(result.deception_analysis, DeceptionAnalysis)
        assert isinstance(result.gaslighting_analysis, GaslightingAnalysis)
        assert isinstance(result.threat_assessment, ThreatAssessment)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.fixture
    def service(self):
        """Create a ForensicScoringService with mocked dependencies."""
        return ForensicScoringService(
            audio_feature_service=MagicMock(),
            stress_analysis_service=MagicMock(),
            crime_language_service=MagicMock(),
            ser_service=MagicMock(),
            cross_validation_service=MagicMock(),
        )

    def test_empty_category_scores(self, service):
        """Test overall score calculation with empty category scores."""
        category_scores = []

        overall_score, risk_level = service.calculate_overall_score(category_scores)

        assert overall_score == 0.0
        assert risk_level == RiskLevel.MINIMAL

    def test_partial_category_scores(self, service):
        """Test overall score calculation with partial category scores."""
        # Only some categories present
        category_scores = [
            CategoryScore(
                category=ForensicCategory.GASLIGHTING,
                score=60.0,
                confidence=0.8,
                evidence_count=3,
                key_indicators=["denial"],
            ),
            CategoryScore(
                category=ForensicCategory.THREAT,
                score=40.0,
                confidence=0.7,
                evidence_count=2,
                key_indicators=["veiled"],
            ),
        ]

        overall_score, risk_level = service.calculate_overall_score(category_scores)

        # Should still calculate based on available scores
        assert 0 <= overall_score <= 100
        assert isinstance(risk_level, RiskLevel)

    def test_boundary_score_values(self, service):
        """Test with boundary score values."""
        # Test with exact boundary values
        category_scores = [
            CategoryScore(
                category=ForensicCategory.GASLIGHTING,
                score=0.0,  # Minimum
                confidence=0.0,  # Minimum
                evidence_count=0,
                key_indicators=[],
            ),
            CategoryScore(
                category=ForensicCategory.THREAT,
                score=100.0,  # Maximum
                confidence=1.0,  # Maximum
                evidence_count=100,
                key_indicators=["all"],
            ),
        ]

        overall_score, risk_level = service.calculate_overall_score(category_scores)

        assert 0.0 <= overall_score <= 100.0
        assert isinstance(risk_level, RiskLevel)
