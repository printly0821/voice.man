"""
Tests for Cross-Validation Service
SPEC-FORENSIC-001 Phase 2-C: Text-Voice cross-validation service tests

TDD RED Phase: These tests define the expected behavior of the CrossValidationService.
"""

import pytest
from unittest.mock import Mock, MagicMock


class TestCrossValidationServiceInit:
    """Tests for CrossValidationService initialization."""

    def test_init_with_dependencies(self):
        """Test CrossValidationService initializes with required dependencies."""
        from voice_man.services.forensic.cross_validation_service import (
            CrossValidationService,
        )
        from voice_man.services.forensic.crime_language_service import (
            CrimeLanguageAnalysisService,
        )
        from voice_man.services.forensic.ser_service import SERService

        crime_service = CrimeLanguageAnalysisService()
        ser_service = SERService(device="cpu")

        service = CrossValidationService(
            crime_language_service=crime_service,
            ser_service=ser_service,
        )

        assert service._crime_language_service is crime_service
        assert service._ser_service is ser_service

    def test_init_stores_services(self):
        """Test that services are stored correctly."""
        from voice_man.services.forensic.cross_validation_service import (
            CrossValidationService,
        )

        mock_crime_service = Mock()
        mock_ser_service = Mock()

        service = CrossValidationService(
            crime_language_service=mock_crime_service,
            ser_service=mock_ser_service,
        )

        assert service._crime_language_service is mock_crime_service
        assert service._ser_service is mock_ser_service


class TestAnalyzeText:
    """Tests for analyze_text() method."""

    def test_analyze_text_returns_valid_result(self):
        """Test analyze_text returns valid TextAnalysisResult."""
        from voice_man.services.forensic.cross_validation_service import (
            CrossValidationService,
        )
        from voice_man.models.forensic.cross_validation import TextAnalysisResult

        mock_crime_service = Mock()
        mock_crime_service.analyze_comprehensive.return_value = Mock(
            gaslighting_score=0.0,
            threat_score=0.0,
            coercion_score=0.0,
            deception_score=0.1,
            overall_risk_score=0.05,
            gaslighting_count=0,
            threat_count=0,
            coercion_count=0,
        )
        mock_crime_service.detect_gaslighting.return_value = []
        mock_crime_service.detect_threats.return_value = []
        mock_crime_service.detect_coercion.return_value = []

        service = CrossValidationService(
            crime_language_service=mock_crime_service,
            ser_service=Mock(),
        )

        result = service.analyze_text("I'm so happy to see you today!")

        assert isinstance(result, TextAnalysisResult)
        assert result.text == "I'm so happy to see you today!"
        assert result.detected_sentiment in ["positive", "negative", "neutral"]
        assert -1.0 <= result.sentiment_score <= 1.0
        assert 0.0 <= result.intensity_level <= 1.0

    def test_analyze_text_positive_sentiment(self):
        """Test analyze_text detects positive sentiment."""
        from voice_man.services.forensic.cross_validation_service import (
            CrossValidationService,
        )

        mock_crime_service = Mock()
        mock_crime_service.analyze_comprehensive.return_value = Mock(
            gaslighting_score=0.0,
            threat_score=0.0,
            coercion_score=0.0,
            deception_score=0.0,
            overall_risk_score=0.0,
            gaslighting_count=0,
            threat_count=0,
            coercion_count=0,
        )
        mock_crime_service.detect_gaslighting.return_value = []
        mock_crime_service.detect_threats.return_value = []
        mock_crime_service.detect_coercion.return_value = []

        service = CrossValidationService(
            crime_language_service=mock_crime_service,
            ser_service=Mock(),
        )

        result = service.analyze_text("I love this! It's wonderful and amazing!")

        assert result.detected_sentiment == "positive"
        assert result.sentiment_score > 0

    def test_analyze_text_negative_sentiment(self):
        """Test analyze_text detects negative sentiment."""
        from voice_man.services.forensic.cross_validation_service import (
            CrossValidationService,
        )

        mock_crime_service = Mock()
        mock_crime_service.analyze_comprehensive.return_value = Mock(
            gaslighting_score=0.0,
            threat_score=0.5,
            coercion_score=0.0,
            deception_score=0.0,
            overall_risk_score=0.2,
            gaslighting_count=0,
            threat_count=1,
            coercion_count=0,
        )
        mock_crime_service.detect_gaslighting.return_value = []
        mock_crime_service.detect_threats.return_value = [Mock(type="direct")]
        mock_crime_service.detect_coercion.return_value = []

        service = CrossValidationService(
            crime_language_service=mock_crime_service,
            ser_service=Mock(),
        )

        result = service.analyze_text("I hate this! It's terrible and awful!")

        assert result.detected_sentiment == "negative"
        assert result.sentiment_score < 0

    def test_analyze_text_detects_crime_patterns(self):
        """Test analyze_text detects crime language patterns."""
        from voice_man.services.forensic.cross_validation_service import (
            CrossValidationService,
        )

        mock_crime_service = Mock()
        mock_crime_service.analyze_comprehensive.return_value = Mock(
            gaslighting_score=0.8,
            threat_score=0.0,
            coercion_score=0.0,
            deception_score=0.3,
            overall_risk_score=0.4,
            gaslighting_count=2,
            threat_count=0,
            coercion_count=0,
        )
        mock_gaslighting_match = Mock()
        mock_gaslighting_match.type = "denial"
        mock_crime_service.detect_gaslighting.return_value = [mock_gaslighting_match]
        mock_crime_service.detect_threats.return_value = []
        mock_crime_service.detect_coercion.return_value = []

        service = CrossValidationService(
            crime_language_service=mock_crime_service,
            ser_service=Mock(),
        )

        result = service.analyze_text("That never happened. You're imagining things.")

        assert len(result.crime_patterns_found) > 0
        assert "gaslighting" in result.crime_patterns_found

    def test_analyze_text_intensity_level(self):
        """Test analyze_text calculates intensity level."""
        from voice_man.services.forensic.cross_validation_service import (
            CrossValidationService,
        )

        mock_crime_service = Mock()
        mock_crime_service.analyze_comprehensive.return_value = Mock(
            gaslighting_score=0.0,
            threat_score=0.9,
            coercion_score=0.0,
            deception_score=0.0,
            overall_risk_score=0.5,
            gaslighting_count=0,
            threat_count=2,
            coercion_count=0,
        )
        mock_crime_service.detect_gaslighting.return_value = []
        mock_threat_match = Mock()
        mock_threat_match.type = "direct"
        mock_crime_service.detect_threats.return_value = [mock_threat_match]
        mock_crime_service.detect_coercion.return_value = []

        service = CrossValidationService(
            crime_language_service=mock_crime_service,
            ser_service=Mock(),
        )

        # High intensity text with threats
        result = service.analyze_text("I WILL DESTROY YOU! YOU'RE DEAD!")

        assert result.intensity_level > 0.5

    def test_analyze_text_empty_text(self):
        """Test analyze_text handles empty text."""
        from voice_man.services.forensic.cross_validation_service import (
            CrossValidationService,
        )

        service = CrossValidationService(
            crime_language_service=Mock(),
            ser_service=Mock(),
        )

        with pytest.raises(ValueError, match="Text cannot be empty"):
            service.analyze_text("")

        with pytest.raises(ValueError, match="Text cannot be empty"):
            service.analyze_text("   ")


class TestAnalyzeVoice:
    """Tests for analyze_voice() method."""

    def test_analyze_voice_returns_valid_result(self):
        """Test analyze_voice returns valid VoiceAnalysisResult."""
        from voice_man.services.forensic.cross_validation_service import (
            CrossValidationService,
        )
        from voice_man.models.forensic.cross_validation import VoiceAnalysisResult
        from voice_man.models.forensic.emotion_recognition import (
            MultiModelEmotionResult,
            EmotionAnalysisResult,
            EmotionDimensions,
            CategoricalEmotion,
            ForensicEmotionIndicators,
        )

        mock_ser_service = Mock()
        mock_result = MultiModelEmotionResult(
            primary_result=EmotionAnalysisResult(
                dimensions=EmotionDimensions(arousal=0.7, dominance=0.5, valence=0.3),
                model_used="test-model",
                processing_time_ms=100.0,
            ),
            secondary_result=EmotionAnalysisResult(
                categorical=CategoricalEmotion(emotion_type="angry", confidence=0.8),
                model_used="test-model-2",
                processing_time_ms=80.0,
            ),
            ensemble_confidence=0.75,
            audio_duration_seconds=3.0,
        )
        mock_ser_service.analyze_ensemble_from_file.return_value = mock_result
        mock_ser_service.get_forensic_emotion_indicators.return_value = ForensicEmotionIndicators(
            high_arousal_low_valence=True,
            emotion_inconsistency_score=0.2,
            dominant_emotion="angry",
            arousal_level="high",
            stress_indicator=True,
            deception_indicator=False,
            confidence=0.75,
        )

        service = CrossValidationService(
            crime_language_service=Mock(),
            ser_service=mock_ser_service,
        )

        result = service.analyze_voice("/path/to/audio.wav")

        assert isinstance(result, VoiceAnalysisResult)
        assert result.dominant_emotion == "angry"
        assert 0.0 <= result.emotion_confidence <= 1.0
        assert 0.0 <= result.arousal <= 1.0
        assert 0.0 <= result.valence <= 1.0
        assert 0.0 <= result.stress_level <= 1.0

    def test_analyze_voice_extracts_dimensions(self):
        """Test analyze_voice extracts emotion dimensions correctly."""
        from voice_man.services.forensic.cross_validation_service import (
            CrossValidationService,
        )
        from voice_man.models.forensic.emotion_recognition import (
            MultiModelEmotionResult,
            EmotionAnalysisResult,
            EmotionDimensions,
            ForensicEmotionIndicators,
        )

        mock_ser_service = Mock()
        mock_result = MultiModelEmotionResult(
            primary_result=EmotionAnalysisResult(
                dimensions=EmotionDimensions(arousal=0.85, dominance=0.6, valence=0.2),
                model_used="test-model",
                processing_time_ms=100.0,
            ),
            ensemble_confidence=0.7,
            audio_duration_seconds=2.5,
        )
        mock_ser_service.analyze_ensemble_from_file.return_value = mock_result
        mock_ser_service.get_forensic_emotion_indicators.return_value = ForensicEmotionIndicators(
            high_arousal_low_valence=True,
            emotion_inconsistency_score=0.1,
            dominant_emotion="angry",
            arousal_level="high",
            stress_indicator=True,
            deception_indicator=False,
            confidence=0.7,
        )

        service = CrossValidationService(
            crime_language_service=Mock(),
            ser_service=mock_ser_service,
        )

        result = service.analyze_voice("/path/to/audio.wav")

        assert result.arousal == 0.85
        assert result.valence == 0.2

    def test_analyze_voice_stress_from_indicators(self):
        """Test analyze_voice extracts stress level from forensic indicators."""
        from voice_man.services.forensic.cross_validation_service import (
            CrossValidationService,
        )
        from voice_man.models.forensic.emotion_recognition import (
            MultiModelEmotionResult,
            EmotionAnalysisResult,
            EmotionDimensions,
            ForensicEmotionIndicators,
        )

        mock_ser_service = Mock()
        mock_result = MultiModelEmotionResult(
            primary_result=EmotionAnalysisResult(
                dimensions=EmotionDimensions(arousal=0.9, dominance=0.5, valence=0.1),
                model_used="test-model",
                processing_time_ms=100.0,
            ),
            ensemble_confidence=0.8,
            audio_duration_seconds=3.0,
        )
        mock_ser_service.analyze_ensemble_from_file.return_value = mock_result
        mock_ser_service.get_forensic_emotion_indicators.return_value = ForensicEmotionIndicators(
            high_arousal_low_valence=True,
            emotion_inconsistency_score=0.3,
            dominant_emotion="fear",
            arousal_level="high",
            stress_indicator=True,
            deception_indicator=False,
            confidence=0.8,
        )

        service = CrossValidationService(
            crime_language_service=Mock(),
            ser_service=mock_ser_service,
        )

        result = service.analyze_voice("/path/to/audio.wav")

        # High arousal + low valence = high stress
        assert result.stress_level > 0.5


class TestDetectDiscrepancies:
    """Tests for detect_discrepancies() method."""

    def test_detect_emotion_text_mismatch_positive_text_negative_voice(self):
        """Test EMOTION_TEXT_MISMATCH: positive text + negative voice."""
        from voice_man.services.forensic.cross_validation_service import (
            CrossValidationService,
        )
        from voice_man.models.forensic.cross_validation import (
            TextAnalysisResult,
            VoiceAnalysisResult,
            DiscrepancyType,
            DiscrepancySeverity,
        )

        service = CrossValidationService(
            crime_language_service=Mock(),
            ser_service=Mock(),
        )

        text_result = TextAnalysisResult(
            text="I'm so happy!",
            detected_sentiment="positive",
            sentiment_score=0.8,
            detected_emotions=["joy"],
            crime_patterns_found=[],
            intensity_level=0.5,
        )

        voice_result = VoiceAnalysisResult(
            dominant_emotion="angry",
            emotion_confidence=0.85,
            arousal=0.7,
            valence=0.2,  # Low valence = negative
            stress_level=0.6,
        )

        discrepancies = service.detect_discrepancies(text_result, voice_result)

        assert len(discrepancies) > 0
        emotion_mismatch = next(
            (
                d
                for d in discrepancies
                if d.discrepancy_type == DiscrepancyType.EMOTION_TEXT_MISMATCH.value
            ),
            None,
        )
        assert emotion_mismatch is not None
        assert emotion_mismatch.severity in [
            DiscrepancySeverity.HIGH.value,
            DiscrepancySeverity.CRITICAL.value,
        ]

    def test_detect_emotion_text_mismatch_negative_text_positive_voice(self):
        """Test EMOTION_TEXT_MISMATCH: negative text + positive voice."""
        from voice_man.services.forensic.cross_validation_service import (
            CrossValidationService,
        )
        from voice_man.models.forensic.cross_validation import (
            TextAnalysisResult,
            VoiceAnalysisResult,
            DiscrepancyType,
        )

        service = CrossValidationService(
            crime_language_service=Mock(),
            ser_service=Mock(),
        )

        text_result = TextAnalysisResult(
            text="I hate this so much!",
            detected_sentiment="negative",
            sentiment_score=-0.7,
            detected_emotions=["anger"],
            crime_patterns_found=[],
            intensity_level=0.6,
        )

        voice_result = VoiceAnalysisResult(
            dominant_emotion="happy",
            emotion_confidence=0.8,
            arousal=0.6,
            valence=0.8,  # High valence = positive
            stress_level=0.2,
        )

        discrepancies = service.detect_discrepancies(text_result, voice_result)

        emotion_mismatch = next(
            (
                d
                for d in discrepancies
                if d.discrepancy_type == DiscrepancyType.EMOTION_TEXT_MISMATCH.value
            ),
            None,
        )
        assert emotion_mismatch is not None

    def test_detect_emotion_text_mismatch_joy_text_sad_voice(self):
        """Test EMOTION_TEXT_MISMATCH: joy in text + sad voice = CRITICAL."""
        from voice_man.services.forensic.cross_validation_service import (
            CrossValidationService,
        )
        from voice_man.models.forensic.cross_validation import (
            TextAnalysisResult,
            VoiceAnalysisResult,
            DiscrepancyType,
            DiscrepancySeverity,
        )

        service = CrossValidationService(
            crime_language_service=Mock(),
            ser_service=Mock(),
        )

        text_result = TextAnalysisResult(
            text="This is the best day ever! I'm thrilled!",
            detected_sentiment="positive",
            sentiment_score=0.9,
            detected_emotions=["joy", "excitement"],
            crime_patterns_found=[],
            intensity_level=0.7,
        )

        voice_result = VoiceAnalysisResult(
            dominant_emotion="sad",
            emotion_confidence=0.85,
            arousal=0.3,
            valence=0.2,
            stress_level=0.4,
        )

        discrepancies = service.detect_discrepancies(text_result, voice_result)

        emotion_mismatch = next(
            (
                d
                for d in discrepancies
                if d.discrepancy_type == DiscrepancyType.EMOTION_TEXT_MISMATCH.value
            ),
            None,
        )
        assert emotion_mismatch is not None
        assert emotion_mismatch.severity == DiscrepancySeverity.CRITICAL.value

    def test_detect_intensity_mismatch_strong_text_low_arousal(self):
        """Test INTENSITY_MISMATCH: strong text + low arousal."""
        from voice_man.services.forensic.cross_validation_service import (
            CrossValidationService,
        )
        from voice_man.models.forensic.cross_validation import (
            TextAnalysisResult,
            VoiceAnalysisResult,
            DiscrepancyType,
            DiscrepancySeverity,
        )

        service = CrossValidationService(
            crime_language_service=Mock(),
            ser_service=Mock(),
        )

        text_result = TextAnalysisResult(
            text="I WILL ABSOLUTELY DESTROY EVERYTHING!",
            detected_sentiment="negative",
            sentiment_score=-0.8,
            detected_emotions=["anger", "rage"],
            crime_patterns_found=[],
            intensity_level=0.9,  # High intensity
        )

        voice_result = VoiceAnalysisResult(
            dominant_emotion="neutral",
            emotion_confidence=0.7,
            arousal=0.2,  # Low arousal
            valence=0.5,
            stress_level=0.1,
        )

        discrepancies = service.detect_discrepancies(text_result, voice_result)

        intensity_mismatch = next(
            (
                d
                for d in discrepancies
                if d.discrepancy_type == DiscrepancyType.INTENSITY_MISMATCH.value
            ),
            None,
        )
        assert intensity_mismatch is not None
        assert intensity_mismatch.severity == DiscrepancySeverity.HIGH.value

    def test_detect_intensity_mismatch_threat_with_low_arousal_critical(self):
        """Test INTENSITY_MISMATCH: threat pattern + low arousal = CRITICAL."""
        from voice_man.services.forensic.cross_validation_service import (
            CrossValidationService,
        )
        from voice_man.models.forensic.cross_validation import (
            TextAnalysisResult,
            VoiceAnalysisResult,
            DiscrepancyType,
            DiscrepancySeverity,
        )

        service = CrossValidationService(
            crime_language_service=Mock(),
            ser_service=Mock(),
        )

        text_result = TextAnalysisResult(
            text="I'm going to hurt you",
            detected_sentiment="negative",
            sentiment_score=-0.9,
            detected_emotions=["anger"],
            crime_patterns_found=["threat"],  # Threat pattern detected
            intensity_level=0.85,
        )

        voice_result = VoiceAnalysisResult(
            dominant_emotion="neutral",
            emotion_confidence=0.8,
            arousal=0.15,  # Very low arousal
            valence=0.5,
            stress_level=0.1,
        )

        discrepancies = service.detect_discrepancies(text_result, voice_result)

        intensity_mismatch = next(
            (
                d
                for d in discrepancies
                if d.discrepancy_type == DiscrepancyType.INTENSITY_MISMATCH.value
            ),
            None,
        )
        assert intensity_mismatch is not None
        assert intensity_mismatch.severity == DiscrepancySeverity.CRITICAL.value

    def test_detect_sentiment_contradiction_apology_with_angry_voice(self):
        """Test SENTIMENT_CONTRADICTION: apology text + angry voice."""
        from voice_man.services.forensic.cross_validation_service import (
            CrossValidationService,
        )
        from voice_man.models.forensic.cross_validation import (
            TextAnalysisResult,
            VoiceAnalysisResult,
            DiscrepancyType,
            DiscrepancySeverity,
        )

        service = CrossValidationService(
            crime_language_service=Mock(),
            ser_service=Mock(),
        )

        text_result = TextAnalysisResult(
            text="I'm so sorry. Please forgive me.",
            detected_sentiment="negative",  # Apology can be slightly negative
            sentiment_score=-0.2,
            detected_emotions=["remorse", "sadness"],
            crime_patterns_found=[],
            intensity_level=0.4,
        )

        voice_result = VoiceAnalysisResult(
            dominant_emotion="angry",
            emotion_confidence=0.85,
            arousal=0.8,
            valence=0.2,
            stress_level=0.7,
        )

        discrepancies = service.detect_discrepancies(text_result, voice_result)

        sentiment_contradiction = next(
            (
                d
                for d in discrepancies
                if d.discrepancy_type == DiscrepancyType.SENTIMENT_CONTRADICTION.value
            ),
            None,
        )
        assert sentiment_contradiction is not None
        assert sentiment_contradiction.severity == DiscrepancySeverity.HIGH.value

    def test_detect_sentiment_contradiction_comfort_with_fear_voice(self):
        """Test SENTIMENT_CONTRADICTION: comfort text + fear voice = MEDIUM."""
        from voice_man.services.forensic.cross_validation_service import (
            CrossValidationService,
        )
        from voice_man.models.forensic.cross_validation import (
            TextAnalysisResult,
            VoiceAnalysisResult,
            DiscrepancyType,
            DiscrepancySeverity,
        )

        service = CrossValidationService(
            crime_language_service=Mock(),
            ser_service=Mock(),
        )

        text_result = TextAnalysisResult(
            text="Don't worry, everything will be okay.",
            detected_sentiment="positive",
            sentiment_score=0.5,
            detected_emotions=["comfort", "reassurance"],
            crime_patterns_found=[],
            intensity_level=0.3,
        )

        voice_result = VoiceAnalysisResult(
            dominant_emotion="fear",
            emotion_confidence=0.75,
            arousal=0.7,
            valence=0.25,
            stress_level=0.65,
        )

        discrepancies = service.detect_discrepancies(text_result, voice_result)

        sentiment_contradiction = next(
            (
                d
                for d in discrepancies
                if d.discrepancy_type == DiscrepancyType.SENTIMENT_CONTRADICTION.value
            ),
            None,
        )
        assert sentiment_contradiction is not None
        assert sentiment_contradiction.severity == DiscrepancySeverity.MEDIUM.value

    def test_detect_stress_content_mismatch_casual_with_high_stress(self):
        """Test STRESS_CONTENT_MISMATCH: casual content + high stress."""
        from voice_man.services.forensic.cross_validation_service import (
            CrossValidationService,
        )
        from voice_man.models.forensic.cross_validation import (
            TextAnalysisResult,
            VoiceAnalysisResult,
            DiscrepancyType,
            DiscrepancySeverity,
        )

        service = CrossValidationService(
            crime_language_service=Mock(),
            ser_service=Mock(),
        )

        text_result = TextAnalysisResult(
            text="The weather is nice today.",
            detected_sentiment="neutral",
            sentiment_score=0.1,
            detected_emotions=[],
            crime_patterns_found=[],
            intensity_level=0.1,  # Very casual/low intensity
        )

        voice_result = VoiceAnalysisResult(
            dominant_emotion="neutral",
            emotion_confidence=0.6,
            arousal=0.8,
            valence=0.4,
            stress_level=0.8,  # High stress
        )

        discrepancies = service.detect_discrepancies(text_result, voice_result)

        stress_mismatch = next(
            (
                d
                for d in discrepancies
                if d.discrepancy_type == DiscrepancyType.STRESS_CONTENT_MISMATCH.value
            ),
            None,
        )
        assert stress_mismatch is not None
        assert stress_mismatch.severity == DiscrepancySeverity.MEDIUM.value

    def test_detect_no_discrepancies_when_consistent(self):
        """Test no discrepancies when text and voice are consistent."""
        from voice_man.services.forensic.cross_validation_service import (
            CrossValidationService,
        )
        from voice_man.models.forensic.cross_validation import (
            TextAnalysisResult,
            VoiceAnalysisResult,
        )

        service = CrossValidationService(
            crime_language_service=Mock(),
            ser_service=Mock(),
        )

        text_result = TextAnalysisResult(
            text="I'm feeling happy today!",
            detected_sentiment="positive",
            sentiment_score=0.7,
            detected_emotions=["joy"],
            crime_patterns_found=[],
            intensity_level=0.5,
        )

        voice_result = VoiceAnalysisResult(
            dominant_emotion="happy",
            emotion_confidence=0.8,
            arousal=0.6,
            valence=0.75,  # High valence = positive
            stress_level=0.2,
        )

        discrepancies = service.detect_discrepancies(text_result, voice_result)

        assert len(discrepancies) == 0


class TestCalculateDeceptionProbability:
    """Tests for calculate_deception_probability() method."""

    def test_calculate_deception_no_discrepancies(self):
        """Test deception probability is low with no discrepancies."""
        from voice_man.services.forensic.cross_validation_service import (
            CrossValidationService,
        )

        service = CrossValidationService(
            crime_language_service=Mock(),
            ser_service=Mock(),
        )

        probability = service.calculate_deception_probability([])

        assert probability < 0.2

    def test_calculate_deception_single_low_severity(self):
        """Test deception probability with single low severity discrepancy."""
        from voice_man.services.forensic.cross_validation_service import (
            CrossValidationService,
        )
        from voice_man.models.forensic.cross_validation import (
            Discrepancy,
            DiscrepancyType,
            DiscrepancySeverity,
        )

        service = CrossValidationService(
            crime_language_service=Mock(),
            ser_service=Mock(),
        )

        discrepancies = [
            Discrepancy(
                discrepancy_type=DiscrepancyType.STRESS_CONTENT_MISMATCH,
                severity=DiscrepancySeverity.LOW,
                description="Minor stress mismatch",
                confidence=0.6,
            )
        ]

        probability = service.calculate_deception_probability(discrepancies)

        # Low severity with moderate confidence should give low-moderate probability
        assert 0.05 <= probability <= 0.4

    def test_calculate_deception_multiple_high_severity(self):
        """Test deception probability with multiple high severity discrepancies."""
        from voice_man.services.forensic.cross_validation_service import (
            CrossValidationService,
        )
        from voice_man.models.forensic.cross_validation import (
            Discrepancy,
            DiscrepancyType,
            DiscrepancySeverity,
        )

        service = CrossValidationService(
            crime_language_service=Mock(),
            ser_service=Mock(),
        )

        discrepancies = [
            Discrepancy(
                discrepancy_type=DiscrepancyType.EMOTION_TEXT_MISMATCH,
                severity=DiscrepancySeverity.HIGH,
                description="Emotion mismatch",
                confidence=0.85,
            ),
            Discrepancy(
                discrepancy_type=DiscrepancyType.INTENSITY_MISMATCH,
                severity=DiscrepancySeverity.HIGH,
                description="Intensity mismatch",
                confidence=0.8,
            ),
        ]

        probability = service.calculate_deception_probability(discrepancies)

        # Multiple high severity discrepancies should give high probability
        assert probability >= 0.5

    def test_calculate_deception_critical_severity(self):
        """Test deception probability with critical severity discrepancy."""
        from voice_man.services.forensic.cross_validation_service import (
            CrossValidationService,
        )
        from voice_man.models.forensic.cross_validation import (
            Discrepancy,
            DiscrepancyType,
            DiscrepancySeverity,
        )

        service = CrossValidationService(
            crime_language_service=Mock(),
            ser_service=Mock(),
        )

        discrepancies = [
            Discrepancy(
                discrepancy_type=DiscrepancyType.EMOTION_TEXT_MISMATCH,
                severity=DiscrepancySeverity.CRITICAL,
                description="Critical emotion mismatch",
                confidence=0.9,
            )
        ]

        probability = service.calculate_deception_probability(discrepancies)

        assert probability >= 0.7

    def test_calculate_deception_weighted_by_confidence(self):
        """Test deception probability is weighted by confidence scores."""
        from voice_man.services.forensic.cross_validation_service import (
            CrossValidationService,
        )
        from voice_man.models.forensic.cross_validation import (
            Discrepancy,
            DiscrepancyType,
            DiscrepancySeverity,
        )

        service = CrossValidationService(
            crime_language_service=Mock(),
            ser_service=Mock(),
        )

        # High confidence discrepancy
        high_conf = [
            Discrepancy(
                discrepancy_type=DiscrepancyType.EMOTION_TEXT_MISMATCH,
                severity=DiscrepancySeverity.MEDIUM,
                description="Test",
                confidence=0.95,
            )
        ]

        # Low confidence discrepancy
        low_conf = [
            Discrepancy(
                discrepancy_type=DiscrepancyType.EMOTION_TEXT_MISMATCH,
                severity=DiscrepancySeverity.MEDIUM,
                description="Test",
                confidence=0.3,
            )
        ]

        prob_high = service.calculate_deception_probability(high_conf)
        prob_low = service.calculate_deception_probability(low_conf)

        assert prob_high > prob_low


class TestCrossValidate:
    """Tests for cross_validate() method (main entry point)."""

    def test_cross_validate_returns_complete_result(self):
        """Test cross_validate returns complete CrossValidationResult."""
        from voice_man.services.forensic.cross_validation_service import (
            CrossValidationService,
        )
        from voice_man.models.forensic.cross_validation import CrossValidationResult
        from voice_man.models.forensic.emotion_recognition import (
            MultiModelEmotionResult,
            EmotionAnalysisResult,
            EmotionDimensions,
            CategoricalEmotion,
            ForensicEmotionIndicators,
        )

        mock_crime_service = Mock()
        mock_crime_service.analyze_comprehensive.return_value = Mock(
            gaslighting_score=0.0,
            threat_score=0.0,
            coercion_score=0.0,
            deception_score=0.0,
            overall_risk_score=0.0,
            gaslighting_count=0,
            threat_count=0,
            coercion_count=0,
        )
        mock_crime_service.detect_gaslighting.return_value = []
        mock_crime_service.detect_threats.return_value = []
        mock_crime_service.detect_coercion.return_value = []

        mock_ser_service = Mock()
        mock_result = MultiModelEmotionResult(
            primary_result=EmotionAnalysisResult(
                dimensions=EmotionDimensions(arousal=0.6, dominance=0.5, valence=0.7),
                model_used="test-model",
                processing_time_ms=100.0,
            ),
            secondary_result=EmotionAnalysisResult(
                categorical=CategoricalEmotion(emotion_type="happy", confidence=0.8),
                model_used="test-model-2",
                processing_time_ms=80.0,
            ),
            ensemble_confidence=0.75,
            audio_duration_seconds=3.0,
        )
        mock_ser_service.analyze_ensemble_from_file.return_value = mock_result
        mock_ser_service.get_forensic_emotion_indicators.return_value = ForensicEmotionIndicators(
            high_arousal_low_valence=False,
            emotion_inconsistency_score=0.1,
            dominant_emotion="happy",
            arousal_level="medium",
            stress_indicator=False,
            deception_indicator=False,
            confidence=0.75,
        )

        service = CrossValidationService(
            crime_language_service=mock_crime_service,
            ser_service=mock_ser_service,
        )

        result = service.cross_validate(
            text="I'm happy to be here!",
            audio_path="/path/to/audio.wav",
        )

        assert isinstance(result, CrossValidationResult)
        assert result.text_analysis is not None
        assert result.voice_analysis is not None
        assert isinstance(result.discrepancies, list)
        assert 0.0 <= result.overall_consistency_score <= 1.0
        assert 0.0 <= result.deception_probability <= 1.0
        assert result.risk_level is not None

    def test_cross_validate_detects_discrepancies(self):
        """Test cross_validate correctly detects discrepancies."""
        from voice_man.services.forensic.cross_validation_service import (
            CrossValidationService,
        )
        from voice_man.models.forensic.emotion_recognition import (
            MultiModelEmotionResult,
            EmotionAnalysisResult,
            EmotionDimensions,
            CategoricalEmotion,
            ForensicEmotionIndicators,
        )

        mock_crime_service = Mock()
        mock_crime_service.analyze_comprehensive.return_value = Mock(
            gaslighting_score=0.0,
            threat_score=0.0,
            coercion_score=0.0,
            deception_score=0.0,
            overall_risk_score=0.0,
            gaslighting_count=0,
            threat_count=0,
            coercion_count=0,
        )
        mock_crime_service.detect_gaslighting.return_value = []
        mock_crime_service.detect_threats.return_value = []
        mock_crime_service.detect_coercion.return_value = []

        mock_ser_service = Mock()
        # Angry voice for positive text
        mock_result = MultiModelEmotionResult(
            primary_result=EmotionAnalysisResult(
                dimensions=EmotionDimensions(arousal=0.8, dominance=0.6, valence=0.2),
                model_used="test-model",
                processing_time_ms=100.0,
            ),
            secondary_result=EmotionAnalysisResult(
                categorical=CategoricalEmotion(emotion_type="angry", confidence=0.85),
                model_used="test-model-2",
                processing_time_ms=80.0,
            ),
            ensemble_confidence=0.8,
            audio_duration_seconds=3.0,
        )
        mock_ser_service.analyze_ensemble_from_file.return_value = mock_result
        mock_ser_service.get_forensic_emotion_indicators.return_value = ForensicEmotionIndicators(
            high_arousal_low_valence=True,
            emotion_inconsistency_score=0.3,
            dominant_emotion="angry",
            arousal_level="high",
            stress_indicator=True,
            deception_indicator=False,
            confidence=0.8,
        )

        service = CrossValidationService(
            crime_language_service=mock_crime_service,
            ser_service=mock_ser_service,
        )

        result = service.cross_validate(
            text="I'm so happy and grateful!",  # Positive text
            audio_path="/path/to/audio.wav",  # Angry voice
        )

        assert len(result.discrepancies) > 0
        assert result.deception_probability > 0.3

    def test_cross_validate_high_consistency_low_deception(self):
        """Test consistent text/voice yields high consistency, low deception."""
        from voice_man.services.forensic.cross_validation_service import (
            CrossValidationService,
        )
        from voice_man.models.forensic.emotion_recognition import (
            MultiModelEmotionResult,
            EmotionAnalysisResult,
            EmotionDimensions,
            CategoricalEmotion,
            ForensicEmotionIndicators,
        )

        mock_crime_service = Mock()
        mock_crime_service.analyze_comprehensive.return_value = Mock(
            gaslighting_score=0.0,
            threat_score=0.0,
            coercion_score=0.0,
            deception_score=0.0,
            overall_risk_score=0.0,
            gaslighting_count=0,
            threat_count=0,
            coercion_count=0,
        )
        mock_crime_service.detect_gaslighting.return_value = []
        mock_crime_service.detect_threats.return_value = []
        mock_crime_service.detect_coercion.return_value = []

        mock_ser_service = Mock()
        mock_result = MultiModelEmotionResult(
            primary_result=EmotionAnalysisResult(
                dimensions=EmotionDimensions(arousal=0.6, dominance=0.5, valence=0.8),
                model_used="test-model",
                processing_time_ms=100.0,
            ),
            secondary_result=EmotionAnalysisResult(
                categorical=CategoricalEmotion(emotion_type="happy", confidence=0.85),
                model_used="test-model-2",
                processing_time_ms=80.0,
            ),
            ensemble_confidence=0.85,
            audio_duration_seconds=3.0,
        )
        mock_ser_service.analyze_ensemble_from_file.return_value = mock_result
        mock_ser_service.get_forensic_emotion_indicators.return_value = ForensicEmotionIndicators(
            high_arousal_low_valence=False,
            emotion_inconsistency_score=0.1,
            dominant_emotion="happy",
            arousal_level="medium",
            stress_indicator=False,
            deception_indicator=False,
            confidence=0.85,
        )

        service = CrossValidationService(
            crime_language_service=mock_crime_service,
            ser_service=mock_ser_service,
        )

        result = service.cross_validate(
            text="I'm so happy!",  # Positive text
            audio_path="/path/to/audio.wav",  # Happy voice
        )

        assert result.overall_consistency_score > 0.7
        assert result.deception_probability < 0.3

    def test_cross_validate_assigns_correct_risk_level(self):
        """Test cross_validate assigns correct risk level based on discrepancies."""
        from voice_man.services.forensic.cross_validation_service import (
            CrossValidationService,
        )
        from voice_man.models.forensic.cross_validation import DiscrepancySeverity
        from voice_man.models.forensic.emotion_recognition import (
            MultiModelEmotionResult,
            EmotionAnalysisResult,
            EmotionDimensions,
            CategoricalEmotion,
            ForensicEmotionIndicators,
        )

        mock_crime_service = Mock()
        mock_crime_service.analyze_comprehensive.return_value = Mock(
            gaslighting_score=0.0,
            threat_score=0.9,
            coercion_score=0.0,
            deception_score=0.0,
            overall_risk_score=0.5,
            gaslighting_count=0,
            threat_count=1,
            coercion_count=0,
        )
        mock_crime_service.detect_gaslighting.return_value = []
        mock_threat = Mock()
        mock_threat.type = "direct"
        mock_crime_service.detect_threats.return_value = [mock_threat]
        mock_crime_service.detect_coercion.return_value = []

        mock_ser_service = Mock()
        mock_result = MultiModelEmotionResult(
            primary_result=EmotionAnalysisResult(
                dimensions=EmotionDimensions(arousal=0.2, dominance=0.5, valence=0.5),
                model_used="test-model",
                processing_time_ms=100.0,
            ),
            secondary_result=EmotionAnalysisResult(
                categorical=CategoricalEmotion(emotion_type="neutral", confidence=0.7),
                model_used="test-model-2",
                processing_time_ms=80.0,
            ),
            ensemble_confidence=0.7,
            audio_duration_seconds=3.0,
        )
        mock_ser_service.analyze_ensemble_from_file.return_value = mock_result
        mock_ser_service.get_forensic_emotion_indicators.return_value = ForensicEmotionIndicators(
            high_arousal_low_valence=False,
            emotion_inconsistency_score=0.2,
            dominant_emotion="neutral",
            arousal_level="low",
            stress_indicator=False,
            deception_indicator=False,
            confidence=0.7,
        )

        service = CrossValidationService(
            crime_language_service=mock_crime_service,
            ser_service=mock_ser_service,
        )

        # Threatening text with calm voice should trigger CRITICAL
        result = service.cross_validate(
            text="I will hurt you if you don't comply",
            audio_path="/path/to/audio.wav",
        )

        assert result.risk_level in [
            DiscrepancySeverity.HIGH.value,
            DiscrepancySeverity.CRITICAL.value,
        ]


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_text_raises_error(self):
        """Test empty text raises ValueError."""
        from voice_man.services.forensic.cross_validation_service import (
            CrossValidationService,
        )

        service = CrossValidationService(
            crime_language_service=Mock(),
            ser_service=Mock(),
        )

        with pytest.raises(ValueError, match="Text cannot be empty"):
            service.cross_validate(text="", audio_path="/path/to/audio.wav")

    def test_whitespace_only_text_raises_error(self):
        """Test whitespace-only text raises ValueError."""
        from voice_man.services.forensic.cross_validation_service import (
            CrossValidationService,
        )

        service = CrossValidationService(
            crime_language_service=Mock(),
            ser_service=Mock(),
        )

        with pytest.raises(ValueError, match="Text cannot be empty"):
            service.cross_validate(text="   \t\n  ", audio_path="/path/to/audio.wav")

    def test_audio_file_not_found(self):
        """Test handling of non-existent audio file."""
        from voice_man.services.forensic.cross_validation_service import (
            CrossValidationService,
        )

        mock_crime_service = Mock()
        mock_crime_service.analyze_comprehensive.return_value = Mock(
            gaslighting_score=0.0,
            threat_score=0.0,
            coercion_score=0.0,
            deception_score=0.0,
            overall_risk_score=0.0,
            gaslighting_count=0,
            threat_count=0,
            coercion_count=0,
        )
        mock_crime_service.detect_gaslighting.return_value = []
        mock_crime_service.detect_threats.return_value = []
        mock_crime_service.detect_coercion.return_value = []

        mock_ser_service = Mock()
        mock_ser_service.analyze_ensemble_from_file.side_effect = FileNotFoundError(
            "Audio file not found"
        )

        service = CrossValidationService(
            crime_language_service=mock_crime_service,
            ser_service=mock_ser_service,
        )

        with pytest.raises(FileNotFoundError):
            service.cross_validate(
                text="Hello world",
                audio_path="/nonexistent/audio.wav",
            )


class TestServiceImport:
    """Tests for service import from package."""

    def test_import_from_forensic_services_package(self):
        """Test importing service from voice_man.services.forensic package."""
        from voice_man.services.forensic.cross_validation_service import (
            CrossValidationService,
        )

        assert CrossValidationService is not None


class TestAdditionalEmotionDetection:
    """Additional tests for emotion detection coverage."""

    def test_detect_sadness_in_text(self):
        """Test detecting sadness emotion in text."""
        from voice_man.services.forensic.cross_validation_service import (
            CrossValidationService,
        )

        mock_crime_service = Mock()
        mock_crime_service.analyze_comprehensive.return_value = Mock(
            gaslighting_score=0.0,
            threat_score=0.0,
            coercion_score=0.0,
            deception_score=0.0,
            overall_risk_score=0.0,
            gaslighting_count=0,
            threat_count=0,
            coercion_count=0,
        )
        mock_crime_service.detect_gaslighting.return_value = []
        mock_crime_service.detect_threats.return_value = []
        mock_crime_service.detect_coercion.return_value = []

        service = CrossValidationService(
            crime_language_service=mock_crime_service,
            ser_service=Mock(),
        )

        result = service.analyze_text("I feel so sad and depressed today")

        assert "sadness" in result.detected_emotions

    def test_detect_fear_in_text(self):
        """Test detecting fear emotion in text."""
        from voice_man.services.forensic.cross_validation_service import (
            CrossValidationService,
        )

        mock_crime_service = Mock()
        mock_crime_service.analyze_comprehensive.return_value = Mock(
            gaslighting_score=0.0,
            threat_score=0.0,
            coercion_score=0.0,
            deception_score=0.0,
            overall_risk_score=0.0,
            gaslighting_count=0,
            threat_count=0,
            coercion_count=0,
        )
        mock_crime_service.detect_gaslighting.return_value = []
        mock_crime_service.detect_threats.return_value = []
        mock_crime_service.detect_coercion.return_value = []

        service = CrossValidationService(
            crime_language_service=mock_crime_service,
            ser_service=Mock(),
        )

        result = service.analyze_text("I'm so afraid and scared of what might happen")

        assert "fear" in result.detected_emotions

    def test_detect_remorse_in_text(self):
        """Test detecting remorse/apology in text."""
        from voice_man.services.forensic.cross_validation_service import (
            CrossValidationService,
        )

        mock_crime_service = Mock()
        mock_crime_service.analyze_comprehensive.return_value = Mock(
            gaslighting_score=0.0,
            threat_score=0.0,
            coercion_score=0.0,
            deception_score=0.0,
            overall_risk_score=0.0,
            gaslighting_count=0,
            threat_count=0,
            coercion_count=0,
        )
        mock_crime_service.detect_gaslighting.return_value = []
        mock_crime_service.detect_threats.return_value = []
        mock_crime_service.detect_coercion.return_value = []

        service = CrossValidationService(
            crime_language_service=mock_crime_service,
            ser_service=Mock(),
        )

        result = service.analyze_text("I'm sorry for what I did. Please forgive me.")

        assert "remorse" in result.detected_emotions

    def test_detect_comfort_in_text(self):
        """Test detecting comfort/reassurance in text."""
        from voice_man.services.forensic.cross_validation_service import (
            CrossValidationService,
        )

        mock_crime_service = Mock()
        mock_crime_service.analyze_comprehensive.return_value = Mock(
            gaslighting_score=0.0,
            threat_score=0.0,
            coercion_score=0.0,
            deception_score=0.0,
            overall_risk_score=0.0,
            gaslighting_count=0,
            threat_count=0,
            coercion_count=0,
        )
        mock_crime_service.detect_gaslighting.return_value = []
        mock_crime_service.detect_threats.return_value = []
        mock_crime_service.detect_coercion.return_value = []

        service = CrossValidationService(
            crime_language_service=mock_crime_service,
            ser_service=Mock(),
        )

        result = service.analyze_text("Don't worry, everything will be okay.")

        assert "comfort" in result.detected_emotions

    def test_detect_coercion_pattern(self):
        """Test detecting coercion crime pattern."""
        from voice_man.services.forensic.cross_validation_service import (
            CrossValidationService,
        )

        mock_crime_service = Mock()
        mock_crime_service.analyze_comprehensive.return_value = Mock(
            gaslighting_score=0.0,
            threat_score=0.0,
            coercion_score=0.8,
            deception_score=0.0,
            overall_risk_score=0.3,
            gaslighting_count=0,
            threat_count=0,
            coercion_count=2,
        )
        mock_crime_service.detect_gaslighting.return_value = []
        mock_crime_service.detect_threats.return_value = []
        mock_coercion = Mock()
        mock_coercion.type = "emotional"
        mock_crime_service.detect_coercion.return_value = [mock_coercion]

        service = CrossValidationService(
            crime_language_service=mock_crime_service,
            ser_service=Mock(),
        )

        result = service.analyze_text("If you really loved me, you would do this.")

        assert "coercion" in result.crime_patterns_found


class TestRiskLevelDetermination:
    """Tests for risk level determination coverage."""

    def test_risk_level_medium(self):
        """Test risk level determination returns MEDIUM."""
        from voice_man.services.forensic.cross_validation_service import (
            CrossValidationService,
        )
        from voice_man.models.forensic.cross_validation import (
            Discrepancy,
            DiscrepancyType,
            DiscrepancySeverity,
        )

        service = CrossValidationService(
            crime_language_service=Mock(),
            ser_service=Mock(),
        )

        discrepancies = [
            Discrepancy(
                discrepancy_type=DiscrepancyType.STRESS_CONTENT_MISMATCH,
                severity=DiscrepancySeverity.MEDIUM,
                description="Medium severity issue",
                confidence=0.7,
            )
        ]

        risk_level = service._determine_risk_level(discrepancies)

        assert risk_level == DiscrepancySeverity.MEDIUM

    def test_risk_level_low_only(self):
        """Test risk level determination returns LOW when only LOW discrepancies."""
        from voice_man.services.forensic.cross_validation_service import (
            CrossValidationService,
        )
        from voice_man.models.forensic.cross_validation import (
            Discrepancy,
            DiscrepancyType,
            DiscrepancySeverity,
        )

        service = CrossValidationService(
            crime_language_service=Mock(),
            ser_service=Mock(),
        )

        discrepancies = [
            Discrepancy(
                discrepancy_type=DiscrepancyType.STRESS_CONTENT_MISMATCH,
                severity=DiscrepancySeverity.LOW,
                description="Low severity issue",
                confidence=0.5,
            )
        ]

        risk_level = service._determine_risk_level(discrepancies)

        assert risk_level == DiscrepancySeverity.LOW

    def test_deception_probability_zero_confidence(self):
        """Test deception probability when confidence is zero."""
        from voice_man.services.forensic.cross_validation_service import (
            CrossValidationService,
        )
        from voice_man.models.forensic.cross_validation import (
            Discrepancy,
            DiscrepancyType,
            DiscrepancySeverity,
        )

        service = CrossValidationService(
            crime_language_service=Mock(),
            ser_service=Mock(),
        )

        # Edge case: discrepancy with zero confidence
        discrepancies = [
            Discrepancy(
                discrepancy_type=DiscrepancyType.STRESS_CONTENT_MISMATCH,
                severity=DiscrepancySeverity.LOW,
                description="Zero confidence issue",
                confidence=0.0,
            )
        ]

        probability = service.calculate_deception_probability(discrepancies)

        # Should return a small probability
        assert 0.0 <= probability <= 0.2
