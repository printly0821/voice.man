"""
Tests for Emotion Recognition Models
SPEC-FORENSIC-001 Phase 2-B: Pydantic model tests for SER

TDD RED Phase: These tests define the expected behavior of the emotion recognition models.
"""

import pytest
from pydantic import ValidationError


class TestEmotionDimensions:
    """Tests for EmotionDimensions model."""

    def test_valid_emotion_dimensions(self):
        """Test creating valid EmotionDimensions."""
        from voice_man.models.forensic.emotion_recognition import EmotionDimensions

        dimensions = EmotionDimensions(
            arousal=0.7,
            dominance=0.5,
            valence=0.3,
        )

        assert dimensions.arousal == 0.7
        assert dimensions.dominance == 0.5
        assert dimensions.valence == 0.3

    def test_emotion_dimensions_boundary_values(self):
        """Test EmotionDimensions with boundary values."""
        from voice_man.models.forensic.emotion_recognition import EmotionDimensions

        # Minimum values
        dim_min = EmotionDimensions(arousal=0.0, dominance=0.0, valence=0.0)
        assert dim_min.arousal == 0.0
        assert dim_min.dominance == 0.0
        assert dim_min.valence == 0.0

        # Maximum values
        dim_max = EmotionDimensions(arousal=1.0, dominance=1.0, valence=1.0)
        assert dim_max.arousal == 1.0
        assert dim_max.dominance == 1.0
        assert dim_max.valence == 1.0

    def test_emotion_dimensions_invalid_arousal(self):
        """Test EmotionDimensions rejects invalid arousal values."""
        from voice_man.models.forensic.emotion_recognition import EmotionDimensions

        with pytest.raises(ValidationError):
            EmotionDimensions(arousal=1.5, dominance=0.5, valence=0.5)

        with pytest.raises(ValidationError):
            EmotionDimensions(arousal=-0.1, dominance=0.5, valence=0.5)

    def test_emotion_dimensions_invalid_dominance(self):
        """Test EmotionDimensions rejects invalid dominance values."""
        from voice_man.models.forensic.emotion_recognition import EmotionDimensions

        with pytest.raises(ValidationError):
            EmotionDimensions(arousal=0.5, dominance=1.1, valence=0.5)

    def test_emotion_dimensions_invalid_valence(self):
        """Test EmotionDimensions rejects invalid valence values."""
        from voice_man.models.forensic.emotion_recognition import EmotionDimensions

        with pytest.raises(ValidationError):
            EmotionDimensions(arousal=0.5, dominance=0.5, valence=-0.5)


class TestCategoricalEmotion:
    """Tests for CategoricalEmotion model."""

    def test_valid_categorical_emotion(self):
        """Test creating valid CategoricalEmotion."""
        from voice_man.models.forensic.emotion_recognition import CategoricalEmotion

        emotion = CategoricalEmotion(emotion_type="angry", confidence=0.85)

        assert emotion.emotion_type == "angry"
        assert emotion.confidence == 0.85

    def test_categorical_emotion_all_types(self):
        """Test all valid emotion types."""
        from voice_man.models.forensic.emotion_recognition import CategoricalEmotion

        valid_types = ["angry", "happy", "sad", "neutral", "fear", "disgust", "surprise"]

        for emotion_type in valid_types:
            emotion = CategoricalEmotion(emotion_type=emotion_type, confidence=0.5)
            assert emotion.emotion_type == emotion_type

    def test_categorical_emotion_invalid_type(self):
        """Test CategoricalEmotion rejects invalid emotion types."""
        from voice_man.models.forensic.emotion_recognition import CategoricalEmotion

        with pytest.raises(ValidationError):
            CategoricalEmotion(emotion_type="excited", confidence=0.5)

    def test_categorical_emotion_confidence_boundary(self):
        """Test confidence boundary values."""
        from voice_man.models.forensic.emotion_recognition import CategoricalEmotion

        # Valid boundaries
        assert CategoricalEmotion(emotion_type="happy", confidence=0.0).confidence == 0.0
        assert CategoricalEmotion(emotion_type="happy", confidence=1.0).confidence == 1.0

        # Invalid
        with pytest.raises(ValidationError):
            CategoricalEmotion(emotion_type="happy", confidence=1.1)


class TestEmotionProbabilities:
    """Tests for EmotionProbabilities model."""

    def test_valid_emotion_probabilities(self):
        """Test creating valid EmotionProbabilities."""
        from voice_man.models.forensic.emotion_recognition import EmotionProbabilities

        probs = EmotionProbabilities(
            angry=0.1,
            happy=0.2,
            sad=0.1,
            neutral=0.6,
        )

        assert probs.angry == 0.1
        assert probs.happy == 0.2
        assert probs.sad == 0.1
        assert probs.neutral == 0.6
        assert probs.fear is None
        assert probs.disgust is None
        assert probs.surprise is None

    def test_emotion_probabilities_with_all_emotions(self):
        """Test EmotionProbabilities with all emotion categories."""
        from voice_man.models.forensic.emotion_recognition import EmotionProbabilities

        probs = EmotionProbabilities(
            angry=0.1,
            happy=0.1,
            sad=0.1,
            neutral=0.3,
            fear=0.15,
            disgust=0.1,
            surprise=0.15,
        )

        assert probs.fear == 0.15
        assert probs.disgust == 0.1
        assert probs.surprise == 0.15


class TestEmotionAnalysisResult:
    """Tests for EmotionAnalysisResult model."""

    def test_valid_result_with_dimensions(self):
        """Test creating result with dimensional analysis."""
        from voice_man.models.forensic.emotion_recognition import (
            EmotionAnalysisResult,
            EmotionDimensions,
        )

        result = EmotionAnalysisResult(
            dimensions=EmotionDimensions(arousal=0.7, dominance=0.5, valence=0.3),
            model_used="audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",
            processing_time_ms=150.5,
        )

        assert result.dimensions is not None
        assert result.categorical is None
        assert result.model_used == "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
        assert result.processing_time_ms == 150.5

    def test_valid_result_with_categorical(self):
        """Test creating result with categorical analysis."""
        from voice_man.models.forensic.emotion_recognition import (
            EmotionAnalysisResult,
            CategoricalEmotion,
        )

        result = EmotionAnalysisResult(
            categorical=CategoricalEmotion(emotion_type="angry", confidence=0.85),
            model_used="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
            processing_time_ms=100.0,
        )

        assert result.categorical is not None
        assert result.dimensions is None

    def test_result_requires_at_least_one_analysis(self):
        """Test that result requires at least dimensions or categorical."""
        from voice_man.models.forensic.emotion_recognition import EmotionAnalysisResult

        with pytest.raises(ValidationError):
            EmotionAnalysisResult(
                model_used="test-model",
                processing_time_ms=100.0,
            )


class TestForensicEmotionIndicators:
    """Tests for ForensicEmotionIndicators model."""

    def test_valid_forensic_indicators(self):
        """Test creating valid ForensicEmotionIndicators."""
        from voice_man.models.forensic.emotion_recognition import ForensicEmotionIndicators

        indicators = ForensicEmotionIndicators(
            high_arousal_low_valence=True,
            emotion_inconsistency_score=0.3,
            dominant_emotion="angry",
            arousal_level="high",
            stress_indicator=True,
            deception_indicator=False,
            confidence=0.85,
        )

        assert indicators.high_arousal_low_valence is True
        assert indicators.emotion_inconsistency_score == 0.3
        assert indicators.dominant_emotion == "angry"
        assert indicators.arousal_level == "high"
        assert indicators.stress_indicator is True
        assert indicators.deception_indicator is False
        assert indicators.confidence == 0.85

    def test_forensic_indicators_arousal_levels(self):
        """Test all valid arousal levels."""
        from voice_man.models.forensic.emotion_recognition import ForensicEmotionIndicators

        for level in ["low", "medium", "high"]:
            indicators = ForensicEmotionIndicators(
                high_arousal_low_valence=False,
                emotion_inconsistency_score=0.1,
                dominant_emotion="neutral",
                arousal_level=level,
                stress_indicator=False,
                deception_indicator=False,
                confidence=0.9,
            )
            assert indicators.arousal_level == level


class TestMultiModelEmotionResult:
    """Tests for MultiModelEmotionResult model."""

    def test_valid_multi_model_result(self):
        """Test creating valid MultiModelEmotionResult."""
        from voice_man.models.forensic.emotion_recognition import (
            MultiModelEmotionResult,
            EmotionAnalysisResult,
            EmotionDimensions,
            CategoricalEmotion,
        )

        result = MultiModelEmotionResult(
            primary_result=EmotionAnalysisResult(
                dimensions=EmotionDimensions(arousal=0.7, dominance=0.5, valence=0.3),
                model_used="primary-model",
                processing_time_ms=100.0,
            ),
            secondary_result=EmotionAnalysisResult(
                categorical=CategoricalEmotion(emotion_type="angry", confidence=0.8),
                model_used="secondary-model",
                processing_time_ms=80.0,
            ),
            ensemble_confidence=0.85,
            audio_duration_seconds=5.0,
        )

        assert result.primary_result is not None
        assert result.secondary_result is not None
        assert result.ensemble_confidence == 0.85
        assert result.audio_duration_seconds == 5.0

    def test_multi_model_result_with_only_primary(self):
        """Test MultiModelEmotionResult with only primary result."""
        from voice_man.models.forensic.emotion_recognition import (
            MultiModelEmotionResult,
            EmotionAnalysisResult,
            EmotionDimensions,
        )

        result = MultiModelEmotionResult(
            primary_result=EmotionAnalysisResult(
                dimensions=EmotionDimensions(arousal=0.5, dominance=0.5, valence=0.5),
                model_used="primary-model",
                processing_time_ms=100.0,
            ),
            ensemble_confidence=0.7,
            audio_duration_seconds=3.0,
        )

        assert result.primary_result is not None
        assert result.secondary_result is None

    def test_multi_model_result_requires_at_least_one(self):
        """Test that at least one result is required."""
        from voice_man.models.forensic.emotion_recognition import MultiModelEmotionResult

        with pytest.raises(ValidationError):
            MultiModelEmotionResult(
                ensemble_confidence=0.5,
                audio_duration_seconds=3.0,
            )

    def test_multi_model_result_with_forensic_indicators(self):
        """Test MultiModelEmotionResult with forensic indicators."""
        from voice_man.models.forensic.emotion_recognition import (
            MultiModelEmotionResult,
            EmotionAnalysisResult,
            EmotionDimensions,
            ForensicEmotionIndicators,
        )

        result = MultiModelEmotionResult(
            primary_result=EmotionAnalysisResult(
                dimensions=EmotionDimensions(arousal=0.8, dominance=0.6, valence=0.2),
                model_used="primary-model",
                processing_time_ms=100.0,
            ),
            ensemble_confidence=0.75,
            audio_duration_seconds=4.0,
            forensic_indicators=ForensicEmotionIndicators(
                high_arousal_low_valence=True,
                emotion_inconsistency_score=0.2,
                dominant_emotion="angry",
                arousal_level="high",
                stress_indicator=True,
                deception_indicator=False,
                confidence=0.8,
            ),
        )

        assert result.forensic_indicators is not None
        assert result.forensic_indicators.high_arousal_low_valence is True
        assert result.forensic_indicators.stress_indicator is True

    def test_multi_model_result_invalid_duration(self):
        """Test that audio_duration_seconds must be positive."""
        from voice_man.models.forensic.emotion_recognition import (
            MultiModelEmotionResult,
            EmotionAnalysisResult,
            CategoricalEmotion,
        )

        with pytest.raises(ValidationError):
            MultiModelEmotionResult(
                primary_result=EmotionAnalysisResult(
                    categorical=CategoricalEmotion(emotion_type="neutral", confidence=0.9),
                    model_used="model",
                    processing_time_ms=50.0,
                ),
                ensemble_confidence=0.9,
                audio_duration_seconds=0.0,  # Invalid: must be > 0
            )
