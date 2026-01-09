"""
Tests for Cross-Validation Pydantic Models
SPEC-FORENSIC-001 Phase 2-C: Cross-validation model tests

TDD RED Phase: These tests define the expected behavior of cross-validation models.
"""

import pytest
from pydantic import ValidationError


class TestDiscrepancyType:
    """Tests for DiscrepancyType enum."""

    def test_all_discrepancy_types_defined(self):
        """Test all required discrepancy types are defined."""
        from voice_man.models.forensic.cross_validation import DiscrepancyType

        assert DiscrepancyType.EMOTION_TEXT_MISMATCH == "emotion_text_mismatch"
        assert DiscrepancyType.INTENSITY_MISMATCH == "intensity_mismatch"
        assert DiscrepancyType.SENTIMENT_CONTRADICTION == "sentiment_contradiction"
        assert DiscrepancyType.STRESS_CONTENT_MISMATCH == "stress_content_mismatch"

    def test_discrepancy_type_values(self):
        """Test DiscrepancyType enum values."""
        from voice_man.models.forensic.cross_validation import DiscrepancyType

        assert len(DiscrepancyType) == 4


class TestDiscrepancySeverity:
    """Tests for DiscrepancySeverity enum."""

    def test_all_severity_levels_defined(self):
        """Test all required severity levels are defined."""
        from voice_man.models.forensic.cross_validation import DiscrepancySeverity

        assert DiscrepancySeverity.LOW == "low"
        assert DiscrepancySeverity.MEDIUM == "medium"
        assert DiscrepancySeverity.HIGH == "high"
        assert DiscrepancySeverity.CRITICAL == "critical"

    def test_severity_level_count(self):
        """Test severity level count."""
        from voice_man.models.forensic.cross_validation import DiscrepancySeverity

        assert len(DiscrepancySeverity) == 4


class TestDiscrepancy:
    """Tests for Discrepancy model."""

    def test_discrepancy_valid_creation(self):
        """Test creating a valid Discrepancy instance."""
        from voice_man.models.forensic.cross_validation import (
            Discrepancy,
            DiscrepancyType,
            DiscrepancySeverity,
        )

        discrepancy = Discrepancy(
            discrepancy_type=DiscrepancyType.EMOTION_TEXT_MISMATCH,
            severity=DiscrepancySeverity.HIGH,
            description="Positive text with negative voice emotion",
            text_evidence="I'm so happy to see you",
            voice_evidence="angry tone detected",
            confidence=0.85,
        )

        assert discrepancy.discrepancy_type == "emotion_text_mismatch"
        assert discrepancy.severity == "high"
        assert discrepancy.confidence == 0.85

    def test_discrepancy_optional_fields(self):
        """Test Discrepancy with optional fields omitted."""
        from voice_man.models.forensic.cross_validation import (
            Discrepancy,
            DiscrepancyType,
            DiscrepancySeverity,
        )

        discrepancy = Discrepancy(
            discrepancy_type=DiscrepancyType.INTENSITY_MISMATCH,
            severity=DiscrepancySeverity.MEDIUM,
            description="Strong expression with low arousal",
            confidence=0.7,
        )

        assert discrepancy.text_evidence is None
        assert discrepancy.voice_evidence is None

    def test_discrepancy_confidence_validation(self):
        """Test confidence field validation (0.0-1.0)."""
        from voice_man.models.forensic.cross_validation import (
            Discrepancy,
            DiscrepancyType,
            DiscrepancySeverity,
        )

        # Valid range
        d1 = Discrepancy(
            discrepancy_type=DiscrepancyType.EMOTION_TEXT_MISMATCH,
            severity=DiscrepancySeverity.LOW,
            description="Test",
            confidence=0.0,
        )
        assert d1.confidence == 0.0

        d2 = Discrepancy(
            discrepancy_type=DiscrepancyType.EMOTION_TEXT_MISMATCH,
            severity=DiscrepancySeverity.LOW,
            description="Test",
            confidence=1.0,
        )
        assert d2.confidence == 1.0

        # Invalid range
        with pytest.raises(ValidationError):
            Discrepancy(
                discrepancy_type=DiscrepancyType.EMOTION_TEXT_MISMATCH,
                severity=DiscrepancySeverity.LOW,
                description="Test",
                confidence=1.5,
            )

        with pytest.raises(ValidationError):
            Discrepancy(
                discrepancy_type=DiscrepancyType.EMOTION_TEXT_MISMATCH,
                severity=DiscrepancySeverity.LOW,
                description="Test",
                confidence=-0.1,
            )

    def test_discrepancy_description_required(self):
        """Test description is required and non-empty."""
        from voice_man.models.forensic.cross_validation import (
            Discrepancy,
            DiscrepancyType,
            DiscrepancySeverity,
        )

        with pytest.raises(ValidationError):
            Discrepancy(
                discrepancy_type=DiscrepancyType.EMOTION_TEXT_MISMATCH,
                severity=DiscrepancySeverity.LOW,
                description="",  # Empty string
                confidence=0.5,
            )


class TestTextAnalysisResult:
    """Tests for TextAnalysisResult model."""

    def test_text_analysis_valid_creation(self):
        """Test creating a valid TextAnalysisResult instance."""
        from voice_man.models.forensic.cross_validation import TextAnalysisResult

        result = TextAnalysisResult(
            text="I'm so happy to see you!",
            detected_sentiment="positive",
            sentiment_score=0.8,
            detected_emotions=["joy", "excitement"],
            crime_patterns_found=[],
            intensity_level=0.6,
        )

        assert result.text == "I'm so happy to see you!"
        assert result.detected_sentiment == "positive"
        assert result.sentiment_score == 0.8
        assert "joy" in result.detected_emotions
        assert result.intensity_level == 0.6

    def test_text_analysis_sentiment_score_range(self):
        """Test sentiment_score is within valid range (-1.0 to 1.0)."""
        from voice_man.models.forensic.cross_validation import TextAnalysisResult

        # Valid positive
        r1 = TextAnalysisResult(
            text="Test",
            detected_sentiment="positive",
            sentiment_score=1.0,
            intensity_level=0.5,
        )
        assert r1.sentiment_score == 1.0

        # Valid negative
        r2 = TextAnalysisResult(
            text="Test",
            detected_sentiment="negative",
            sentiment_score=-1.0,
            intensity_level=0.5,
        )
        assert r2.sentiment_score == -1.0

        # Invalid
        with pytest.raises(ValidationError):
            TextAnalysisResult(
                text="Test",
                detected_sentiment="positive",
                sentiment_score=1.5,
                intensity_level=0.5,
            )

        with pytest.raises(ValidationError):
            TextAnalysisResult(
                text="Test",
                detected_sentiment="negative",
                sentiment_score=-1.5,
                intensity_level=0.5,
            )

    def test_text_analysis_intensity_level_range(self):
        """Test intensity_level is within valid range (0.0-1.0)."""
        from voice_man.models.forensic.cross_validation import TextAnalysisResult

        # Valid range
        r1 = TextAnalysisResult(
            text="Test",
            detected_sentiment="neutral",
            sentiment_score=0.0,
            intensity_level=0.0,
        )
        assert r1.intensity_level == 0.0

        r2 = TextAnalysisResult(
            text="Test",
            detected_sentiment="neutral",
            sentiment_score=0.0,
            intensity_level=1.0,
        )
        assert r2.intensity_level == 1.0

        # Invalid
        with pytest.raises(ValidationError):
            TextAnalysisResult(
                text="Test",
                detected_sentiment="neutral",
                sentiment_score=0.0,
                intensity_level=1.5,
            )

    def test_text_analysis_with_crime_patterns(self):
        """Test TextAnalysisResult with crime patterns."""
        from voice_man.models.forensic.cross_validation import TextAnalysisResult

        result = TextAnalysisResult(
            text="You're crazy, that never happened",
            detected_sentiment="negative",
            sentiment_score=-0.3,
            detected_emotions=["anger"],
            crime_patterns_found=["gaslighting", "denial"],
            intensity_level=0.7,
        )

        assert "gaslighting" in result.crime_patterns_found
        assert "denial" in result.crime_patterns_found


class TestVoiceAnalysisResult:
    """Tests for VoiceAnalysisResult model."""

    def test_voice_analysis_valid_creation(self):
        """Test creating a valid VoiceAnalysisResult instance."""
        from voice_man.models.forensic.cross_validation import VoiceAnalysisResult

        result = VoiceAnalysisResult(
            dominant_emotion="angry",
            emotion_confidence=0.85,
            arousal=0.8,
            valence=0.2,
            stress_level=0.75,
        )

        assert result.dominant_emotion == "angry"
        assert result.emotion_confidence == 0.85
        assert result.arousal == 0.8
        assert result.valence == 0.2
        assert result.stress_level == 0.75

    def test_voice_analysis_field_ranges(self):
        """Test all fields are within valid ranges (0.0-1.0)."""
        from voice_man.models.forensic.cross_validation import VoiceAnalysisResult

        # Valid boundary values
        result = VoiceAnalysisResult(
            dominant_emotion="neutral",
            emotion_confidence=0.0,
            arousal=0.0,
            valence=0.0,
            stress_level=0.0,
        )
        assert result.emotion_confidence == 0.0

        result = VoiceAnalysisResult(
            dominant_emotion="neutral",
            emotion_confidence=1.0,
            arousal=1.0,
            valence=1.0,
            stress_level=1.0,
        )
        assert result.emotion_confidence == 1.0

        # Invalid values
        with pytest.raises(ValidationError):
            VoiceAnalysisResult(
                dominant_emotion="neutral",
                emotion_confidence=1.5,
                arousal=0.5,
                valence=0.5,
                stress_level=0.5,
            )

        with pytest.raises(ValidationError):
            VoiceAnalysisResult(
                dominant_emotion="neutral",
                emotion_confidence=0.5,
                arousal=-0.1,
                valence=0.5,
                stress_level=0.5,
            )


class TestCrossValidationResult:
    """Tests for CrossValidationResult model."""

    def test_cross_validation_result_valid_creation(self):
        """Test creating a valid CrossValidationResult instance."""
        from voice_man.models.forensic.cross_validation import (
            CrossValidationResult,
            TextAnalysisResult,
            VoiceAnalysisResult,
            Discrepancy,
            DiscrepancyType,
            DiscrepancySeverity,
        )

        text_result = TextAnalysisResult(
            text="I'm happy",
            detected_sentiment="positive",
            sentiment_score=0.7,
            detected_emotions=["joy"],
            crime_patterns_found=[],
            intensity_level=0.5,
        )

        voice_result = VoiceAnalysisResult(
            dominant_emotion="angry",
            emotion_confidence=0.8,
            arousal=0.7,
            valence=0.2,
            stress_level=0.6,
        )

        discrepancy = Discrepancy(
            discrepancy_type=DiscrepancyType.EMOTION_TEXT_MISMATCH,
            severity=DiscrepancySeverity.HIGH,
            description="Positive text with angry voice",
            confidence=0.85,
        )

        result = CrossValidationResult(
            text_analysis=text_result,
            voice_analysis=voice_result,
            discrepancies=[discrepancy],
            overall_consistency_score=0.3,
            deception_probability=0.7,
            risk_level=DiscrepancySeverity.HIGH,
            analysis_notes=["Significant mismatch detected"],
        )

        assert result.text_analysis.text == "I'm happy"
        assert result.voice_analysis.dominant_emotion == "angry"
        assert len(result.discrepancies) == 1
        assert result.overall_consistency_score == 0.3
        assert result.deception_probability == 0.7
        assert result.risk_level == "high"

    def test_cross_validation_result_empty_discrepancies(self):
        """Test CrossValidationResult with no discrepancies (consistent)."""
        from voice_man.models.forensic.cross_validation import (
            CrossValidationResult,
            TextAnalysisResult,
            VoiceAnalysisResult,
            DiscrepancySeverity,
        )

        text_result = TextAnalysisResult(
            text="I'm happy",
            detected_sentiment="positive",
            sentiment_score=0.8,
            detected_emotions=["joy"],
            crime_patterns_found=[],
            intensity_level=0.6,
        )

        voice_result = VoiceAnalysisResult(
            dominant_emotion="happy",
            emotion_confidence=0.85,
            arousal=0.6,
            valence=0.8,
            stress_level=0.2,
        )

        result = CrossValidationResult(
            text_analysis=text_result,
            voice_analysis=voice_result,
            discrepancies=[],
            overall_consistency_score=0.95,
            deception_probability=0.05,
            risk_level=DiscrepancySeverity.LOW,
            analysis_notes=["Consistent text and voice"],
        )

        assert len(result.discrepancies) == 0
        assert result.overall_consistency_score == 0.95

    def test_cross_validation_result_field_ranges(self):
        """Test score fields are within valid ranges."""
        from voice_man.models.forensic.cross_validation import (
            CrossValidationResult,
            TextAnalysisResult,
            VoiceAnalysisResult,
            DiscrepancySeverity,
        )

        text_result = TextAnalysisResult(
            text="Test",
            detected_sentiment="neutral",
            sentiment_score=0.0,
            intensity_level=0.5,
        )

        voice_result = VoiceAnalysisResult(
            dominant_emotion="neutral",
            emotion_confidence=0.5,
            arousal=0.5,
            valence=0.5,
            stress_level=0.5,
        )

        # Invalid consistency score
        with pytest.raises(ValidationError):
            CrossValidationResult(
                text_analysis=text_result,
                voice_analysis=voice_result,
                overall_consistency_score=1.5,
                deception_probability=0.5,
                risk_level=DiscrepancySeverity.LOW,
            )

        # Invalid deception probability
        with pytest.raises(ValidationError):
            CrossValidationResult(
                text_analysis=text_result,
                voice_analysis=voice_result,
                overall_consistency_score=0.5,
                deception_probability=-0.1,
                risk_level=DiscrepancySeverity.LOW,
            )

    def test_cross_validation_result_multiple_discrepancies(self):
        """Test CrossValidationResult with multiple discrepancies."""
        from voice_man.models.forensic.cross_validation import (
            CrossValidationResult,
            TextAnalysisResult,
            VoiceAnalysisResult,
            Discrepancy,
            DiscrepancyType,
            DiscrepancySeverity,
        )

        text_result = TextAnalysisResult(
            text="I will destroy you",
            detected_sentiment="negative",
            sentiment_score=-0.8,
            detected_emotions=["anger"],
            crime_patterns_found=["threat"],
            intensity_level=0.9,
        )

        voice_result = VoiceAnalysisResult(
            dominant_emotion="neutral",
            emotion_confidence=0.7,
            arousal=0.2,
            valence=0.5,
            stress_level=0.1,
        )

        discrepancies = [
            Discrepancy(
                discrepancy_type=DiscrepancyType.INTENSITY_MISMATCH,
                severity=DiscrepancySeverity.CRITICAL,
                description="Threatening text with calm voice",
                confidence=0.9,
            ),
            Discrepancy(
                discrepancy_type=DiscrepancyType.SENTIMENT_CONTRADICTION,
                severity=DiscrepancySeverity.HIGH,
                description="Negative sentiment with neutral emotion",
                confidence=0.85,
            ),
        ]

        result = CrossValidationResult(
            text_analysis=text_result,
            voice_analysis=voice_result,
            discrepancies=discrepancies,
            overall_consistency_score=0.1,
            deception_probability=0.9,
            risk_level=DiscrepancySeverity.CRITICAL,
            analysis_notes=["Multiple critical discrepancies", "High deception risk"],
        )

        assert len(result.discrepancies) == 2
        assert result.risk_level == "critical"


class TestModelImportFromPackage:
    """Tests for model import from package."""

    def test_import_from_forensic_package(self):
        """Test importing models from voice_man.models.forensic package."""
        from voice_man.models.forensic import (
            DiscrepancyType,
            DiscrepancySeverity,
            Discrepancy,
            TextAnalysisResult,
            VoiceAnalysisResult,
            CrossValidationResult,
        )

        # Verify all imports work correctly
        assert DiscrepancyType.EMOTION_TEXT_MISMATCH is not None
        assert DiscrepancySeverity.HIGH is not None
        assert Discrepancy is not None
        assert TextAnalysisResult is not None
        assert VoiceAnalysisResult is not None
        assert CrossValidationResult is not None
