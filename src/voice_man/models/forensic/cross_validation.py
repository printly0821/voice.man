"""
Cross-Validation Models for Text-Voice Analysis
SPEC-FORENSIC-001 Phase 2-C: Text-Voice cross-validation models

Detects discrepancies between text content (what was said) and voice emotion
(how it was said) to identify potential deception.
"""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class DiscrepancyType(str, Enum):
    """Types of discrepancies between text and voice analysis."""

    EMOTION_TEXT_MISMATCH = "emotion_text_mismatch"  # Emotion-text mismatch
    INTENSITY_MISMATCH = "intensity_mismatch"  # Intensity mismatch
    SENTIMENT_CONTRADICTION = "sentiment_contradiction"  # Sentiment contradiction
    STRESS_CONTENT_MISMATCH = "stress_content_mismatch"  # Stress-content mismatch


class DiscrepancySeverity(str, Enum):
    """Severity levels for detected discrepancies."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Discrepancy(BaseModel):
    """Individual discrepancy item detected during cross-validation.

    Attributes:
        discrepancy_type: Type of the detected discrepancy.
        severity: Severity level of the discrepancy.
        description: Human-readable description of the discrepancy.
        text_evidence: Evidence from text analysis (optional).
        voice_evidence: Evidence from voice analysis (optional).
        confidence: Confidence score for the detection (0.0-1.0).
    """

    model_config = ConfigDict(use_enum_values=True, str_strip_whitespace=True)

    discrepancy_type: DiscrepancyType = Field(..., description="Type of the detected discrepancy")
    severity: DiscrepancySeverity = Field(..., description="Severity level of the discrepancy")
    description: str = Field(
        ..., min_length=1, description="Human-readable description of the discrepancy"
    )
    text_evidence: Optional[str] = Field(default=None, description="Evidence from text analysis")
    voice_evidence: Optional[str] = Field(default=None, description="Evidence from voice analysis")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score for the detection (0.0-1.0)"
    )


class TextAnalysisResult(BaseModel):
    """Result of text analysis for cross-validation.

    Attributes:
        text: The original text analyzed.
        detected_sentiment: Overall sentiment (positive/negative/neutral).
        sentiment_score: Sentiment score (-1.0 to 1.0, negative to positive).
        detected_emotions: List of emotions detected in text.
        crime_patterns_found: List of crime patterns detected (gaslighting, threat, etc.).
        intensity_level: Overall intensity level of the text (0.0-1.0).
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    text: str = Field(..., min_length=1, description="The original text analyzed")
    detected_sentiment: str = Field(
        ..., description="Overall sentiment (positive/negative/neutral)"
    )
    sentiment_score: float = Field(
        ..., ge=-1.0, le=1.0, description="Sentiment score (-1.0 to 1.0)"
    )
    detected_emotions: List[str] = Field(
        default_factory=list, description="List of emotions detected in text"
    )
    crime_patterns_found: List[str] = Field(
        default_factory=list, description="List of crime patterns detected"
    )
    intensity_level: float = Field(
        ..., ge=0.0, le=1.0, description="Overall intensity level of the text (0.0-1.0)"
    )


class VoiceAnalysisResult(BaseModel):
    """Result of voice analysis for cross-validation.

    Attributes:
        dominant_emotion: The primary emotion detected in voice.
        emotion_confidence: Confidence score for the emotion detection (0.0-1.0).
        arousal: Arousal level from dimensional emotion analysis (0.0-1.0).
        valence: Valence level from dimensional emotion analysis (0.0-1.0).
        stress_level: Detected stress level (0.0-1.0).
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    dominant_emotion: str = Field(..., description="The primary emotion detected in voice")
    emotion_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score for the emotion detection"
    )
    arousal: float = Field(
        ..., ge=0.0, le=1.0, description="Arousal level (0.0-1.0, low=calm, high=excited)"
    )
    valence: float = Field(
        ..., ge=0.0, le=1.0, description="Valence level (0.0-1.0, low=negative, high=positive)"
    )
    stress_level: float = Field(..., ge=0.0, le=1.0, description="Detected stress level (0.0-1.0)")


class CrossValidationResult(BaseModel):
    """Complete result of text-voice cross-validation analysis.

    Attributes:
        text_analysis: Results from text analysis.
        voice_analysis: Results from voice analysis.
        discrepancies: List of detected discrepancies.
        overall_consistency_score: How consistent text and voice are (0.0-1.0, higher=more consistent).
        deception_probability: Estimated probability of deception (0.0-1.0).
        risk_level: Overall risk level based on discrepancies.
        analysis_notes: Additional notes from the analysis.
    """

    model_config = ConfigDict(use_enum_values=True, str_strip_whitespace=True)

    text_analysis: TextAnalysisResult = Field(..., description="Results from text analysis")
    voice_analysis: VoiceAnalysisResult = Field(..., description="Results from voice analysis")
    discrepancies: List[Discrepancy] = Field(
        default_factory=list, description="List of detected discrepancies"
    )
    overall_consistency_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="How consistent text and voice are (0.0-1.0)",
    )
    deception_probability: float = Field(
        ..., ge=0.0, le=1.0, description="Estimated probability of deception (0.0-1.0)"
    )
    risk_level: DiscrepancySeverity = Field(
        ..., description="Overall risk level based on discrepancies"
    )
    analysis_notes: List[str] = Field(
        default_factory=list, description="Additional notes from the analysis"
    )
