"""
Emotion Recognition Models
SPEC-FORENSIC-001 Phase 2-B: Pydantic models for Speech Emotion Recognition (SER)

These models define the data structures for emotion analysis
using HuggingFace wav2vec2 and SpeechBrain models.
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class EmotionDimensions(BaseModel):
    """
    Dimensional emotion analysis results (VAD model).

    Based on audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim model.

    Attributes:
        arousal: Level of activation/energy (0.0-1.0, low=calm, high=excited)
        dominance: Level of control/power (0.0-1.0, low=submissive, high=dominant)
        valence: Level of pleasantness (0.0-1.0, low=negative, high=positive)
    """

    arousal: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Arousal level (0=calm, 1=excited)",
    )
    dominance: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Dominance level (0=submissive, 1=dominant)",
    )
    valence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Valence level (0=negative, 1=positive)",
    )


class CategoricalEmotion(BaseModel):
    """
    Categorical emotion classification result.

    Based on speechbrain/emotion-recognition-wav2vec2-IEMOCAP model.

    Attributes:
        emotion_type: Classified emotion category
        confidence: Classification confidence score (0.0-1.0)
    """

    emotion_type: Literal["angry", "happy", "sad", "neutral", "fear", "disgust", "surprise"] = (
        Field(..., description="Classified emotion category")
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Classification confidence score",
    )

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Validate confidence is within valid range."""
        if v < 0.0 or v > 1.0:
            raise ValueError(f"confidence must be between 0.0 and 1.0, got {v}")
        return v


class EmotionProbabilities(BaseModel):
    """
    Full probability distribution over emotion categories.

    Attributes:
        angry: Probability of angry emotion
        happy: Probability of happy emotion
        sad: Probability of sad emotion
        neutral: Probability of neutral emotion
        fear: Probability of fear emotion (optional, depends on model)
        disgust: Probability of disgust emotion (optional, depends on model)
        surprise: Probability of surprise emotion (optional, depends on model)
    """

    angry: float = Field(..., ge=0.0, le=1.0, description="Probability of angry emotion")
    happy: float = Field(..., ge=0.0, le=1.0, description="Probability of happy emotion")
    sad: float = Field(..., ge=0.0, le=1.0, description="Probability of sad emotion")
    neutral: float = Field(..., ge=0.0, le=1.0, description="Probability of neutral emotion")
    fear: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Probability of fear emotion"
    )
    disgust: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Probability of disgust emotion"
    )
    surprise: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Probability of surprise emotion"
    )


class EmotionAnalysisResult(BaseModel):
    """
    Single model emotion analysis result.

    Attributes:
        dimensions: Dimensional emotion analysis (VAD)
        categorical: Primary categorical emotion classification
        probabilities: Full probability distribution (optional)
        model_used: Name of the model used for analysis
        processing_time_ms: Time taken for inference in milliseconds
    """

    dimensions: Optional[EmotionDimensions] = Field(
        default=None, description="Dimensional emotion analysis (VAD)"
    )
    categorical: Optional[CategoricalEmotion] = Field(
        default=None, description="Primary categorical emotion classification"
    )
    probabilities: Optional[EmotionProbabilities] = Field(
        default=None, description="Full probability distribution over emotions"
    )
    model_used: str = Field(..., description="Name of the model used for analysis")
    processing_time_ms: float = Field(..., ge=0.0, description="Processing time in milliseconds")

    @model_validator(mode="after")
    def validate_has_result(self) -> "EmotionAnalysisResult":
        """Validate that at least one analysis type is present."""
        if self.dimensions is None and self.categorical is None:
            raise ValueError("At least one of dimensions or categorical must be provided")
        return self


class ForensicEmotionIndicators(BaseModel):
    """
    Forensic-relevant emotion indicators extracted from analysis.

    These indicators are useful for voice forensics applications
    such as deception detection and emotional state assessment.

    Attributes:
        high_arousal_low_valence: Indicates possible anger/fear (stress indicator)
        emotion_inconsistency_score: Mismatch between dimensional and categorical (deception indicator)
        dominant_emotion: The most likely emotion category
        arousal_level: Categorized arousal level
        stress_indicator: Whether high stress is indicated
        deception_indicator: Whether emotion patterns suggest potential deception
    """

    high_arousal_low_valence: bool = Field(
        ..., description="High arousal + low valence = possible anger/fear"
    )
    emotion_inconsistency_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Score indicating mismatch between analysis methods (higher = more inconsistent)",
    )
    dominant_emotion: Literal["angry", "happy", "sad", "neutral", "fear", "disgust", "surprise"] = (
        Field(..., description="Most likely emotion category")
    )
    arousal_level: Literal["low", "medium", "high"] = Field(
        ..., description="Categorized arousal level"
    )
    stress_indicator: bool = Field(..., description="Whether high stress is indicated")
    deception_indicator: bool = Field(
        ..., description="Whether patterns suggest potential deception"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall confidence in forensic indicators",
    )


class MultiModelEmotionResult(BaseModel):
    """
    Combined results from multiple emotion recognition models.

    Attributes:
        primary_result: Result from primary model (audeering wav2vec2)
        secondary_result: Result from secondary model (speechbrain)
        ensemble_emotion: Combined emotion classification
        ensemble_confidence: Weighted confidence from ensemble
        confidence_weighted: Confidence-weighted combination flag
        forensic_indicators: Forensic-relevant emotion indicators
        audio_duration_seconds: Duration of analyzed audio
    """

    primary_result: Optional[EmotionAnalysisResult] = Field(
        default=None, description="Result from primary model (dimensional)"
    )
    secondary_result: Optional[EmotionAnalysisResult] = Field(
        default=None, description="Result from secondary model (categorical)"
    )
    ensemble_emotion: Optional[CategoricalEmotion] = Field(
        default=None, description="Combined emotion classification from ensemble"
    )
    ensemble_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Weighted confidence score from ensemble",
    )
    confidence_weighted: bool = Field(
        default=True, description="Whether ensemble used confidence weighting"
    )
    forensic_indicators: Optional[ForensicEmotionIndicators] = Field(
        default=None, description="Forensic-relevant emotion indicators"
    )
    audio_duration_seconds: float = Field(
        ..., gt=0.0, description="Duration of analyzed audio in seconds"
    )

    @model_validator(mode="after")
    def validate_has_at_least_one_result(self) -> "MultiModelEmotionResult":
        """Validate that at least one model result is present."""
        if self.primary_result is None and self.secondary_result is None:
            raise ValueError("At least one of primary_result or secondary_result must be provided")
        return self
