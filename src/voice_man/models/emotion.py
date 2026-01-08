"""
Emotion Analysis Models
TASK-010: Emotion analysis using KoBERT-based model with 7 emotion classes
"""

from enum import Enum
from typing import Dict
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, model_validator


class EmotionLabel(str, Enum):
    """7 emotion classes for emotion classification"""

    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    DISGUST = "disgust"
    SURPRISE = "surprise"
    NEUTRAL = "neutral"


class EmotionAnalysis(BaseModel):
    """Single emotion analysis result"""

    primary_emotion: EmotionLabel = Field(..., description="Primary detected emotion")
    intensity: float = Field(..., ge=0.0, le=1.0, description="Emotion intensity score (0.0-1.0)")
    emotion_distribution: Dict[str, float] = Field(
        ..., description="Distribution of all 7 emotions, should sum to ~1.0"
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0.0-1.0)")

    @field_validator("emotion_distribution")
    @classmethod
    def validate_emotion_distribution(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Validate that emotion distribution has all 7 emotions and sums to ~1.0"""
        required_emotions = {e.value for e in EmotionLabel}

        if set(v.keys()) != required_emotions:
            raise ValueError(
                f"emotion_distribution must contain all 7 emotions: {required_emotions}"
            )

        total = sum(v.values())
        if not (0.95 <= total <= 1.05):
            raise ValueError(f"emotion_distribution must sum to ~1.0, got {total}")

        return v


class EmotionProfile(BaseModel):
    """Emotion profile for a speaker across conversation"""

    speaker_id: str = Field(..., description="Speaker identifier")
    dominant_emotion: EmotionLabel = Field(..., description="Most frequent emotion")
    average_intensity: float = Field(..., ge=0.0, le=1.0, description="Average emotion intensity")
    emotion_frequency: Dict[str, int] = Field(..., description="Frequency count of each emotion")


class EmotionTimeline(BaseModel):
    """Timeline of emotion changes for a speaker"""

    speaker_id: str = Field(..., description="Speaker identifier")
    timestamps: list[datetime] = Field(..., description="Timestamps of utterances")
    emotions: list[EmotionLabel] = Field(..., description="Emotion at each timestamp")
    intensities: list[float] = Field(..., description="Intensity at each timestamp")

    @model_validator(mode="after")
    def validate_timeline_length(self) -> "EmotionTimeline":
        """Validate that all lists have the same length"""
        if not (len(self.timestamps) == len(self.emotions) == len(self.intensities)):
            raise ValueError(
                f"timestamps, emotions, and intensities must have the same length: "
                f"got {len(self.timestamps)}, {len(self.emotions)}, {len(self.intensities)}"
            )
        return self

    @field_validator("intensities")
    @classmethod
    def validate_intensity_range(cls, v: list[float]) -> list[float]:
        """Validate that all intensities are in valid range"""
        for intensity in v:
            if not (0.0 <= intensity <= 1.0):
                raise ValueError(f"All intensities must be between 0.0 and 1.0, got {intensity}")
        return v
