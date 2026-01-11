"""
Emotion Analysis Data Models
SPEC-NLP-KOBERT-001 TAG-002: Emotion Classification
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


class EmotionCategory(str, Enum):
    """Emotion categories for classification"""

    HAPPINESS = "happiness"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    DISGUST = "disgust"
    SURPRISE = "surprise"
    NEUTRAL = "neutral"


@dataclass
class EmotionResult:
    """
    Emotion classification result

    Attributes:
        primary_emotion: Primary emotion category
        confidence: Confidence score for primary emotion (0-1)
        emotion_scores: All emotion scores (emotion -> score)
        is_uncertain: Whether prediction is uncertain (confidence < threshold)
        key_tokens: Tokens that contributed most to prediction
    """

    primary_emotion: EmotionCategory
    confidence: float
    emotion_scores: Dict[str, float]
    is_uncertain: bool
    key_tokens: Optional[List[str]] = None

    def __post_init__(self):
        """Validate emotion result"""
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")

        # Ensure primary_emotion is in emotion_scores
        if self.primary_emotion.value not in self.emotion_scores:
            raise ValueError(f"Primary emotion {self.primary_emotion.value} not in emotion_scores")
