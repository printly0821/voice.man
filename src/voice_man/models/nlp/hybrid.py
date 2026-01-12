"""
Hybrid Analysis Data Models
SPEC-NLP-KOBERT-001 TAG-004: Hybrid ML + Keyword Analysis
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

from voice_man.models.nlp.emotion import EmotionResult
from voice_man.models.nlp.cultural import SpeechLevelResult


class HybridFallbackReason(str, Enum):
    """Reasons for falling back to keyword analysis"""

    MODEL_NOT_LOADED = "model_not_loaded"
    INFERENCE_TIMEOUT = "inference_timeout"
    GPU_MEMORY_INSUFFICIENT = "gpu_memory_insufficient"
    MODEL_ERROR = "model_error"
    SHORT_TEXT = "short_text"
    USER_PREFERENCE = "user_preference"


@dataclass
class HybridAnalysisResult:
    """
    Hybrid analysis result combining ML and keyword approaches

    Attributes:
        emotion: Emotion classification result
        speech_level: Speech level analysis result
        ml_weight: Weight applied to ML predictions (0-1)
        keyword_weight: Weight applied to keyword analysis (0-1)
        used_fallback: Whether keyword fallback was used
        fallback_reason: Reason for fallback (if applicable)
        inference_time_ms: Time taken for inference
    """

    emotion: EmotionResult
    speech_level: SpeechLevelResult
    ml_weight: float
    keyword_weight: float
    used_fallback: bool
    fallback_reason: Optional[HybridFallbackReason] = None
    inference_time_ms: Optional[float] = None

    def __post_init__(self):
        """Validate hybrid analysis result"""
        if not 0 <= self.ml_weight <= 1:
            raise ValueError(f"ml_weight must be between 0 and 1, got {self.ml_weight}")

        if not 0 <= self.keyword_weight <= 1:
            raise ValueError(f"keyword_weight must be between 0 and 1, got {self.keyword_weight}")

        # Weights should sum to approximately 1
        total = self.ml_weight + self.keyword_weight
        if abs(total - 1.0) > 0.1:  # Allow 10% tolerance
            raise ValueError(f"ml_weight + keyword_weight should sum to ~1.0, got {total}")

        # If fallback was used, reason should be provided
        if self.used_fallback and self.fallback_reason is None:
            raise ValueError("fallback_reason must be provided when used_fallback is True")
