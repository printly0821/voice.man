"""
Forensic Analysis Models
SPEC-FORENSIC-001: Audio forensic analysis models for voice evidence analysis
"""

from voice_man.models.forensic.audio_features import (
    VolumeFeatures,
    PitchFeatures,
    SpeechRateFeatures,
    StressFeatures,
    AudioFeatureAnalysis,
    PauseInfo,
    EscalationZone,
)

from voice_man.models.forensic.crime_language import (
    # Enums
    GaslightingType,
    ThreatType,
    CoercionType,
    DeceptionCategory,
    # Pattern Definitions
    GaslightingPattern,
    ThreatPattern,
    CoercionPattern,
    DeceptionMarker,
    # Match Results
    GaslightingMatch,
    ThreatMatch,
    CoercionMatch,
    DeceptionMarkerMatch,
    # Analysis Results
    DeceptionAnalysis,
    CrimeLanguageScore,
)

from voice_man.models.forensic.emotion_recognition import (
    # Dimensional Emotion
    EmotionDimensions,
    # Categorical Emotion
    CategoricalEmotion,
    EmotionProbabilities,
    # Analysis Results
    EmotionAnalysisResult,
    MultiModelEmotionResult,
    # Forensic Indicators
    ForensicEmotionIndicators,
)

from voice_man.models.forensic.cross_validation import (
    # Enums
    DiscrepancyType,
    DiscrepancySeverity,
    # Models
    Discrepancy,
    TextAnalysisResult,
    VoiceAnalysisResult,
    CrossValidationResult,
)

__all__ = [
    # Audio Features
    "VolumeFeatures",
    "PitchFeatures",
    "SpeechRateFeatures",
    "StressFeatures",
    "AudioFeatureAnalysis",
    "PauseInfo",
    "EscalationZone",
    # Crime Language - Enums
    "GaslightingType",
    "ThreatType",
    "CoercionType",
    "DeceptionCategory",
    # Crime Language - Pattern Definitions
    "GaslightingPattern",
    "ThreatPattern",
    "CoercionPattern",
    "DeceptionMarker",
    # Crime Language - Match Results
    "GaslightingMatch",
    "ThreatMatch",
    "CoercionMatch",
    "DeceptionMarkerMatch",
    # Crime Language - Analysis Results
    "DeceptionAnalysis",
    "CrimeLanguageScore",
    # Emotion Recognition - Dimensional
    "EmotionDimensions",
    # Emotion Recognition - Categorical
    "CategoricalEmotion",
    "EmotionProbabilities",
    # Emotion Recognition - Analysis Results
    "EmotionAnalysisResult",
    "MultiModelEmotionResult",
    # Emotion Recognition - Forensic Indicators
    "ForensicEmotionIndicators",
    # Cross-Validation - Enums
    "DiscrepancyType",
    "DiscrepancySeverity",
    # Cross-Validation - Models
    "Discrepancy",
    "TextAnalysisResult",
    "VoiceAnalysisResult",
    "CrossValidationResult",
]
