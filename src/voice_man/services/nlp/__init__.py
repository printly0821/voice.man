"""
NLP Services Module
SPEC-NLP-KOBERT-001: KoBERT-based NLP analysis services
"""

# Import only what exists to avoid import errors during TDD
_available_exports = []

try:
    from voice_man.services.nlp.kobert_model import KoBERTModel, DeviceType

    _available_exports.extend(["KoBERTModel", "DeviceType"])
except ImportError:
    pass

try:
    from voice_man.services.nlp.emotion_classifier import (
        KoBERTEmotionClassifier,
        EmotionResult,
        EmotionClassificationHead,
    )

    _available_exports.extend(
        ["KoBERTEmotionClassifier", "EmotionResult", "EmotionClassificationHead"]
    )
except ImportError:
    pass

try:
    from voice_man.services.nlp.cultural_analyzer import (
        KoreanCulturalAnalyzer,
        SpeechLevelResult,
        HierarchyContext,
        ManipulationPattern,
        LevelTransition,
        ComprehensiveAnalysisResult,
    )

    _available_exports.extend(
        [
            "KoreanCulturalAnalyzer",
            "SpeechLevelResult",
            "HierarchyContext",
            "ManipulationPattern",
            "LevelTransition",
            "ComprehensiveAnalysisResult",
        ]
    )
except ImportError:
    pass

__all__ = _available_exports
