"""
NLP Services Module
SPEC-NLP-KOBERT-001: KoBERT-based NLP analysis services

Phase 1 Improvement (from gaslighting-forensic-ai-guide.md):
- KorporaService: Korean corpus integration (NSMC, hate speech, etc.)

Phase 2 Improvement (KoBERT Fine-tuning):
- KoBERTFineTuner: Training pipeline for custom emotion models
- NSMCAdapter: Adapts NSMC dataset for emotion classification
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

try:
    from voice_man.services.nlp.korpora_service import (
        KorporaService,
        KorpusDataset,
        DatasetInfo,
        get_korpora_service,
    )

    _available_exports.extend(
        ["KorporaService", "KorpusDataset", "DatasetInfo", "get_korpora_service"]
    )
except ImportError:
    pass

try:
    from voice_man.services.nlp.training import (
        KoBERTFineTuner,
        NSMCAdapter,
        create_default_config,
        TrainingConfig,
        TrainingHistory,
        TrainingMetrics,
        ModelMetadata,
        OptimizerType,
        SchedulerType,
        LossType,
    )

    _available_exports.extend(
        [
            "KoBERTFineTuner",
            "NSMCAdapter",
            "create_default_config",
            "TrainingConfig",
            "TrainingHistory",
            "TrainingMetrics",
            "ModelMetadata",
            "OptimizerType",
            "SchedulerType",
            "LossType",
        ]
    )
except ImportError:
    pass

__all__ = _available_exports
