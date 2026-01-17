"""
KoBERT Fine-Tuning Package
SPEC-NLP-KOBERT-001 TAG-003: KoBERT Fine-tuning for Korean Emotion Classification

This package provides comprehensive fine-tuning capabilities for KoBERT models
on Korean emotion classification tasks.
"""

from voice_man.models.nlp.training import (
    CheckpointState,
    LossType,
    ModelMetadata,
    OptimizerType,
    SchedulerType,
    TrainingConfig,
    TrainingHistory,
    TrainingMetrics,
)
from voice_man.services.nlp.training.kobert_finetuning import (
    KoBERTFineTuner,
    NSMCAdapter,
    create_default_config,
)

__all__ = [
    # Main fine-tuner
    "KoBERTFineTuner",
    # Utilities
    "NSMCAdapter",
    "create_default_config",
    # Data models
    "TrainingConfig",
    "TrainingHistory",
    "TrainingMetrics",
    "ModelMetadata",
    "CheckpointState",
    # Enums
    "OptimizerType",
    "SchedulerType",
    "LossType",
]
