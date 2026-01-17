"""
WhisperX Configuration Module

Configuration dataclass and utilities for WhisperX pipeline.
Implements SPEC-WHISPERX-001 requirements.

EARS Requirements Implemented:
- U1: GPU context consistency configuration
- U2: Word-level timestamp accuracy target (100ms)
- E1: HF token retrieval from environment only
- E3: Progress stages configuration
- E4: Speaker count limits (1-10)
- S1: Language-specific alignment model mapping
- S2: GPU memory threshold for sequential loading (70%)
- S3: Chunk settings for long audio (10min chunks, 30s overlap)
- N1: Model memory requirements
- N2: No hardcoded HF tokens
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


class HFTokenNotFoundError(Exception):
    """Raised when HF_TOKEN environment variable is not set."""

    def __init__(self, message: str = "HF_TOKEN environment variable not found"):
        self.message = message
        super().__init__(self.message)


def get_hf_token() -> str:
    """
    Retrieve Hugging Face token from environment variable.

    E1: HF token should be retrieved from environment only.
    N2: Token is never hardcoded.

    Returns:
        str: Hugging Face token

    Raises:
        HFTokenNotFoundError: If HF_TOKEN is not set
    """
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise HFTokenNotFoundError(
            "HF_TOKEN environment variable not found. "
            "Please set it with: export HF_TOKEN='your_token'"
        )
    return token


# S1: Language-specific alignment model mapping
ALIGNMENT_MODELS: Dict[str, str] = {
    "ko": "jonatasgrosman/wav2vec2-large-xlsr-53-korean",
    "en": "WAV2VEC2_ASR_BASE_960H",
    "ja": "jonatasgrosman/wav2vec2-large-xlsr-53-japanese",
    "zh": "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
    "es": "jonatasgrosman/wav2vec2-large-xlsr-53-spanish",
    "fr": "jonatasgrosman/wav2vec2-large-xlsr-53-french",
    "de": "jonatasgrosman/wav2vec2-large-xlsr-53-german",
}


@dataclass
class WhisperXConfig:
    """
    WhisperX pipeline configuration.

    Implements SPEC-WHISPERX-001 requirements for configuration management.
    """

    # Model settings
    model_size: str = "distil-large-v3"  # Changed from "large-v3" for memory optimization
    device: str = "cuda"
    compute_type: str = "int8"  # Changed from "float16" for 2x memory savings
    language: str = "ko"

    # F3: Diarization model
    diarization_model: str = "pyannote/speaker-diarization-3.1"

    # S2: GPU memory threshold for sequential loading (70%)
    gpu_memory_threshold: float = 70.0

    # S3: Chunk settings for long audio
    max_audio_duration: int = 1800  # 30 minutes in seconds
    chunk_duration: int = 600  # 10 minutes in seconds
    chunk_overlap: int = 30  # 30 seconds overlap

    # E2: Supported audio formats
    supported_formats: List[str] = field(
        default_factory=lambda: ["m4a", "mp3", "wav", "flac", "ogg"]
    )

    # E4: Speaker count limits
    min_speakers: int = 1
    max_speakers: int = 10

    # U2: Timestamp accuracy target (100ms)
    timestamp_accuracy_ms: int = 100

    # N1: Model memory requirements (in GB)
    whisper_memory_gb: float = 1.5  # Reduced from 3.0 for distil-large-v3
    wav2vec_memory_gb: float = 1.2
    pyannote_memory_gb: float = 1.0

    # E3: Progress stages (start_percent, end_percent)
    progress_stages: Dict[str, Tuple[int, int]] = field(
        default_factory=lambda: {
            "transcription": (0, 40),
            "alignment": (40, 70),
            "diarization": (70, 100),
        }
    )

    @property
    def alignment_model(self) -> str:
        """
        Get alignment model for current language.

        S1: Korean language uses jonatasgrosman/wav2vec2-large-xlsr-53-korean.
        """
        return ALIGNMENT_MODELS.get(self.language, ALIGNMENT_MODELS["en"])

    @property
    def total_memory_gb(self) -> float:
        """
        Calculate total memory required for all models.

        N1: Used for OOM prevention.
        """
        return self.whisper_memory_gb + self.wav2vec_memory_gb + self.pyannote_memory_gb

    @classmethod
    def from_env(cls) -> "WhisperXConfig":
        """
        Create configuration from environment variables.

        Environment variables:
        - WHISPERX_MODEL_SIZE: Whisper model size (default: distil-large-v3)
        - WHISPERX_LANGUAGE: Language code (default: ko)
        - WHISPERX_DEVICE: Device (cuda/cpu) (default: cuda)
        - WHISPERX_COMPUTE_TYPE: Compute type (default: int8)
        """
        return cls(
            model_size=os.environ.get("WHISPERX_MODEL_SIZE", "distil-large-v3"),
            language=os.environ.get("WHISPERX_LANGUAGE", "ko"),
            device=os.environ.get("WHISPERX_DEVICE", "cuda"),
            compute_type=os.environ.get("WHISPERX_COMPUTE_TYPE", "int8"),
        )

    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return {
            "model_size": self.model_size,
            "device": self.device,
            "compute_type": self.compute_type,
            "language": self.language,
            "alignment_model": self.alignment_model,
            "diarization_model": self.diarization_model,
            "gpu_memory_threshold": self.gpu_memory_threshold,
            "max_audio_duration": self.max_audio_duration,
            "chunk_duration": self.chunk_duration,
            "chunk_overlap": self.chunk_overlap,
            "supported_formats": self.supported_formats,
            "min_speakers": self.min_speakers,
            "max_speakers": self.max_speakers,
            "timestamp_accuracy_ms": self.timestamp_accuracy_ms,
            "total_memory_gb": self.total_memory_gb,
            "progress_stages": self.progress_stages,
        }

    def validate(self) -> bool:
        """
        Validate configuration values.

        Returns:
            bool: True if configuration is valid

        Raises:
            ValueError: If configuration is invalid
        """
        valid_models = [
            "tiny",
            "base",
            "small",
            "medium",
            "large-v1",
            "large-v2",
            "large-v3",
            # Distil-Whisper models for memory optimization
            "distil-large-v3",  # 4-6x faster, 99% of large-v3 accuracy
            "distil-medium.en",  # Medium distilled model
            "distil-small.en",   # Small distilled model
            "distil-tiny.en",    # Tiny distilled model (lowest memory)
        ]
        if self.model_size not in valid_models:
            raise ValueError(
                f"Invalid model_size: {self.model_size}. Valid options: {valid_models}"
            )

        if self.device not in ["cuda", "cpu", "auto"]:
            raise ValueError(f"Invalid device: {self.device}")

        # Validate compute_type
        valid_compute_types = ["float16", "float32", "int8", "int16"]
        if self.compute_type not in valid_compute_types:
            raise ValueError(
                f"Invalid compute_type: {self.compute_type}. Valid options: {valid_compute_types}"
            )

        if self.min_speakers < 1 or self.max_speakers > 10:
            raise ValueError("Speaker count must be between 1 and 10")

        return True

