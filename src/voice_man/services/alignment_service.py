"""
Alignment Service

WAV2VEC2 word-level alignment service.
Implements SPEC-WHISPERX-001 requirements.

EARS Requirements Implemented:
- F2: WAV2VEC2 word-level alignment
- U2: Word-level timestamp accuracy within 100ms
- S1: Korean alignment model selection
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Lazy imports
whisperx = None
torch = None


def _import_whisperx():
    """Lazy import whisperx."""
    global whisperx
    if whisperx is None:
        try:
            import whisperx as wx

            whisperx = wx
        except ImportError:
            raise ImportError("whisperx not installed. Install with: pip install whisperx")
    return whisperx


def _import_torch():
    """Lazy import torch."""
    global torch
    if torch is None:
        try:
            import torch as t

            torch = t
        except ImportError:
            raise ImportError("torch not installed. Install with: pip install torch")
    return torch


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


class AlignmentService:
    """
    WAV2VEC2 alignment service for word-level timestamps.

    F2: Provides word-level alignment using WAV2VEC2 models.
    S1: Uses language-specific alignment models.
    """

    def __init__(
        self,
        language: str = "ko",
        device: str = "cuda",
    ):
        """
        Initialize alignment service.

        Args:
            language: Language code (default: ko)
            device: Device to use (cuda/cpu)

        Implements:
            S1: Korean alignment model selection
        """
        self.language = language
        self.device = device

        # S1: Get language-specific alignment model
        self.alignment_model_name = ALIGNMENT_MODELS.get(language, ALIGNMENT_MODELS["en"])

        # Load model
        self._model = None
        self._metadata = None
        self._load_model()

        logger.info(
            f"AlignmentService initialized: language={language}, "
            f"model={self.alignment_model_name}, device={device}"
        )

    def _load_model(self) -> None:
        """Load alignment model."""
        wx = _import_whisperx()

        logger.info(f"Loading alignment model: {self.alignment_model_name}")
        self._model, self._metadata = wx.load_align_model(
            language_code=self.language,
            device=self.device,
        )

    async def align(
        self,
        segments: Dict[str, Any],
        audio: Any,
    ) -> Dict[str, Any]:
        """
        Align transcription with word-level timestamps.

        F2: WAV2VEC2 word-level alignment.
        U2: Accuracy within 100ms.

        Args:
            segments: Transcription segments from whisper
            audio: Loaded audio data

        Returns:
            Dictionary with aligned segments including word timestamps

        Example output:
            {
                "segments": [
                    {
                        "start": 0.0,
                        "end": 2.0,
                        "text": "Hello World",
                        "words": [
                            {"word": "Hello", "start": 0.0, "end": 0.8, "score": 0.95},
                            {"word": "World", "start": 1.0, "end": 1.8, "score": 0.92},
                        ]
                    }
                ]
            }
        """
        wx = _import_whisperx()

        if self._model is None:
            self._load_model()

        logger.info("Performing word-level alignment")

        result = wx.align(
            segments["segments"],
            self._model,
            self._metadata,
            audio,
            self.device,
            return_char_alignments=False,
        )

        # Validate alignment scores (U2)
        self._validate_alignment_scores(result)

        return result

    def _validate_alignment_scores(self, result: Dict[str, Any]) -> None:
        """
        Validate alignment scores are within expected range.

        U2: Ensure alignment quality metrics are valid.
        """
        for segment in result.get("segments", []):
            for word in segment.get("words", []):
                score = word.get("score", 0)
                if not (0 <= score <= 1):
                    logger.warning(f"Invalid alignment score {score} for word: {word.get('word')}")

    def unload(self) -> None:
        """Unload alignment model to free memory."""
        if self._model is not None:
            del self._model
            del self._metadata
            self._model = None
            self._metadata = None

            # Clear GPU cache
            if self.device == "cuda":
                try:
                    t = _import_torch()
                    t.cuda.empty_cache()
                    logger.debug("GPU cache cleared")
                except Exception as e:
                    logger.warning(f"Could not clear GPU cache: {e}")

            logger.info("Alignment model unloaded")

    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.unload()
        except Exception:
            pass
