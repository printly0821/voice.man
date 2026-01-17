"""
Faster-Whisper Wrapper for WhisperX Pipeline Compatibility

Phase 2 Intermediate Optimization (20-30x speedup target):
- CTranslate2-based implementation (4-6x faster than WhisperX)
- INT8/FP16 quantization support
- WhisperX-compatible interface
- Model loading optimizations

Reference: SPEC-GPUOPT-001 Phase 2
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

if TYPE_CHECKING:
    from voice_man.services.gpu_optimization.transcription_cache import TranscriptionCache

logger = logging.getLogger(__name__)


class FasterWhisperWrapper:
    """
    Faster-Whisper wrapper with WhisperX-compatible interface.

    Faster-Whisper Benefits:
    - CTranslate2 engine: 4-6x faster than original Whisper
    - INT8 quantization: 30-40% additional speedup
    - Lower memory footprint
    - Same accuracy as Whisper

    Performance Target: 4-6x over baseline, 20-30x cumulative with Phase 1
    """

    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "int8_float16",
        language: str = "ko",
        cache_service: Optional["TranscriptionCache"] = None,
    ):
        """
        Initialize Faster-Whisper wrapper.

        Args:
            model_size: Whisper model size (default: large-v3)
            device: Device to use (cuda, cpu)
            compute_type: Computation type (default, int8, int8_float16, float16)
            language: Language code (default: ko)
            cache_service: Optional TranscriptionCache service
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.cache_service = cache_service

        # Lazy loading: model is loaded on first use
        self._model = None
        self._loaded = False

        logger.info(
            f"FasterWhisperWrapper initialized: "
            f"model={model_size}, device={device}, compute_type={compute_type}"
        )

    def _load_model(self) -> None:
        """Load Faster-Whisper model."""
        if self._loaded:
            return

        try:
            from faster_whisper import WhisperModel

            logger.info(f"Loading Faster-Whisper model: {self.model_size}")

            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )

            self._loaded = True
            logger.info("Faster-Whisper model loaded successfully")

        except ImportError:
            logger.warning("faster-whisper not installed. Install with: pip install faster-whisper")
            raise
        except Exception as e:
            logger.error(f"Failed to load Faster-Whisper model: {e}")
            raise

    def transcribe(
        self,
        audio: Union[str, bytes, Any],  # Any for np.ndarray
        language: Optional[str] = None,
        batch_size: int = 16,
        word_timestamps: bool = False,
        vad_filter: bool = True,
    ) -> Dict[str, Any]:
        """
        Transcribe audio using Faster-Whisper.

        Args:
            audio: Audio file path or numpy array
            language: Language code (None to auto-detect)
            batch_size: Batch size for processing
            word_timestamps: Whether to return word timestamps
            vad_filter: Whether to use VAD filter

        Returns:
            Dictionary with segments and language (WhisperX-compatible format)
        """
        language = language or self.language

        # Load model if not loaded
        self._load_model()

        if self._model is None:
            raise RuntimeError("Model not loaded")

        import numpy as np

        # Handle different audio input types
        audio_path = None
        audio_array = None

        if isinstance(audio, str):
            audio_path = audio
        elif isinstance(audio, bytes):
            # Write bytes to temp file
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(audio)
                audio_path = f.name
        elif isinstance(audio, np.ndarray):
            audio_array = audio
        else:
            raise TypeError(f"Unsupported audio type: {type(audio)}")

        logger.info(
            f"Transcribing with Faster-Whisper: language={language}, batch_size={batch_size}"
        )

        # Transcribe using Faster-Whisper
        segments, info = self._model.transcribe(
            audio_path or audio_array,
            language=language,
            batch_size=batch_size,
            word_timestamps=word_timestamps,
            vad_filter=vad_filter,
        )

        # Convert to WhisperX-compatible format
        result_segments = []
        for segment in segments:
            segment_dict = {
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
            }

            if word_timestamps and hasattr(segment, "words"):
                segment_dict["words"] = [
                    {
                        "start": word.start,
                        "end": word.end,
                        "word": word.word,
                        "probability": word.probability,
                    }
                    for word in segment.words
                ]

            result_segments.append(segment_dict)

        result = {
            "segments": result_segments,
            "language": info.language if hasattr(info, "language") else language,
        }

        logger.info(f"Transcription complete: {len(result_segments)} segments")

        return result

    def transcribe_stripped(
        self,
        audio: Union[str, bytes, Any],  # Any for np.ndarray
        language: Optional[str] = None,
    ) -> str:
        """
        Transcribe audio and return stripped text only.

        Args:
            audio: Audio file path or numpy array
            language: Language code

        Returns:
            Transcribed text (stripped)
        """
        result = self.transcribe(audio, language=language, word_timestamps=False)
        text = " ".join(seg["text"] for seg in result["segments"])
        return text.strip()

    def unload(self) -> None:
        """Unload model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
            self._loaded = False

            # Clear GPU cache
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

            logger.info("Faster-Whisper model unloaded")

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.

        Returns:
            Dictionary with model info
        """
        return {
            "model_size": self.model_size,
            "device": self.device,
            "compute_type": self.compute_type,
            "language": self.language,
            "loaded": self._loaded,
            "engine": "faster-whisper (CTRANSlate2)",
        }

    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.unload()
        except Exception:
            pass
