"""
Faster-whisper model wrapper for GPU-accelerated STT.

Provides a unified interface for faster-whisper with automatic device selection
and compute type optimization based on EARS requirements (SPEC-PARALLEL-001).

EARS Requirements Implemented:
- F1: faster-whisper based STT (GPU: float16, CPU: int8)
- E1: GPU availability check with CPU fallback
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Iterator
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionSegment:
    """A single transcription segment with timing information."""

    start: float
    end: float
    text: str


@dataclass
class TranscriptionResult:
    """Complete transcription result."""

    text: str
    segments: List[Dict]
    language: str
    language_probability: float


class WhisperModelWrapper:
    """
    Wrapper for faster-whisper model with automatic device and compute type selection.

    Implements F1: faster-whisper based STT with GPU (float16) and CPU (int8) support.
    """

    # Supported model sizes
    SUPPORTED_MODELS = [
        "tiny",
        "tiny.en",
        "base",
        "base.en",
        "small",
        "small.en",
        "medium",
        "medium.en",
        "large-v1",
        "large-v2",
        "large-v3",
    ]

    # Compute type mapping
    COMPUTE_TYPES = {
        "cuda": "float16",  # F1: GPU uses float16
        "cpu": "int8",  # F1: CPU uses int8
    }

    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "auto",
        compute_type: Optional[str] = None,
        download_root: Optional[str] = None,
        num_workers: int = 1,
    ):
        """
        Initialize the Whisper model wrapper.

        Args:
            model_size: Whisper model size (default: large-v3)
            device: Device to use - "auto", "cuda", or "cpu" (default: auto)
            compute_type: Compute type override (default: auto-selected based on device)
            download_root: Directory for model downloads
            num_workers: Number of workers for transcription

        Implements:
            F1: faster-whisper based STT (GPU: float16, CPU: int8)
            E1: Auto device selection with GPU preference
        """
        self.model_size = model_size
        self.num_workers = num_workers
        self.download_root = download_root

        # Determine device
        self.device = self._resolve_device(device)

        # Determine compute type
        self.compute_type = compute_type or self.COMPUTE_TYPES.get(self.device, "int8")

        # Initialize model
        self._model = None
        self._load_model()

        logger.info(
            f"WhisperModelWrapper initialized: model={model_size}, "
            f"device={self.device}, compute_type={self.compute_type}"
        )

    def _resolve_device(self, device: str) -> str:
        """
        Resolve device string to actual device.

        Args:
            device: Device specification ("auto", "cuda", "cpu")

        Returns:
            Resolved device string ("cuda" or "cpu")

        Implements:
            E1: GPU availability check with CPU fallback
        """
        if device == "auto":
            return "cuda" if self._is_cuda_available() else "cpu"
        return device

    def _is_cuda_available(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            logger.warning("PyTorch not installed, assuming CUDA unavailable")
            return False
        except Exception as e:
            logger.warning(f"Error checking CUDA availability: {e}")
            return False

    def _load_model(self) -> None:
        """
        Load the faster-whisper model.

        Implements:
            F1: Load model with appropriate compute type for device
        """
        try:
            from faster_whisper import WhisperModel

            logger.info(f"Loading faster-whisper model: {self.model_size}")

            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                download_root=self.download_root,
                num_workers=self.num_workers,
            )

            logger.info(
                f"Model loaded successfully on {self.device} with {self.compute_type}"
            )

        except ImportError:
            logger.error(
                "faster-whisper not installed. Install with: pip install faster-whisper"
            )
            raise RuntimeError("faster-whisper not available")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise

    def transcribe(
        self,
        audio_path: Union[str, Path],
        language: Optional[str] = "ko",
        task: str = "transcribe",
        beam_size: int = 5,
        best_of: int = 5,
        patience: float = 1.0,
        temperature: Union[float, List[float]] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        vad_filter: bool = True,
        word_timestamps: bool = False,
    ) -> Dict:
        """
        Transcribe audio file to text.

        Args:
            audio_path: Path to audio file
            language: Language code (default: "ko" for Korean)
            task: Task type - "transcribe" or "translate"
            beam_size: Beam size for decoding
            best_of: Number of candidates to consider
            patience: Beam search patience factor
            temperature: Temperature for sampling
            vad_filter: Enable Voice Activity Detection filter
            word_timestamps: Include word-level timestamps

        Returns:
            Dictionary with transcription results:
            - text: Full transcribed text
            - segments: List of segment dictionaries with timing
            - language: Detected or specified language
            - language_probability: Confidence in language detection
        """
        if self._model is None:
            raise RuntimeError("Model not loaded")

        audio_path = str(audio_path)
        logger.info(f"Transcribing: {audio_path}")

        try:
            segments_iter, info = self._model.transcribe(
                audio_path,
                language=language,
                task=task,
                beam_size=beam_size,
                best_of=best_of,
                patience=patience,
                temperature=temperature,
                vad_filter=vad_filter,
                word_timestamps=word_timestamps,
            )

            # Process segments
            segments = []
            texts = []

            for segment in segments_iter:
                segment_dict = {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip(),
                }
                segments.append(segment_dict)
                texts.append(segment.text.strip())

            full_text = " ".join(texts)

            result = {
                "text": full_text,
                "segments": segments,
                "language": info.language,
                "language_probability": info.language_probability,
            }

            logger.info(
                f"Transcription complete: {len(segments)} segments, "
                f"{len(full_text)} characters, language={info.language}"
            )

            return result

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise

    def transcribe_batch(
        self,
        audio_paths: List[Union[str, Path]],
        language: Optional[str] = "ko",
        **kwargs,
    ) -> List[Dict]:
        """
        Transcribe multiple audio files.

        Args:
            audio_paths: List of paths to audio files
            language: Language code
            **kwargs: Additional arguments passed to transcribe()

        Returns:
            List of transcription result dictionaries
        """
        results = []
        for path in audio_paths:
            try:
                result = self.transcribe(path, language=language, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to transcribe {path}: {e}")
                results.append(
                    {
                        "text": "",
                        "segments": [],
                        "language": language or "unknown",
                        "language_probability": 0.0,
                        "error": str(e),
                    }
                )
        return results

    def get_model_info(self) -> Dict:
        """
        Get model configuration information.

        Returns:
            Dictionary with model configuration
        """
        return {
            "model_size": self.model_size,
            "device": self.device,
            "compute_type": self.compute_type,
            "num_workers": self.num_workers,
            "loaded": self._model is not None,
        }

    def unload(self) -> None:
        """Unload the model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None

            # Clear GPU cache if using CUDA
            if self.device == "cuda":
                try:
                    import torch

                    torch.cuda.empty_cache()
                    logger.info("GPU cache cleared after model unload")
                except Exception:
                    pass

            logger.info("Model unloaded")
