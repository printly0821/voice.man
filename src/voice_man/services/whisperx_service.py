"""
WhisperX Service

High-level service layer for WhisperX pipeline.
Implements SPEC-WHISPERX-001 requirements.

EARS Requirements Implemented:
- F1: WhisperXPipeline integration class
- F5: Backward compatibility with existing diarization_service interface
- E3: Real-time progress updates per pipeline stage
- E4: Auto speaker count detection or manual specification
"""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from voice_man.models.whisperx_pipeline import WhisperXPipeline, PipelineResult
from voice_man.services.audio_converter_service import AudioConverterService

logger = logging.getLogger(__name__)


class WhisperXService:
    """
    High-level WhisperX service for audio processing.

    F1: Provides unified interface for WhisperX pipeline.
    F5: Maintains backward compatibility with existing services.
    """

    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "cuda",
        language: str = "ko",
        compute_type: str = "float16",
    ):
        """
        Initialize WhisperX service.

        Args:
            model_size: Whisper model size (default: large-v3)
            device: Device to use (cuda/cpu/auto)
            language: Language code (default: ko)
            compute_type: Compute type (default: float16)
        """
        self.model_size = model_size
        self.device = device
        self.language = language
        self.compute_type = compute_type

        # Initialize pipeline
        self._pipeline = WhisperXPipeline(
            model_size=model_size,
            device=device,
            language=language,
            compute_type=compute_type,
        )

        # Initialize audio converter
        self._converter = AudioConverterService()

        # Track cleanup state for idempotency
        self._cleaned_up = False

        logger.info(
            f"WhisperXService initialized: model={model_size}, device={device}, language={language}"
        )

    async def process_audio(
        self,
        audio_path: str,
        num_speakers: Optional[int] = None,
        progress_callback: Optional[Callable[[str, float, str], None]] = None,
        existing_transcription: Optional[Dict[str, Any]] = None,
    ) -> PipelineResult:
        """
        Process audio file through WhisperX pipeline.

        Args:
            audio_path: Path to audio file
            num_speakers: Number of speakers (None for auto-detection)
            progress_callback: Optional callback for progress updates
            existing_transcription: Optional existing transcription to reuse.
                                  Skips transcription/alignment, only diarization.

        Returns:
            PipelineResult with transcription and speaker data

        Implements:
            E3: Real-time progress updates per pipeline stage
            E4: Auto speaker count detection or manual specification
        """
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info(f"Processing audio: {audio_path}")

        # Convert audio if needed
        async with self._converter.convert_context(audio_path) as converted_path:
            # Process through pipeline
            result = await self._pipeline.process(
                converted_path,
                num_speakers=num_speakers,
                progress_callback=progress_callback,
                existing_transcription=existing_transcription,
            )

        return result

    async def transcribe_only(self, audio_path: str) -> Dict[str, Any]:
        """
        Transcribe audio without diarization.

        Args:
            audio_path: Path to audio file

        Returns:
            Dictionary with transcription result
        """
        async with self._converter.convert_context(audio_path) as converted_path:
            result = await self._pipeline.transcribe(converted_path)

        return result

    def get_speaker_stats(self, segments: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Generate speaker statistics from segments.

        F4: Per-speaker speech statistics.

        Args:
            segments: List of segments with speaker information

        Returns:
            Dictionary with per-speaker statistics
        """
        return self._pipeline.generate_speaker_stats(segments)

    def cleanup(self) -> None:
        """
        Release all resources and clear memory.

        Implements ServiceCleanupProtocol for memory management.
        This method is idempotent and can be called multiple times safely.

        Actions performed:
        - Unloads WhisperX pipeline models
        - Clears GPU memory cache
        - Clears internal caches
        """
        if self._cleaned_up:
            logger.debug("WhisperXService already cleaned up, skipping")
            return

        try:
            logger.info("Starting WhisperXService cleanup...")

            # Unload pipeline models
            if hasattr(self, "_pipeline") and self._pipeline is not None:
                self._pipeline.unload()
                logger.debug("WhisperX pipeline unloaded")

            # Clear GPU cache if available
            try:
                import gc
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    if hasattr(torch.cuda, "ipc_collect"):
                        torch.cuda.ipc_collect()
                    # Force garbage collection to free Python object memory
                    gc.collect()
                    logger.debug("GPU cache cleared and GC performed")
            except ImportError:
                pass
            except Exception as e:
                logger.warning(f"Failed to clear GPU cache: {e}")

            # Mark as cleaned up
            self._cleaned_up = True
            logger.info("WhisperXService cleanup completed")

        except Exception as e:
            logger.error(f"Error during WhisperXService cleanup: {e}")
            # Still mark as cleaned up to prevent repeated attempts
            self._cleaned_up = True

    def get_memory_usage(self) -> float:
        """
        Get current memory usage in MB.

        Implements ServiceCleanupProtocol for memory monitoring.

        Returns:
            Memory usage in megabytes
        """
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except Exception as e:
            logger.warning(f"Failed to get memory usage: {e}")
            return 0.0

    def unload(self) -> None:
        """Unload all models to free memory."""
        self._pipeline.unload()
        logger.info("WhisperX service unloaded")

    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.cleanup()
        except Exception:
            pass


class WhisperXServiceFactory:
    """Factory for creating WhisperXService instances."""

    _instance: Optional[WhisperXService] = None

    @classmethod
    def get_instance(
        cls,
        model_size: str = "large-v3",
        device: str = "cuda",
        language: str = "ko",
    ) -> WhisperXService:
        """
        Get singleton instance of WhisperXService.

        Args:
            model_size: Whisper model size
            device: Device to use
            language: Language code

        Returns:
            WhisperXService instance
        """
        if cls._instance is None:
            cls._instance = WhisperXService(
                model_size=model_size,
                device=device,
                language=language,
            )
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset singleton instance."""
        if cls._instance is not None:
            cls._instance.unload()
            cls._instance = None
