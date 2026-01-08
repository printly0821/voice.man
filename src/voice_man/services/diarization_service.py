"""
Speaker Diarization Service

Implements SPEC-WHISPERX-001 requirements.
Uses pyannote-audio for speaker diarization.

EARS Requirements Implemented:
- F3: Pyannote 3.1 speaker diarization
- F4: Per-speaker speech statistics
- F5: Backward compatibility with existing interface
- E4: Auto speaker count detection or manual specification
- U3: Consistent speaker ID assignment
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from voice_man.models.diarization import (
    DiarizationResult,
    Speaker,
    SpeakerTurn,
    SpeakerStats,
)
from voice_man.models.database import TranscriptSegment

logger = logging.getLogger(__name__)

# Lazy imports
Pipeline = None
torch = None


def _import_pyannote():
    """Lazy import pyannote pipeline."""
    global Pipeline
    if Pipeline is None:
        try:
            from pyannote.audio import Pipeline as PyPipeline

            Pipeline = PyPipeline
        except ImportError:
            raise ImportError(
                "pyannote.audio not installed. Install with: pip install pyannote.audio"
            )
    return Pipeline


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


class DiarizationService:
    """
    Speaker Diarization Service.

    F3: Uses pyannote-audio for speaker diarization.
    F5: Maintains backward compatibility with existing interface.
    """

    # Default diarization model
    DEFAULT_MODEL = "pyannote/speaker-diarization-3.1"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cuda",
        use_auth_token: Optional[str] = None,
    ):
        """
        Initialize diarization service.

        Args:
            model_name: Pyannote model name (default: speaker-diarization-3.1)
            device: Device to use (cuda/cpu)
            use_auth_token: HuggingFace token (default: from HF_TOKEN env var)

        F5: Backward compatible initialization.
        """
        self.model_name = model_name
        self.device = device
        self.model_loaded = False
        self._pipeline = None

        # E1: Get HF token from environment if not provided
        self._hf_token = use_auth_token or os.environ.get("HF_TOKEN")

        logger.info(f"DiarizationService initialized: model={model_name}, device={device}")

    def _load_pipeline(self) -> None:
        """
        Load pyannote pipeline.

        F3: Uses pyannote/speaker-diarization-3.1.
        """
        if self._pipeline is not None:
            return

        if not self._hf_token:
            logger.warning("HF_TOKEN not set. Some models may require authentication.")

        try:
            PipelineClass = _import_pyannote()

            logger.info(f"Loading diarization pipeline: {self.model_name}")
            self._pipeline = PipelineClass.from_pretrained(
                self.model_name,
                use_auth_token=self._hf_token,
            )

            # Move to device
            t = _import_torch()
            if self.device == "cuda" and t.cuda.is_available():
                self._pipeline = self._pipeline.to(t.device("cuda"))
            else:
                self._pipeline = self._pipeline.to(t.device("cpu"))

            self.model_loaded = True
            logger.info("Diarization pipeline loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load diarization pipeline: {e}")
            raise

    async def diarize_speakers(
        self,
        audio_path: str,
        num_speakers: Optional[int] = None,
    ) -> DiarizationResult:
        """
        Perform speaker diarization on audio file.

        F3: Pyannote 3.1 speaker diarization.
        E4: Auto speaker count detection or manual specification.
        U3: Consistent speaker ID assignment.

        Args:
            audio_path: Path to audio file
            num_speakers: Number of speakers (None for auto-detection)

        Returns:
            DiarizationResult with speaker information

        Raises:
            ValueError: If audio file is invalid
            FileNotFoundError: If audio file does not exist

        F5: Backward compatible interface.
        """
        # Input validation (F5: same validation as before)
        if not audio_path:
            raise ValueError("Audio path cannot be empty")

        audio_file = Path(audio_path)
        if not audio_file.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if audio_file.stat().st_size < 100:
            raise ValueError("Audio file is too small to process")

        # Load pipeline if needed
        self._load_pipeline()

        logger.info(f"Starting diarization: {audio_path}")

        try:
            # E4: Configure speaker count
            diarization_kwargs: Dict[str, Any] = {}
            if num_speakers is not None:
                diarization_kwargs["num_speakers"] = num_speakers
                logger.info(f"Using specified speaker count: {num_speakers}")
            else:
                logger.info("Using auto speaker count detection")

            # F3: Run pyannote diarization
            diarization = self._pipeline(audio_path, **diarization_kwargs)

            # U3: Convert to consistent speaker format
            speakers = self._convert_diarization_to_speakers(diarization)

            # Calculate total duration and unique speaker count
            total_duration = max(s.end_time for s in speakers) if speakers else 0.0
            unique_speakers = len(set(s.speaker_id for s in speakers))

            result = DiarizationResult(
                speakers=speakers,
                total_duration=total_duration,
                num_speakers=max(1, unique_speakers),  # Ensure at least 1 speaker
            )

            logger.info(
                f"Diarization complete: {result.num_speakers} speakers, "
                f"{total_duration:.2f}s duration"
            )

            return result

        except Exception as e:
            logger.error(f"Diarization failed: {e}")
            raise

    def _convert_diarization_to_speakers(
        self,
        diarization: Any,
    ) -> List[Speaker]:
        """
        Convert pyannote diarization result to Speaker objects.

        U3: Ensures consistent speaker ID assignment.

        Args:
            diarization: Pyannote diarization annotation

        Returns:
            List of Speaker objects
        """
        speakers = []
        speaker_durations: Dict[str, float] = {}

        # Iterate over diarization segments
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            # U3: Consistent speaker ID format
            speaker_id = f"SPEAKER_{speaker.split('_')[-1].zfill(2)}" if "_" in speaker else speaker

            duration = turn.end - turn.start

            # Track cumulative duration per speaker
            speaker_durations[speaker_id] = speaker_durations.get(speaker_id, 0) + duration

            speaker_obj = Speaker(
                speaker_id=speaker_id,
                start_time=turn.start,
                end_time=turn.end,
                duration=duration,
                confidence=0.9,  # Pyannote doesn't provide per-segment confidence
            )
            speakers.append(speaker_obj)

        return speakers

    def merge_with_transcript(
        self,
        stt_segments: List[TranscriptSegment],
        diarization_result: DiarizationResult,
    ) -> List[TranscriptSegment]:
        """
        Merge STT segments with diarization result.

        F5: Backward compatible interface.

        Args:
            stt_segments: STT transcript segments
            diarization_result: Diarization result

        Returns:
            List of TranscriptSegments with speaker IDs assigned
        """
        merged_segments = []

        for segment in stt_segments:
            # Find best matching speaker based on overlap
            best_speaker = None
            max_overlap = 0.0

            for speaker in diarization_result.speakers:
                # Calculate overlap duration
                overlap_start = max(segment.start_time, speaker.start_time)
                overlap_end = min(segment.end_time, speaker.end_time)
                overlap_duration = max(0, overlap_end - overlap_start)

                if overlap_duration > max_overlap:
                    max_overlap = overlap_duration
                    best_speaker = speaker

            # Assign speaker ID
            if best_speaker and max_overlap > 0:
                segment.speaker_id = best_speaker.speaker_id

            merged_segments.append(segment)

        return merged_segments

    def generate_speaker_stats(self, speakers: List[Speaker]) -> SpeakerStats:
        """
        Generate speaker statistics.

        F4: Per-speaker speech statistics.
        F5: Backward compatible interface.

        Args:
            speakers: List of Speaker objects

        Returns:
            SpeakerStats with aggregated statistics
        """
        total_duration = sum(s.duration for s in speakers)

        return SpeakerStats(
            total_speakers=len(set(s.speaker_id for s in speakers)),
            total_speech_duration=total_duration,
            speaker_details=speakers,
        )

    def detect_speaker_turns(self, speakers: List[Speaker]) -> List[SpeakerTurn]:
        """
        Detect speaker turn-taking events.

        F5: Backward compatible interface.

        Args:
            speakers: List of Speaker objects

        Returns:
            List of SpeakerTurn objects representing turn changes
        """
        # Sort by start time
        sorted_speakers = sorted(speakers, key=lambda s: s.start_time)

        turns = []
        for speaker in sorted_speakers:
            turn = SpeakerTurn(
                speaker_id=speaker.speaker_id,
                start_time=speaker.start_time,
                end_time=speaker.end_time,
                duration=speaker.duration,
            )
            turns.append(turn)

        return turns

    def unload(self) -> None:
        """Unload diarization model to free memory."""
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
            self.model_loaded = False

            # Clear GPU cache
            if self.device == "cuda":
                try:
                    t = _import_torch()
                    t.cuda.empty_cache()
                    logger.debug("GPU cache cleared")
                except Exception as e:
                    logger.warning(f"Could not clear GPU cache: {e}")

            logger.info("Diarization model unloaded")

    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.unload()
        except Exception:
            pass
