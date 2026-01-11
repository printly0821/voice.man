"""
WhisperX Pipeline Module

End-to-end WhisperX pipeline for STT + Alignment + Diarization.
Implements SPEC-WHISPERX-001 requirements.

EARS Requirements Implemented:
- F1: WhisperXPipeline integration class
- F2: WAV2VEC2 word-level alignment
- F3: Pyannote GPU parallel speaker diarization
- F4: Per-speaker speech statistics
- U1: All pipeline steps on same GPU context
- U2: Word-level timestamp accuracy within 100ms
- U3: Consistent speaker ID assignment
- E1: Hugging Face token validation at initialization
- E3: Real-time progress updates per pipeline stage
- E4: Auto speaker count detection or manual specification
- S1: Korean alignment model selection
- S2: Sequential model loading when GPU memory > 70%
- S3: 10-minute chunk splitting for audio > 30 minutes
- N1: No OOM from simultaneous model loading
"""

import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)

# Lazy imports for optional dependencies
whisperx = None
torch = None


class _DistilWhisperWrapper:
    """
    Wrapper for Distil-Whisper model to be compatible with WhisperX pipeline.

    Distil-Whisper is a distilled version of Whisper that provides:
    - 4-6x faster transcription
    - 99% of large-v3 accuracy
    - Lower memory footprint
    """

    def __init__(self, model, processor, language: str = "ko"):
        """
        Initialize Distil-Whisper wrapper.

        Args:
            model: Transformers model for speech seq2seq
            processor: Transformers processor for tokenization
            language: Language code for transcription
        """
        self.model = model
        self.processor = processor
        self.language = language
        self._device = next(model.parameters()).device

    def transcribe(
        self,
        audio: Any,  # np.ndarray
        batch_size: int = 16,
        language: str | None = None,
    ) -> dict:
        """
        Transcribe audio using Distil-Whisper.

        Args:
            audio: Audio numpy array
            batch_size: Batch size for processing (ignored for single file)
            language: Language code (uses init language if None)

        Returns:
            Dictionary with segments and language
        """
        import torch
        import numpy as np

        language = language or self.language

        # Prepare input features - match model dtype
        model_dtype = next(self.model.parameters()).dtype
        input_features = self.processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
        ).input_features.to(self._device, dtype=model_dtype)

        # Generate transcription
        with torch.no_grad():
            forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                language=language,
                task="transcribe",
            )
            pred_ids = self.model.generate(
                input_features,
                forced_decoder_ids=forced_decoder_ids,
            )

        # Decode prediction
        transcription_text = self.processor.batch_decode(
            pred_ids,
            skip_special_tokens=True,
        )[0]

        # Create compatible output format
        # For simplicity, create a single segment ( WhisperX will refine)
        audio_duration = len(audio) / 16000

        result = {
            "segments": [
                {
                    "start": 0.0,
                    "end": audio_duration,
                    "text": transcription_text.strip(),
                }
            ],
            "language": language,
        }

        return result


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


@dataclass
class PipelineResult:
    """
    Result from WhisperX pipeline processing.

    Contains transcription text, segments with timing, speaker information,
    and speaker statistics.
    """

    text: str
    segments: List[Dict[str, Any]]
    speakers: List[str]
    speaker_stats: Dict[str, Dict[str, Any]]
    language: str
    word_segments: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "text": self.text,
            "segments": self.segments,
            "speakers": self.speakers,
            "speaker_stats": self.speaker_stats,
            "language": self.language,
            "word_segments": self.word_segments,
        }


class WhisperXPipeline:
    """
    WhisperX integration pipeline for end-to-end audio processing.

    F1: Provides unified interface for transcription, alignment, and diarization.
    U1: All models run on the same GPU context.
    """

    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "cuda",
        language: str = "ko",
        compute_type: str = "float16",
    ):
        """
        Initialize WhisperX pipeline.

        Args:
            model_size: Whisper model size (default: large-v3)
            device: Device to use - "auto", "cuda", or "cpu" (default: cuda)
            language: Language code (default: ko)
            compute_type: Compute type (default: float16)

        Raises:
            HFTokenNotFoundError: If HF_TOKEN is not set

        Implements:
            E1: HF token validation at initialization
            U1: All models on same GPU context
        """
        from voice_man.config.whisperx_config import (
            WhisperXConfig,
            get_hf_token,
        )
        from voice_man.services.gpu_monitor_service import GPUMonitorService

        # E1: Validate HF token at initialization
        self._hf_token = get_hf_token()

        # Initialize configuration
        self.config = WhisperXConfig(
            model_size=model_size,
            device=device,
            language=language,
            compute_type=compute_type,
        )

        self.model_size = model_size
        self.language = language

        # Resolve device
        self.device = self._resolve_device(device)

        # U1: All models use same device
        self._whisper_device = self.device
        self._align_device = self.device
        self._diarize_device = self.device

        # Initialize GPU monitor for memory management
        self._gpu_monitor = GPUMonitorService()

        # S2: Check if sequential loading is needed
        self._sequential_loading = self._check_sequential_loading_needed()

        # Initialize models (lazy loading)
        self._whisper_model = None
        self._align_model = None
        self._align_metadata = None
        self._diarize_pipeline = None

        # Load whisper model
        self._load_whisper_model()

        logger.info(
            f"WhisperXPipeline initialized: model={model_size}, "
            f"device={self.device}, language={language}, "
            f"sequential_loading={self._sequential_loading}"
        )

    def _resolve_device(self, device: str) -> str:
        """
        Resolve device string to actual device.

        Args:
            device: Device specification ("auto", "cuda", "cpu")

        Returns:
            Resolved device string
        """
        if device == "auto":
            torch = _import_torch()
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _check_sequential_loading_needed(self) -> bool:
        """
        Check if sequential model loading is needed based on GPU memory.

        S2: Sequential loading when GPU memory > 70%.

        Returns:
            True if sequential loading is needed
        """
        if self.device == "cpu":
            return False

        try:
            memory_stats = self._gpu_monitor.get_gpu_memory_stats()
            if memory_stats.get("available", False):
                usage = memory_stats.get("usage_percentage", 0)
                if usage > self.config.gpu_memory_threshold:
                    logger.warning(
                        f"GPU memory usage ({usage:.1f}%) exceeds threshold "
                        f"({self.config.gpu_memory_threshold}%). Using sequential loading."
                    )
                    return True
        except Exception as e:
            logger.warning(f"Could not check GPU memory: {e}")

        return False

    def _load_whisper_model(self) -> None:
        """
        Load Whisper model.

        Supports Distil-Whisper for 4-6x faster transcription with 99% accuracy.
        """
        wx = _import_whisperx()

        logger.info(f"Loading Whisper model: {self.model_size}")

        # Handle Distil-Whisper model
        if self.model_size.startswith("distil-"):
            # Distil-Whisper models use HuggingFace transformers
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
            import torch

            model_name = f"distil-whisper/{self.model_size}"

            logger.info(f"Loading Distil-Whisper model from: {model_name}")

            # Load model
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_name,
                torch_dtype=torch.float16
                if self.config.compute_type == "float16"
                else torch.float32,
                low_cpu_mem_usage=True,
                use_safetensors=True,
            )

            # Load processor
            processor = AutoProcessor.from_pretrained(model_name)

            # Move to device
            if self.device == "cuda" and torch.cuda.is_available():
                model.to(torch.device(self.device))

            # Wrap in compatible format for whisperx
            self._whisper_model = _DistilWhisperWrapper(model, processor, self.language)
        else:
            # Standard Whisper model loading
            self._whisper_model = wx.load_model(
                self.model_size,
                device=self.device,
                compute_type=self.config.compute_type,
                language=self.language,
            )

    def _load_align_model(self) -> None:
        """
        Load alignment model.

        S1: Uses language-specific alignment model.
        """
        if self._align_model is not None:
            return

        wx = _import_whisperx()

        logger.info(f"Loading alignment model for language: {self.language}")
        self._align_model, self._align_metadata = wx.load_align_model(
            language_code=self.language,
            device=self.device,
        )

    def _load_diarize_pipeline(self) -> None:
        """Load diarization pipeline using pyannote.audio directly."""
        if self._diarize_pipeline is not None:
            return

        logger.info("Loading diarization pipeline")
        try:
            from pyannote.audio import Pipeline

            self._diarize_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self._hf_token,
            )
            # Move to device
            import torch

            if self.device == "cuda" and torch.cuda.is_available():
                self._diarize_pipeline.to(torch.device("cuda"))
            logger.info(f"Diarization pipeline loaded on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load diarization pipeline: {e}")
            raise

    def _get_audio_duration(self, audio_path: str) -> float:
        """
        Get audio duration in seconds.

        Args:
            audio_path: Path to audio file

        Returns:
            Duration in seconds
        """
        try:
            import librosa

            duration = librosa.get_duration(path=audio_path)
            return duration
        except Exception as e:
            logger.warning(f"Could not get audio duration: {e}")
            return 0.0

    def _split_audio_to_chunks(self, audio_path: str) -> List[Dict[str, Any]]:
        """
        Split long audio into chunks.

        S3: 10-minute chunks with 30-second overlap for audio > 30 minutes.

        Args:
            audio_path: Path to audio file

        Returns:
            List of chunk information dictionaries
        """
        duration = self._get_audio_duration(audio_path)

        if duration <= self.config.max_audio_duration:
            return [{"path": audio_path, "start": 0, "end": duration}]

        chunks = []
        chunk_start = 0
        chunk_duration = self.config.chunk_duration
        overlap = self.config.chunk_overlap

        while chunk_start < duration:
            chunk_end = min(chunk_start + chunk_duration, duration)
            chunks.append(
                {
                    "path": audio_path,
                    "start": chunk_start,
                    "end": chunk_end,
                }
            )
            chunk_start = chunk_end - overlap

        logger.info(f"Split audio into {len(chunks)} chunks")
        return chunks

    async def transcribe(self, audio_path: str) -> Dict[str, Any]:
        """
        Transcribe audio file.

        F1: STT stage of pipeline.

        Args:
            audio_path: Path to audio file

        Returns:
            Dictionary with segments and language
        """
        wx = _import_whisperx()

        logger.info(f"Transcribing: {audio_path}")
        audio = wx.load_audio(audio_path)

        result = self._whisper_model.transcribe(
            audio,
            batch_size=16,
            language=self.language,
        )

        return result

    async def _transcribe(self, audio_path: str) -> Dict[str, Any]:
        """Internal transcribe method."""
        return await self.transcribe(audio_path)

    async def align(self, segments: Dict[str, Any], audio: Any) -> Dict[str, Any]:
        """
        Align transcription with word-level timestamps.

        F2: WAV2VEC2 word-level alignment.
        U2: Accuracy within 100ms.

        Args:
            segments: Transcription segments
            audio: Loaded audio data

        Returns:
            Dictionary with aligned segments including word timestamps
        """
        wx = _import_whisperx()

        # Load alignment model if needed
        self._load_align_model()

        logger.info("Aligning transcription with word-level timestamps")
        result = wx.align(
            segments["segments"],
            self._align_model,
            self._align_metadata,
            audio,
            self.device,
            return_char_alignments=False,
        )

        return result

    async def _align(self, segments: Dict[str, Any], audio: Any) -> Dict[str, Any]:
        """Internal align method."""
        return await self.align(segments, audio)

    async def diarize(
        self,
        audio: Any,
        segments: Dict[str, Any],
        num_speakers: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Perform speaker diarization.

        F3: Pyannote GPU parallel speaker diarization.
        U3: Consistent speaker ID assignment.
        E4: Auto speaker count detection or manual specification.

        Args:
            audio: Loaded audio data (numpy array or dict with waveform/sample_rate)
            segments: Aligned segments
            num_speakers: Number of speakers (None for auto-detection)

        Returns:
            Dictionary with speaker-assigned segments
        """
        import torch
        import pandas as pd

        wx = _import_whisperx()

        # Load diarization pipeline if needed
        self._load_diarize_pipeline()

        logger.info(f"Performing speaker diarization (num_speakers={num_speakers or 'auto'})")

        # Prepare audio for pyannote (expects dict with waveform tensor and sample_rate)
        if isinstance(audio, dict) and "waveform" in audio:
            audio_input = audio
        else:
            # Convert numpy array to pyannote format
            import numpy as np

            if isinstance(audio, np.ndarray):
                waveform = torch.from_numpy(audio).unsqueeze(0)  # Add channel dimension
                if waveform.dtype != torch.float32:
                    waveform = waveform.float()
                audio_input = {"waveform": waveform, "sample_rate": 16000}
            else:
                audio_input = audio

        # Run pyannote diarization
        min_spk = self.config.min_speakers if num_speakers is None else num_speakers
        max_spk = self.config.max_speakers if num_speakers is None else num_speakers

        diarization = self._diarize_pipeline(
            audio_input,
            min_speakers=min_spk,
            max_speakers=max_spk,
        )

        # Convert pyannote Annotation to DataFrame format expected by whisperx
        diarize_segments = pd.DataFrame(
            [
                {"start": turn.start, "end": turn.end, "speaker": speaker}
                for turn, _, speaker in diarization.itertracks(yield_label=True)
            ]
        )

        # Assign speakers to segments
        result = wx.assign_word_speakers(diarize_segments, segments)

        return result

    async def _diarize(
        self,
        audio: Any,
        segments: Dict[str, Any],
        num_speakers: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Internal diarize method."""
        return await self.diarize(audio, segments, num_speakers)

    def generate_speaker_stats(self, segments: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Generate speaker statistics.

        F4: Per-speaker speech statistics.

        Args:
            segments: List of segments with speaker information

        Returns:
            Dictionary with per-speaker statistics
        """
        stats: Dict[str, Dict[str, Any]] = {}

        for segment in segments:
            speaker = segment.get("speaker", "UNKNOWN")
            start = segment.get("start", 0.0)
            end = segment.get("end", 0.0)
            duration = end - start

            if speaker not in stats:
                stats[speaker] = {
                    "total_duration": 0.0,
                    "turn_count": 0,
                    "segments": [],
                }

            stats[speaker]["total_duration"] += duration
            stats[speaker]["turn_count"] += 1
            stats[speaker]["segments"].append(segment)

        # Calculate speech ratios
        total_duration = sum(s["total_duration"] for s in stats.values())
        for speaker in stats:
            if total_duration > 0:
                stats[speaker]["speech_ratio"] = stats[speaker]["total_duration"] / total_duration
            else:
                stats[speaker]["speech_ratio"] = 0.0

        return stats

    async def process(
        self,
        audio_path: str,
        num_speakers: Optional[int] = None,
        progress_callback: Optional[Callable[[str, float, str], None]] = None,
        existing_transcription: Optional[Dict[str, Any]] = None,
    ) -> PipelineResult:
        """
        Process audio file through complete pipeline.

        F1: End-to-end pipeline execution.
        E3: Real-time progress updates per pipeline stage.

        Args:
            audio_path: Path to audio file
            num_speakers: Number of speakers (None for auto-detection)
            progress_callback: Optional callback for progress updates
                              (stage: str, progress: float, message: str)
            existing_transcription: Optional existing transcription data to reuse.
                                  If provided, skips transcription and alignment,
                                  only performs diarization.

        Returns:
            PipelineResult with complete transcription and speaker data
        """
        wx = _import_whisperx()

        def report_progress(stage: str, progress: float, message: str):
            if progress_callback:
                progress_callback(stage, progress, message)
            logger.info(f"[{stage}] {progress:.1f}%: {message}")

        # Load audio
        report_progress("transcription", 0, f"Loading audio: {audio_path}")
        audio = wx.load_audio(audio_path)

        # Check if we can reuse existing transcription
        if existing_transcription:
            logger.info("Reusing existing transcription, skipping transcribe/align stages")
            report_progress("transcription", 40, "Reusing existing transcription")

            # Create aligned format from existing data
            aligned = {"segments": existing_transcription.get("segments", [])}
        else:
            # Stage 1: Transcription (0-40%)
            report_progress("transcription", 10, "Starting transcription")
            transcription = await self._transcribe(audio_path)
            report_progress("transcription", 40, "Transcription complete")

            # Unload whisper model if sequential loading
            if self._sequential_loading:
                self._unload_whisper_model()

            # Stage 2: Alignment (40-70%)
            report_progress("alignment", 40, "Starting alignment")
            aligned = await self._align(transcription, audio)
            report_progress("alignment", 70, "Alignment complete")

            # Unload alignment model if sequential loading
            if self._sequential_loading:
                self._unload_align_model()

        # Stage 3: Diarization (70-100%)
        report_progress("diarization", 70, "Starting diarization")
        diarized = await self._diarize(audio, aligned, num_speakers)
        report_progress("diarization", 100, "Diarization complete")

        # Extract results
        segments = diarized.get("segments", [])
        speakers = list(set(seg.get("speaker", "UNKNOWN") for seg in segments))
        speaker_stats = self.generate_speaker_stats(segments)

        # Extract word segments
        word_segments = []
        for seg in segments:
            if "words" in seg:
                word_segments.extend(seg["words"])

        # Combine text
        text = " ".join(seg.get("text", "") for seg in segments)

        return PipelineResult(
            text=text,
            segments=segments,
            speakers=speakers,
            speaker_stats=speaker_stats,
            language=self.language,
            word_segments=word_segments,
        )

    def _unload_whisper_model(self) -> None:
        """Unload Whisper model to free memory."""
        if self._whisper_model is not None:
            del self._whisper_model
            self._whisper_model = None
            self._clear_gpu_cache()
            logger.info("Whisper model unloaded")

    def _unload_align_model(self) -> None:
        """Unload alignment model to free memory."""
        if self._align_model is not None:
            del self._align_model
            del self._align_metadata
            self._align_model = None
            self._align_metadata = None
            self._clear_gpu_cache()
            logger.info("Alignment model unloaded")

    def _clear_gpu_cache(self) -> None:
        """Clear GPU cache."""
        if self.device == "cuda":
            try:
                torch = _import_torch()
                torch.cuda.empty_cache()
                logger.debug("GPU cache cleared")
            except Exception as e:
                logger.warning(f"Could not clear GPU cache: {e}")

    def unload(self) -> None:
        """
        Unload all models to free memory.

        N1: Proper cleanup to prevent OOM.
        """
        self._unload_whisper_model()
        self._unload_align_model()

        if self._diarize_pipeline is not None:
            del self._diarize_pipeline
            self._diarize_pipeline = None

        self._clear_gpu_cache()
        logger.info("All models unloaded")

    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.unload()
        except Exception:
            pass
