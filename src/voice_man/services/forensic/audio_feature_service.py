"""
Audio Feature Service
SPEC-FORENSIC-001 TASK-002~010, 015~016: Audio feature analysis service
SPEC-GPUAUDIO-001 Phase 3: GPU-accelerated audio feature extraction

This service provides audio feature extraction for forensic voice analysis
using librosa for signal processing with optional GPU acceleration.
"""

from typing import List, Dict, Any, Tuple, Optional, TYPE_CHECKING
import logging
import numpy as np

from voice_man.models.forensic.audio_features import (
    VolumeFeatures,
    PitchFeatures,
    SpeechRateFeatures,
    StressFeatures,
    AudioFeatureAnalysis,
    PauseInfo,
    EscalationZone,
)

if TYPE_CHECKING:
    from voice_man.services.forensic.gpu import GPUAudioBackend

logger = logging.getLogger(__name__)


class AudioFeatureService:
    """
    Service for extracting audio features from voice recordings.

    This service provides methods for:
    - Volume analysis (RMS, peak, dynamic range)
    - Pitch analysis (F0, jitter) with optional GPU acceleration
    - Speech rate analysis (WPM, pauses)
    - Emotional escalation detection with batch GPU processing
    """

    # Default parameters
    DEFAULT_HOP_LENGTH = 512
    DEFAULT_FRAME_LENGTH = 2048
    DEFAULT_MIN_PAUSE_DURATION = 0.3  # seconds
    DEFAULT_SILENCE_THRESHOLD_DB = -40  # dB

    def __init__(self, use_gpu: bool = True):
        """
        Initialize the AudioFeatureService.

        Args:
            use_gpu: Whether to attempt GPU acceleration for F0 extraction.
                     Default is True. Falls back to CPU if GPU unavailable.
        """
        self._librosa = None
        self._parselmouth = None
        self._use_gpu = use_gpu
        self._gpu_backend: Optional["GPUAudioBackend"] = None

    @property
    def gpu_backend(self) -> "GPUAudioBackend":
        """
        Get the GPU audio backend (lazy initialization).

        Returns:
            GPUAudioBackend instance configured with use_gpu setting.
        """
        if self._gpu_backend is None:
            from voice_man.services.forensic.gpu import GPUAudioBackend

            self._gpu_backend = GPUAudioBackend(use_gpu=self._use_gpu)
        return self._gpu_backend

    @property
    def librosa(self):
        """Lazy load librosa."""
        if self._librosa is None:
            import librosa

            self._librosa = librosa
        return self._librosa

    def calculate_rms_amplitude(self, audio: np.ndarray, sr: int) -> Tuple[float, float]:
        """
        Calculate RMS (Root Mean Square) amplitude.

        TASK-002: RMS amplitude calculation using librosa.feature.rms

        Args:
            audio: Audio signal as numpy array
            sr: Sample rate in Hz

        Returns:
            Tuple of (rms_amplitude, rms_db)
            - rms_amplitude: Normalized RMS value (0-1)
            - rms_db: RMS in decibels
        """
        # Handle empty or silent audio
        if len(audio) == 0 or np.all(audio == 0):
            return 0.0, -100.0

        # Calculate RMS using librosa
        rms = self.librosa.feature.rms(
            y=audio, frame_length=self.DEFAULT_FRAME_LENGTH, hop_length=self.DEFAULT_HOP_LENGTH
        )[0]

        # Get mean RMS value
        rms_mean = float(np.mean(rms))

        # Normalize to 0-1 range (assuming max amplitude of 1.0)
        rms_normalized = min(1.0, max(0.0, rms_mean))

        # Convert to dB
        if rms_mean > 0:
            rms_db = float(20 * np.log10(rms_mean + 1e-10))
        else:
            rms_db = -100.0

        return rms_normalized, rms_db

    def calculate_peak_amplitude(self, audio: np.ndarray, sr: int) -> Tuple[float, float]:
        """
        Calculate peak amplitude.

        TASK-003: Peak amplitude calculation

        Args:
            audio: Audio signal as numpy array
            sr: Sample rate in Hz

        Returns:
            Tuple of (peak_amplitude, peak_db)
            - peak_amplitude: Normalized peak value (0-1)
            - peak_db: Peak in decibels
        """
        # Handle empty audio
        if len(audio) == 0:
            return 0.0, -100.0

        # Calculate peak amplitude
        peak = float(np.max(np.abs(audio)))

        # Normalize to 0-1 range
        peak_normalized = min(1.0, max(0.0, peak))

        # Convert to dB
        if peak > 0:
            peak_db = float(20 * np.log10(peak + 1e-10))
        else:
            peak_db = -100.0

        return peak_normalized, peak_db

    def calculate_dynamic_range(self, audio: np.ndarray, sr: int) -> float:
        """
        Calculate dynamic range in dB.

        TASK-003: Dynamic range calculation (peak - RMS in dB)

        Args:
            audio: Audio signal as numpy array
            sr: Sample rate in Hz

        Returns:
            Dynamic range in decibels (non-negative)
        """
        _, rms_db = self.calculate_rms_amplitude(audio, sr)
        _, peak_db = self.calculate_peak_amplitude(audio, sr)

        # Handle silent audio
        if peak_db <= -90 or rms_db <= -90:
            return 0.0

        # Dynamic range is peak - RMS (both in dB)
        dynamic_range = max(0.0, peak_db - rms_db)

        return dynamic_range

    def calculate_volume_change_rate(self, audio: np.ndarray, sr: int) -> float:
        """
        Calculate volume change rate in dB/second.

        TASK-004: Volume change rate calculation

        Args:
            audio: Audio signal as numpy array
            sr: Sample rate in Hz

        Returns:
            Volume change rate in dB/second
        """
        # Handle short audio
        if len(audio) < sr:
            return 0.0

        # Calculate RMS over time using frames
        rms = self.librosa.feature.rms(
            y=audio, frame_length=self.DEFAULT_FRAME_LENGTH, hop_length=self.DEFAULT_HOP_LENGTH
        )[0]

        if len(rms) < 2:
            return 0.0

        # Convert to dB
        rms_db = 20 * np.log10(rms + 1e-10)

        # Calculate time per frame
        frame_duration = self.DEFAULT_HOP_LENGTH / sr

        # Calculate rate of change (linear regression slope)
        times = np.arange(len(rms_db)) * frame_duration
        if len(times) > 1:
            # Linear fit to get slope (dB/sec)
            slope, _ = np.polyfit(times, rms_db, 1)
            return float(slope)

        return 0.0

    def _extract_f0_librosa(
        self, audio: np.ndarray, sr: int, fmin: float = 75.0, fmax: float = 600.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract F0 using librosa's pyin algorithm (CPU fallback).

        Args:
            audio: Audio signal as numpy array
            sr: Sample rate in Hz
            fmin: Minimum F0 in Hz
            fmax: Maximum F0 in Hz

        Returns:
            Tuple of (f0_values, times)
        """
        f0, voiced_flag, voiced_probs = self.librosa.pyin(
            audio,
            fmin=fmin,
            fmax=fmax,
            sr=sr,
            frame_length=self.DEFAULT_FRAME_LENGTH,
            hop_length=self.DEFAULT_HOP_LENGTH,
        )

        # Calculate time array
        times = self.librosa.times_like(f0, sr=sr, hop_length=self.DEFAULT_HOP_LENGTH)

        return f0, times

    def extract_f0(
        self, audio: np.ndarray, sr: int, fmin: float = 75.0, fmax: float = 600.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract fundamental frequency (F0) with GPU acceleration.

        TASK-005: F0 extraction using GPU (torchcrepe) or CPU (librosa.pyin)
        SPEC-GPUAUDIO-001 Phase 3: GPU-accelerated F0 extraction

        Args:
            audio: Audio signal as numpy array
            sr: Sample rate in Hz
            fmin: Minimum F0 in Hz (default: 75)
            fmax: Maximum F0 in Hz (default: 600)

        Returns:
            Tuple of (f0_values, times)
            - f0_values: Array of F0 values (NaN for unvoiced frames)
            - times: Array of time values in seconds
        """
        # Try GPU extraction if enabled
        if self._use_gpu:
            try:
                f0, confidence = self.gpu_backend.extract_f0(audio, sr, fmin, fmax)

                # Calculate time array matching librosa's format
                times = self.librosa.times_like(f0, sr=sr, hop_length=self.DEFAULT_HOP_LENGTH)

                # Adjust times array length if needed
                if len(times) != len(f0):
                    times = np.linspace(0, len(audio) / sr, len(f0))

                return f0, times

            except Exception as e:
                logger.warning(f"GPU F0 extraction failed, falling back to CPU: {e}")

        # Fallback to CPU (librosa pyin)
        return self._extract_f0_librosa(audio, sr, fmin, fmax)

    def calculate_jitter(self, audio: np.ndarray, sr: int) -> float:
        """
        Calculate jitter (pitch perturbation) as percentage.

        TASK-006: Jitter calculation

        Args:
            audio: Audio signal as numpy array
            sr: Sample rate in Hz

        Returns:
            Jitter as percentage (0-10%)
        """
        try:
            # Extract F0
            f0, _ = self.extract_f0(audio, sr)

            # Filter valid (voiced) F0 values
            valid_f0 = f0[~np.isnan(f0)]

            if len(valid_f0) < 2:
                return 0.0

            # Calculate period (in samples) from F0
            periods = sr / valid_f0

            # Calculate absolute jitter (mean absolute difference between consecutive periods)
            period_diffs = np.abs(np.diff(periods))
            mean_period = np.mean(periods)

            if mean_period > 0:
                # Jitter as percentage of mean period
                jitter_percent = (np.mean(period_diffs) / mean_period) * 100
                # Clamp to valid range
                return float(min(10.0, max(0.0, jitter_percent)))

            return 0.0

        except Exception:
            return 0.0

    def calculate_pitch_stats(self, audio: np.ndarray, sr: int) -> Dict[str, float]:
        """
        Calculate pitch statistics.

        TASK-007: Pitch statistics calculation

        Args:
            audio: Audio signal as numpy array
            sr: Sample rate in Hz

        Returns:
            Dictionary with f0_mean_hz, f0_std_hz, f0_min_hz, f0_max_hz, f0_range_semitones
        """
        # Extract F0
        f0, _ = self.extract_f0(audio, sr)

        # Filter valid F0 values
        valid_f0 = f0[~np.isnan(f0)]

        if len(valid_f0) == 0:
            # Return default values for silent/unvoiced audio
            return {
                "f0_mean_hz": 150.0,  # Default to typical male voice
                "f0_std_hz": 0.0,
                "f0_min_hz": 150.0,
                "f0_max_hz": 150.0,
                "f0_range_semitones": 0.0,
            }

        f0_mean = float(np.mean(valid_f0))
        f0_std = float(np.std(valid_f0))
        f0_min = float(np.min(valid_f0))
        f0_max = float(np.max(valid_f0))

        # Clamp mean to valid range
        f0_mean = min(600.0, max(75.0, f0_mean))

        # Calculate range in semitones
        if f0_min > 0:
            f0_range_semitones = 12 * np.log2(f0_max / f0_min)
        else:
            f0_range_semitones = 0.0

        return {
            "f0_mean_hz": f0_mean,
            "f0_std_hz": f0_std,
            "f0_min_hz": f0_min,
            "f0_max_hz": f0_max,
            "f0_range_semitones": float(max(0.0, f0_range_semitones)),
        }

    def calculate_wpm(self, segments: List[Dict[str, Any]], total_duration: float) -> float:
        """
        Calculate words per minute from WhisperX segments.

        TASK-008: WPM calculation

        Args:
            segments: List of WhisperX segments with word-level data
            total_duration: Total audio duration in seconds

        Returns:
            Words per minute (clamped to 50-300 range)
        """
        if not segments or total_duration <= 0:
            return 50.0  # Minimum WPM

        # Count total words
        word_count = 0
        for segment in segments:
            words = segment.get("words", [])
            if words:
                word_count += len(words)
            else:
                # Fallback: count words in text
                text = segment.get("text", "")
                word_count += len(text.split())

        if word_count == 0:
            return 50.0

        # Calculate WPM
        minutes = total_duration / 60.0
        wpm = word_count / minutes if minutes > 0 else 50.0

        # Clamp to valid range
        return float(min(300.0, max(50.0, wpm)))

    def calculate_speech_silence_ratio(
        self, audio: np.ndarray, sr: int, top_db: int = 40
    ) -> Tuple[float, float]:
        """
        Calculate speech and silence ratios.

        TASK-009: Speech/silence ratio calculation using librosa.effects.split

        Args:
            audio: Audio signal as numpy array
            sr: Sample rate in Hz
            top_db: Threshold below peak for silence detection

        Returns:
            Tuple of (speech_ratio, silence_ratio)
        """
        if len(audio) == 0:
            return 0.0, 1.0

        total_samples = len(audio)

        # Use librosa to find non-silent intervals
        intervals = self.librosa.effects.split(audio, top_db=top_db)

        if len(intervals) == 0:
            return 0.0, 1.0

        # Calculate speech samples
        speech_samples = sum(end - start for start, end in intervals)

        speech_ratio = speech_samples / total_samples
        silence_ratio = 1.0 - speech_ratio

        # Ensure ratios are valid
        speech_ratio = min(1.0, max(0.0, speech_ratio))
        silence_ratio = min(1.0, max(0.0, silence_ratio))

        return float(speech_ratio), float(silence_ratio)

    def detect_pauses(
        self, audio: np.ndarray, sr: int, min_pause_duration: float = 0.3, top_db: int = 40
    ) -> List[PauseInfo]:
        """
        Detect pauses (silence gaps) in audio.

        TASK-010: Pause detection

        Args:
            audio: Audio signal as numpy array
            sr: Sample rate in Hz
            min_pause_duration: Minimum pause duration in seconds
            top_db: Threshold below peak for silence detection

        Returns:
            List of PauseInfo objects
        """
        if len(audio) == 0:
            return []

        # Find non-silent intervals
        intervals = self.librosa.effects.split(audio, top_db=top_db)

        if len(intervals) < 2:
            return []

        pauses = []
        min_samples = int(min_pause_duration * sr)

        # Find gaps between speech intervals
        for i in range(len(intervals) - 1):
            gap_start = intervals[i][1]
            gap_end = intervals[i + 1][0]
            gap_samples = gap_end - gap_start

            if gap_samples >= min_samples:
                start_time = gap_start / sr
                end_time = gap_end / sr
                duration = end_time - start_time

                pauses.append(
                    PauseInfo(start_time=start_time, end_time=end_time, duration=duration)
                )

        return pauses

    def _calculate_rms_batch(
        self, audio: np.ndarray, sr: int, window_samples: int, hop_samples: int
    ) -> np.ndarray:
        """
        Calculate RMS values for multiple windows efficiently using vectorized operations.

        Args:
            audio: Full audio signal
            sr: Sample rate
            window_samples: Number of samples per window
            hop_samples: Number of samples between windows

        Returns:
            Array of RMS values in dB for each window
        """
        # Calculate number of windows
        num_windows = (len(audio) - window_samples) // hop_samples + 1

        if num_windows <= 0:
            return np.array([])

        # Use stride tricks for efficient windowing (no copy)
        shape = (num_windows, window_samples)
        strides = (hop_samples * audio.strides[0], audio.strides[0])
        windows_view = np.lib.stride_tricks.as_strided(audio, shape=shape, strides=strides)

        # Calculate RMS for all windows at once (vectorized)
        rms_values = np.sqrt(np.mean(windows_view**2, axis=1))

        # Convert to dB
        rms_db = 20 * np.log10(rms_values + 1e-10)

        return rms_db

    def _extract_pitch_batch(self, windows: List[np.ndarray], sr: int) -> List[float]:
        """
        Extract mean pitch values from multiple audio windows.

        Uses GPU batch processing if available, otherwise falls back to sequential CPU.

        Args:
            windows: List of audio windows
            sr: Sample rate in Hz

        Returns:
            List of mean pitch values (NaN for windows without valid pitch)
        """
        if self._use_gpu:
            try:
                # Stack windows into a 2D array for batch processing
                audio_windows = np.stack(windows)
                f0_values, confidence_values = self.gpu_backend.extract_f0_batch(audio_windows, sr)
                return f0_values.tolist()
            except Exception as e:
                logger.warning(f"GPU batch F0 extraction failed, falling back to CPU: {e}")

        # Fallback to sequential CPU processing
        pitch_values = []
        for window in windows:
            f0, _ = self._extract_f0_librosa(window, sr)
            valid_f0 = f0[~np.isnan(f0)]
            pitch_values.append(np.mean(valid_f0) if len(valid_f0) > 0 else np.nan)

        return pitch_values

    def detect_emotional_escalation(
        self,
        audio: np.ndarray,
        sr: int,
        window_duration: float = 1.0,
        threshold_db: float = 3.0,
        threshold_pitch_percent: float = 10.0,
    ) -> List[EscalationZone]:
        """
        Detect emotional escalation zones based on volume and pitch changes.

        TASK-016: Emotional escalation detection
        SPEC-GPUAUDIO-001 Phase 3: GPU batch processing for F0 extraction

        Args:
            audio: Audio signal as numpy array
            sr: Sample rate in Hz
            window_duration: Analysis window duration in seconds
            threshold_db: Volume increase threshold in dB
            threshold_pitch_percent: Pitch increase threshold in percent

        Returns:
            List of EscalationZone objects
        """
        if len(audio) < sr * 2:  # Need at least 2 seconds
            return []

        escalation_zones = []
        window_samples = int(window_duration * sr)
        hop_samples = window_samples // 2

        # Calculate number of windows and times using vectorized operations
        num_windows = (len(audio) - window_samples) // hop_samples + 1

        if num_windows < 2:
            return []

        # Generate times array (vectorized)
        times = np.arange(num_windows) * hop_samples / sr

        # Batch RMS calculation using stride tricks (vectorized, no loop)
        rms_values = self._calculate_rms_batch(audio, sr, window_samples, hop_samples)

        if len(rms_values) < 2:
            return []

        # Collect windows for F0 batch extraction
        windows = []
        for i in range(num_windows):
            start = i * hop_samples
            end = start + window_samples
            windows.append(audio[start:end])

        # Batch F0 extraction using GPU if available
        pitch_values = self._extract_pitch_batch(windows, sr)

        if len(rms_values) < 2:
            return []

        pitch_values = np.array(pitch_values)

        # Detect escalation (increasing RMS and/or pitch)
        i = 0
        while i < len(rms_values) - 1:
            # Look for sustained increase
            start_idx = i
            volume_increase = 0.0
            pitch_increase = 0.0

            while i < len(rms_values) - 1:
                rms_diff = rms_values[i + 1] - rms_values[i]

                # Check pitch if valid
                pitch_diff_percent = 0.0
                if not np.isnan(pitch_values[i]) and not np.isnan(pitch_values[i + 1]):
                    if pitch_values[i] > 0:
                        pitch_diff_percent = (
                            (pitch_values[i + 1] - pitch_values[i]) / pitch_values[i]
                        ) * 100

                # Accumulate increases
                if rms_diff > 0:
                    volume_increase += rms_diff

                if pitch_diff_percent > 0:
                    pitch_increase += pitch_diff_percent

                # Check if this is an escalation
                if rms_diff < -1.0 and pitch_diff_percent < -5.0:
                    # Decreasing - end of potential escalation
                    break

                i += 1

            # Check if accumulated changes exceed threshold
            if volume_increase >= threshold_db or pitch_increase >= threshold_pitch_percent:
                # Calculate intensity score
                volume_score = min(1.0, volume_increase / 10.0)
                pitch_score = min(1.0, pitch_increase / 50.0)
                intensity_score = (volume_score + pitch_score) / 2.0

                if intensity_score > 0.2:  # Minimum threshold
                    escalation_zones.append(
                        EscalationZone(
                            start_time=float(times[start_idx]),
                            end_time=float(times[i] + window_duration),
                            intensity_score=float(intensity_score),
                            volume_increase_db=float(volume_increase),
                            pitch_increase_percent=float(pitch_increase),
                        )
                    )

            i += 1

        return escalation_zones

    def analyze_audio_features(
        self,
        audio: np.ndarray,
        sr: int,
        file_path: str,
        segments: Optional[List[Dict[str, Any]]] = None,
    ) -> AudioFeatureAnalysis:
        """
        Perform complete audio feature analysis.

        TASK-015: Complete analysis pipeline

        Args:
            audio: Audio signal as numpy array
            sr: Sample rate in Hz
            file_path: Path to the audio file
            segments: Optional WhisperX segments for WPM calculation

        Returns:
            AudioFeatureAnalysis with all feature data
        """
        # Import stress analysis service
        from voice_man.services.forensic.stress_analysis_service import StressAnalysisService

        duration = len(audio) / sr
        segments = segments or []

        # Volume features
        rms, rms_db = self.calculate_rms_amplitude(audio, sr)
        peak, peak_db = self.calculate_peak_amplitude(audio, sr)
        dynamic_range = self.calculate_dynamic_range(audio, sr)
        volume_change_rate = self.calculate_volume_change_rate(audio, sr)

        volume_features = VolumeFeatures(
            rms_amplitude=rms,
            rms_db=rms_db,
            peak_amplitude=peak,
            peak_db=peak_db,
            dynamic_range_db=dynamic_range,
            volume_change_rate_db_per_sec=volume_change_rate,
        )

        # Pitch features
        pitch_stats = self.calculate_pitch_stats(audio, sr)
        jitter = self.calculate_jitter(audio, sr)

        pitch_features = PitchFeatures(
            f0_mean_hz=pitch_stats["f0_mean_hz"],
            f0_std_hz=pitch_stats["f0_std_hz"],
            f0_min_hz=pitch_stats["f0_min_hz"],
            f0_max_hz=pitch_stats["f0_max_hz"],
            f0_range_semitones=pitch_stats["f0_range_semitones"],
            jitter_percent=jitter,
        )

        # Speech rate features
        wpm = self.calculate_wpm(segments, duration)
        speech_ratio, silence_ratio = self.calculate_speech_silence_ratio(audio, sr)
        pauses = self.detect_pauses(audio, sr)

        avg_pause_duration = sum(p.duration for p in pauses) / len(pauses) if pauses else 0.0

        speech_rate_features = SpeechRateFeatures(
            words_per_minute=wpm,
            speech_ratio=speech_ratio,
            silence_ratio=silence_ratio,
            pause_count=len(pauses),
            average_pause_duration=avg_pause_duration,
            pauses=pauses,
        )

        # Stress features
        stress_service = StressAnalysisService()
        stress_features = stress_service.analyze_stress(audio, sr, jitter_percent=jitter)

        # Escalation zones
        escalation_zones = self.detect_emotional_escalation(audio, sr)

        return AudioFeatureAnalysis(
            file_path=file_path,
            duration_seconds=duration,
            sample_rate=sr,
            volume_features=volume_features,
            pitch_features=pitch_features,
            speech_rate_features=speech_rate_features,
            stress_features=stress_features,
            escalation_zones=escalation_zones,
        )
