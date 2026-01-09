"""
Audio Features Models
SPEC-FORENSIC-001 TASK-001: Pydantic models for forensic audio analysis

These models define the data structures for audio feature extraction
used in voice evidence analysis.
"""

from typing import List, Literal
from pydantic import BaseModel, Field, field_validator, model_validator


class VolumeFeatures(BaseModel):
    """
    Volume-related audio features.

    Attributes:
        rms_amplitude: Root Mean Square amplitude (0.0-1.0)
        rms_db: RMS amplitude in decibels
        peak_amplitude: Peak amplitude (0.0-1.0)
        peak_db: Peak amplitude in decibels
        dynamic_range_db: Dynamic range in decibels (peak - RMS)
        volume_change_rate_db_per_sec: Rate of volume change in dB/second
    """

    rms_amplitude: float = Field(
        ..., ge=0.0, le=1.0, description="RMS amplitude normalized to 0-1 range"
    )
    rms_db: float = Field(..., description="RMS amplitude in decibels")
    peak_amplitude: float = Field(
        ..., ge=0.0, le=1.0, description="Peak amplitude normalized to 0-1 range"
    )
    peak_db: float = Field(..., description="Peak amplitude in decibels")
    dynamic_range_db: float = Field(
        ..., ge=0.0, description="Dynamic range in decibels (must be non-negative)"
    )
    volume_change_rate_db_per_sec: float = Field(
        ..., description="Rate of volume change in dB/second"
    )


class PitchFeatures(BaseModel):
    """
    Pitch-related audio features.

    Attributes:
        f0_mean_hz: Mean fundamental frequency in Hz (75-600Hz for human voice)
        f0_std_hz: Standard deviation of F0 in Hz
        f0_min_hz: Minimum F0 in Hz
        f0_max_hz: Maximum F0 in Hz
        f0_range_semitones: F0 range in semitones
        jitter_percent: Jitter (pitch perturbation) as percentage (0-10%)
    """

    f0_mean_hz: float = Field(
        ..., ge=75.0, le=600.0, description="Mean fundamental frequency in Hz (human voice range)"
    )
    f0_std_hz: float = Field(..., ge=0.0, description="Standard deviation of F0 in Hz")
    f0_min_hz: float = Field(..., ge=0.0, description="Minimum F0 in Hz")
    f0_max_hz: float = Field(..., ge=0.0, description="Maximum F0 in Hz")
    f0_range_semitones: float = Field(..., ge=0.0, description="F0 range in semitones")
    jitter_percent: float = Field(..., ge=0.0, le=10.0, description="Jitter as percentage (0-10%)")


class PauseInfo(BaseModel):
    """
    Information about a detected pause in speech.

    Attributes:
        start_time: Start time of the pause in seconds
        end_time: End time of the pause in seconds
        duration: Duration of the pause in seconds
    """

    start_time: float = Field(..., ge=0.0, description="Start time of pause in seconds")
    end_time: float = Field(..., ge=0.0, description="End time of pause in seconds")
    duration: float = Field(..., ge=0.0, description="Duration of pause in seconds")

    @model_validator(mode="after")
    def validate_time_order(self) -> "PauseInfo":
        """Validate that end_time is greater than start_time."""
        if self.end_time <= self.start_time:
            raise ValueError(
                f"end_time ({self.end_time}) must be greater than start_time ({self.start_time})"
            )
        return self

    @model_validator(mode="after")
    def validate_duration_consistency(self) -> "PauseInfo":
        """Validate that duration matches end_time - start_time."""
        expected_duration = self.end_time - self.start_time
        tolerance = 0.001  # 1ms tolerance for floating point comparison
        if abs(self.duration - expected_duration) > tolerance:
            raise ValueError(
                f"duration ({self.duration}) must match end_time - start_time ({expected_duration})"
            )
        return self


class SpeechRateFeatures(BaseModel):
    """
    Speech rate and timing features.

    Attributes:
        words_per_minute: Speaking rate in words per minute (50-300)
        speech_ratio: Ratio of speech time to total time (0-1)
        silence_ratio: Ratio of silence time to total time (0-1)
        pause_count: Number of detected pauses
        average_pause_duration: Average pause duration in seconds
        pauses: List of detected pause information
    """

    words_per_minute: float = Field(
        ..., ge=50.0, le=300.0, description="Speaking rate in WPM (reasonable human range)"
    )
    speech_ratio: float = Field(
        ..., ge=0.0, le=1.0, description="Ratio of speech time to total time"
    )
    silence_ratio: float = Field(
        ..., ge=0.0, le=1.0, description="Ratio of silence time to total time"
    )
    pause_count: int = Field(..., ge=0, description="Number of detected pauses")
    average_pause_duration: float = Field(
        ..., ge=0.0, description="Average pause duration in seconds"
    )
    pauses: List[PauseInfo] = Field(default_factory=list, description="List of detected pauses")

    @model_validator(mode="after")
    def validate_ratio_sum(self) -> "SpeechRateFeatures":
        """Validate that speech_ratio + silence_ratio approximately equals 1.0."""
        total = self.speech_ratio + self.silence_ratio
        tolerance = 0.05  # 5% tolerance
        if abs(total - 1.0) > tolerance:
            raise ValueError(
                f"speech_ratio ({self.speech_ratio}) + silence_ratio "
                f"({self.silence_ratio}) must sum to approximately 1.0, "
                f"got {total}"
            )
        return self


class StressFeatures(BaseModel):
    """
    Voice stress analysis features.

    Attributes:
        shimmer_percent: Shimmer (amplitude perturbation) as percentage (0-15%)
        hnr_db: Harmonics-to-Noise Ratio in decibels (0-40dB)
        formant_stability_score: Formant stability score (0-100)
        stress_index: Computed stress index (0-100)
        risk_level: Risk level classification (low/medium/high)
    """

    shimmer_percent: float = Field(
        ..., ge=0.0, le=15.0, description="Shimmer as percentage (0-15%)"
    )
    hnr_db: float = Field(
        ..., ge=0.0, le=40.0, description="Harmonics-to-Noise Ratio in dB (0-40dB)"
    )
    formant_stability_score: float = Field(
        ..., ge=0.0, le=100.0, description="Formant stability score (0-100)"
    )
    stress_index: float = Field(..., ge=0.0, le=100.0, description="Computed stress index (0-100)")
    risk_level: Literal["low", "medium", "high"] = Field(
        ..., description="Risk level classification"
    )


class EscalationZone(BaseModel):
    """
    Detected emotional escalation zone in audio.

    Attributes:
        start_time: Start time of escalation zone in seconds
        end_time: End time of escalation zone in seconds
        intensity_score: Intensity score of escalation (0-1)
        volume_increase_db: Volume increase in decibels
        pitch_increase_percent: Pitch increase as percentage
    """

    start_time: float = Field(..., ge=0.0, description="Start time of escalation zone in seconds")
    end_time: float = Field(..., ge=0.0, description="End time of escalation zone in seconds")
    intensity_score: float = Field(
        ..., ge=0.0, le=1.0, description="Intensity score of escalation (0-1)"
    )
    volume_increase_db: float = Field(..., description="Volume increase in decibels")
    pitch_increase_percent: float = Field(..., description="Pitch increase as percentage")

    @model_validator(mode="after")
    def validate_time_order(self) -> "EscalationZone":
        """Validate that end_time is greater than start_time."""
        if self.end_time <= self.start_time:
            raise ValueError(
                f"end_time ({self.end_time}) must be greater than start_time ({self.start_time})"
            )
        return self


class AudioFeatureAnalysis(BaseModel):
    """
    Complete audio feature analysis result.

    Attributes:
        file_path: Path to the analyzed audio file
        duration_seconds: Total duration in seconds
        sample_rate: Audio sample rate in Hz
        volume_features: Volume analysis results
        pitch_features: Pitch analysis results
        speech_rate_features: Speech rate analysis results
        stress_features: Stress analysis results
        escalation_zones: List of detected emotional escalation zones
    """

    file_path: str = Field(..., description="Path to the analyzed audio file")
    duration_seconds: float = Field(..., gt=0.0, description="Total duration in seconds")
    sample_rate: int = Field(..., gt=0, description="Audio sample rate in Hz")
    volume_features: VolumeFeatures = Field(..., description="Volume analysis results")
    pitch_features: PitchFeatures = Field(..., description="Pitch analysis results")
    speech_rate_features: SpeechRateFeatures = Field(
        ..., description="Speech rate analysis results"
    )
    stress_features: StressFeatures = Field(..., description="Stress analysis results")
    escalation_zones: List[EscalationZone] = Field(
        default_factory=list, description="List of detected emotional escalation zones"
    )
