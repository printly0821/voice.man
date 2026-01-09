"""
Tests for Audio Features Models
SPEC-FORENSIC-001 TASK-001: Data model tests for forensic audio analysis

TDD RED Phase: These tests define the expected behavior of the data models.
"""

import pytest
from pydantic import ValidationError


class TestVolumeFeatures:
    """Tests for VolumeFeatures model."""

    def test_valid_volume_features(self):
        """Test creating VolumeFeatures with valid data."""
        from voice_man.models.forensic.audio_features import VolumeFeatures

        features = VolumeFeatures(
            rms_amplitude=0.5,
            rms_db=-6.0,
            peak_amplitude=0.9,
            peak_db=-0.9,
            dynamic_range_db=20.0,
            volume_change_rate_db_per_sec=5.0,
        )

        assert features.rms_amplitude == 0.5
        assert features.rms_db == -6.0
        assert features.peak_amplitude == 0.9
        assert features.peak_db == -0.9
        assert features.dynamic_range_db == 20.0
        assert features.volume_change_rate_db_per_sec == 5.0

    def test_rms_amplitude_range_valid(self):
        """Test RMS amplitude accepts valid range 0-1."""
        from voice_man.models.forensic.audio_features import VolumeFeatures

        # Minimum valid
        features = VolumeFeatures(
            rms_amplitude=0.0,
            rms_db=-100.0,
            peak_amplitude=0.0,
            peak_db=-100.0,
            dynamic_range_db=0.0,
            volume_change_rate_db_per_sec=0.0,
        )
        assert features.rms_amplitude == 0.0

        # Maximum valid
        features = VolumeFeatures(
            rms_amplitude=1.0,
            rms_db=0.0,
            peak_amplitude=1.0,
            peak_db=0.0,
            dynamic_range_db=0.0,
            volume_change_rate_db_per_sec=0.0,
        )
        assert features.rms_amplitude == 1.0

    def test_rms_amplitude_out_of_range(self):
        """Test RMS amplitude rejects values outside 0-1 range."""
        from voice_man.models.forensic.audio_features import VolumeFeatures

        with pytest.raises(ValidationError):
            VolumeFeatures(
                rms_amplitude=-0.1,
                rms_db=-6.0,
                peak_amplitude=0.9,
                peak_db=-0.9,
                dynamic_range_db=20.0,
                volume_change_rate_db_per_sec=5.0,
            )

        with pytest.raises(ValidationError):
            VolumeFeatures(
                rms_amplitude=1.1,
                rms_db=-6.0,
                peak_amplitude=0.9,
                peak_db=-0.9,
                dynamic_range_db=20.0,
                volume_change_rate_db_per_sec=5.0,
            )

    def test_peak_amplitude_range_valid(self):
        """Test peak amplitude accepts valid range 0-1."""
        from voice_man.models.forensic.audio_features import VolumeFeatures

        features = VolumeFeatures(
            rms_amplitude=0.5,
            rms_db=-6.0,
            peak_amplitude=0.0,
            peak_db=-100.0,
            dynamic_range_db=20.0,
            volume_change_rate_db_per_sec=5.0,
        )
        assert features.peak_amplitude == 0.0

    def test_dynamic_range_non_negative(self):
        """Test dynamic range must be non-negative."""
        from voice_man.models.forensic.audio_features import VolumeFeatures

        with pytest.raises(ValidationError):
            VolumeFeatures(
                rms_amplitude=0.5,
                rms_db=-6.0,
                peak_amplitude=0.9,
                peak_db=-0.9,
                dynamic_range_db=-5.0,
                volume_change_rate_db_per_sec=5.0,
            )


class TestPitchFeatures:
    """Tests for PitchFeatures model."""

    def test_valid_pitch_features(self):
        """Test creating PitchFeatures with valid data."""
        from voice_man.models.forensic.audio_features import PitchFeatures

        features = PitchFeatures(
            f0_mean_hz=150.0,
            f0_std_hz=25.0,
            f0_min_hz=80.0,
            f0_max_hz=300.0,
            f0_range_semitones=12.0,
            jitter_percent=1.5,
        )

        assert features.f0_mean_hz == 150.0
        assert features.f0_std_hz == 25.0
        assert features.f0_min_hz == 80.0
        assert features.f0_max_hz == 300.0
        assert features.f0_range_semitones == 12.0
        assert features.jitter_percent == 1.5

    def test_f0_mean_valid_range(self):
        """Test F0 mean accepts valid human voice range 75-600Hz."""
        from voice_man.models.forensic.audio_features import PitchFeatures

        # Minimum valid
        features = PitchFeatures(
            f0_mean_hz=75.0,
            f0_std_hz=10.0,
            f0_min_hz=70.0,
            f0_max_hz=100.0,
            f0_range_semitones=5.0,
            jitter_percent=1.0,
        )
        assert features.f0_mean_hz == 75.0

        # Maximum valid
        features = PitchFeatures(
            f0_mean_hz=600.0,
            f0_std_hz=50.0,
            f0_min_hz=400.0,
            f0_max_hz=650.0,
            f0_range_semitones=8.0,
            jitter_percent=1.0,
        )
        assert features.f0_mean_hz == 600.0

    def test_f0_mean_out_of_range(self):
        """Test F0 mean rejects values outside 75-600Hz range."""
        from voice_man.models.forensic.audio_features import PitchFeatures

        with pytest.raises(ValidationError):
            PitchFeatures(
                f0_mean_hz=50.0,  # Too low
                f0_std_hz=10.0,
                f0_min_hz=40.0,
                f0_max_hz=80.0,
                f0_range_semitones=5.0,
                jitter_percent=1.0,
            )

        with pytest.raises(ValidationError):
            PitchFeatures(
                f0_mean_hz=700.0,  # Too high
                f0_std_hz=50.0,
                f0_min_hz=600.0,
                f0_max_hz=800.0,
                f0_range_semitones=5.0,
                jitter_percent=1.0,
            )

    def test_jitter_valid_range(self):
        """Test jitter accepts valid range 0-10%."""
        from voice_man.models.forensic.audio_features import PitchFeatures

        features = PitchFeatures(
            f0_mean_hz=150.0,
            f0_std_hz=25.0,
            f0_min_hz=80.0,
            f0_max_hz=300.0,
            f0_range_semitones=12.0,
            jitter_percent=0.0,
        )
        assert features.jitter_percent == 0.0

        features = PitchFeatures(
            f0_mean_hz=150.0,
            f0_std_hz=25.0,
            f0_min_hz=80.0,
            f0_max_hz=300.0,
            f0_range_semitones=12.0,
            jitter_percent=10.0,
        )
        assert features.jitter_percent == 10.0

    def test_jitter_out_of_range(self):
        """Test jitter rejects values outside 0-10% range."""
        from voice_man.models.forensic.audio_features import PitchFeatures

        with pytest.raises(ValidationError):
            PitchFeatures(
                f0_mean_hz=150.0,
                f0_std_hz=25.0,
                f0_min_hz=80.0,
                f0_max_hz=300.0,
                f0_range_semitones=12.0,
                jitter_percent=-1.0,
            )

        with pytest.raises(ValidationError):
            PitchFeatures(
                f0_mean_hz=150.0,
                f0_std_hz=25.0,
                f0_min_hz=80.0,
                f0_max_hz=300.0,
                f0_range_semitones=12.0,
                jitter_percent=15.0,
            )


class TestSpeechRateFeatures:
    """Tests for SpeechRateFeatures model."""

    def test_valid_speech_rate_features(self):
        """Test creating SpeechRateFeatures with valid data."""
        from voice_man.models.forensic.audio_features import SpeechRateFeatures, PauseInfo

        pause = PauseInfo(start_time=1.0, end_time=1.5, duration=0.5)

        features = SpeechRateFeatures(
            words_per_minute=120.0,
            speech_ratio=0.7,
            silence_ratio=0.3,
            pause_count=5,
            average_pause_duration=0.5,
            pauses=[pause],
        )

        assert features.words_per_minute == 120.0
        assert features.speech_ratio == 0.7
        assert features.silence_ratio == 0.3
        assert features.pause_count == 5
        assert features.average_pause_duration == 0.5
        assert len(features.pauses) == 1

    def test_wpm_valid_range(self):
        """Test WPM accepts valid range 50-300."""
        from voice_man.models.forensic.audio_features import SpeechRateFeatures

        features = SpeechRateFeatures(
            words_per_minute=50.0,
            speech_ratio=0.5,
            silence_ratio=0.5,
            pause_count=0,
            average_pause_duration=0.0,
            pauses=[],
        )
        assert features.words_per_minute == 50.0

        features = SpeechRateFeatures(
            words_per_minute=300.0,
            speech_ratio=0.9,
            silence_ratio=0.1,
            pause_count=0,
            average_pause_duration=0.0,
            pauses=[],
        )
        assert features.words_per_minute == 300.0

    def test_wpm_out_of_range(self):
        """Test WPM rejects values outside 50-300 range."""
        from voice_man.models.forensic.audio_features import SpeechRateFeatures

        with pytest.raises(ValidationError):
            SpeechRateFeatures(
                words_per_minute=30.0,  # Too low
                speech_ratio=0.5,
                silence_ratio=0.5,
                pause_count=0,
                average_pause_duration=0.0,
                pauses=[],
            )

        with pytest.raises(ValidationError):
            SpeechRateFeatures(
                words_per_minute=400.0,  # Too high
                speech_ratio=0.5,
                silence_ratio=0.5,
                pause_count=0,
                average_pause_duration=0.0,
                pauses=[],
            )

    def test_speech_silence_ratio_sum(self):
        """Test speech and silence ratios must sum to approximately 1.0."""
        from voice_man.models.forensic.audio_features import SpeechRateFeatures

        # Valid sum
        features = SpeechRateFeatures(
            words_per_minute=120.0,
            speech_ratio=0.6,
            silence_ratio=0.4,
            pause_count=0,
            average_pause_duration=0.0,
            pauses=[],
        )
        assert abs(features.speech_ratio + features.silence_ratio - 1.0) < 0.05

    def test_speech_silence_ratio_invalid_sum(self):
        """Test speech and silence ratios reject invalid sum."""
        from voice_man.models.forensic.audio_features import SpeechRateFeatures

        with pytest.raises(ValidationError):
            SpeechRateFeatures(
                words_per_minute=120.0,
                speech_ratio=0.8,
                silence_ratio=0.5,  # Sum > 1.0
                pause_count=0,
                average_pause_duration=0.0,
                pauses=[],
            )


class TestPauseInfo:
    """Tests for PauseInfo model."""

    def test_valid_pause_info(self):
        """Test creating PauseInfo with valid data."""
        from voice_man.models.forensic.audio_features import PauseInfo

        pause = PauseInfo(start_time=1.0, end_time=2.5, duration=1.5)

        assert pause.start_time == 1.0
        assert pause.end_time == 2.5
        assert pause.duration == 1.5

    def test_pause_time_validation(self):
        """Test end_time must be greater than start_time."""
        from voice_man.models.forensic.audio_features import PauseInfo

        with pytest.raises(ValidationError):
            PauseInfo(start_time=2.0, end_time=1.0, duration=1.0)

    def test_pause_duration_consistency(self):
        """Test duration must match end_time - start_time."""
        from voice_man.models.forensic.audio_features import PauseInfo

        with pytest.raises(ValidationError):
            PauseInfo(start_time=1.0, end_time=2.0, duration=0.5)  # Should be 1.0


class TestStressFeatures:
    """Tests for StressFeatures model."""

    def test_valid_stress_features(self):
        """Test creating StressFeatures with valid data."""
        from voice_man.models.forensic.audio_features import StressFeatures

        features = StressFeatures(
            shimmer_percent=3.0,
            hnr_db=15.0,
            formant_stability_score=75.0,
            stress_index=45.0,
            risk_level="medium",
        )

        assert features.shimmer_percent == 3.0
        assert features.hnr_db == 15.0
        assert features.formant_stability_score == 75.0
        assert features.stress_index == 45.0
        assert features.risk_level == "medium"

    def test_shimmer_valid_range(self):
        """Test shimmer accepts valid range 0-15%."""
        from voice_man.models.forensic.audio_features import StressFeatures

        features = StressFeatures(
            shimmer_percent=0.0,
            hnr_db=15.0,
            formant_stability_score=75.0,
            stress_index=20.0,
            risk_level="low",
        )
        assert features.shimmer_percent == 0.0

        features = StressFeatures(
            shimmer_percent=15.0,
            hnr_db=15.0,
            formant_stability_score=75.0,
            stress_index=80.0,
            risk_level="high",
        )
        assert features.shimmer_percent == 15.0

    def test_shimmer_out_of_range(self):
        """Test shimmer rejects values outside 0-15% range."""
        from voice_man.models.forensic.audio_features import StressFeatures

        with pytest.raises(ValidationError):
            StressFeatures(
                shimmer_percent=-1.0,
                hnr_db=15.0,
                formant_stability_score=75.0,
                stress_index=45.0,
                risk_level="medium",
            )

        with pytest.raises(ValidationError):
            StressFeatures(
                shimmer_percent=20.0,
                hnr_db=15.0,
                formant_stability_score=75.0,
                stress_index=45.0,
                risk_level="medium",
            )

    def test_hnr_valid_range(self):
        """Test HNR accepts valid range 0-40dB."""
        from voice_man.models.forensic.audio_features import StressFeatures

        features = StressFeatures(
            shimmer_percent=3.0,
            hnr_db=0.0,
            formant_stability_score=75.0,
            stress_index=45.0,
            risk_level="medium",
        )
        assert features.hnr_db == 0.0

        features = StressFeatures(
            shimmer_percent=3.0,
            hnr_db=40.0,
            formant_stability_score=75.0,
            stress_index=45.0,
            risk_level="medium",
        )
        assert features.hnr_db == 40.0

    def test_hnr_out_of_range(self):
        """Test HNR rejects values outside 0-40dB range."""
        from voice_man.models.forensic.audio_features import StressFeatures

        with pytest.raises(ValidationError):
            StressFeatures(
                shimmer_percent=3.0,
                hnr_db=-5.0,
                formant_stability_score=75.0,
                stress_index=45.0,
                risk_level="medium",
            )

        with pytest.raises(ValidationError):
            StressFeatures(
                shimmer_percent=3.0,
                hnr_db=50.0,
                formant_stability_score=75.0,
                stress_index=45.0,
                risk_level="medium",
            )

    def test_formant_stability_valid_range(self):
        """Test formant stability score accepts valid range 0-100."""
        from voice_man.models.forensic.audio_features import StressFeatures

        features = StressFeatures(
            shimmer_percent=3.0,
            hnr_db=15.0,
            formant_stability_score=0.0,
            stress_index=45.0,
            risk_level="medium",
        )
        assert features.formant_stability_score == 0.0

        features = StressFeatures(
            shimmer_percent=3.0,
            hnr_db=15.0,
            formant_stability_score=100.0,
            stress_index=45.0,
            risk_level="medium",
        )
        assert features.formant_stability_score == 100.0

    def test_stress_index_valid_range(self):
        """Test stress index accepts valid range 0-100."""
        from voice_man.models.forensic.audio_features import StressFeatures

        features = StressFeatures(
            shimmer_percent=3.0,
            hnr_db=15.0,
            formant_stability_score=75.0,
            stress_index=0.0,
            risk_level="low",
        )
        assert features.stress_index == 0.0

        features = StressFeatures(
            shimmer_percent=3.0,
            hnr_db=15.0,
            formant_stability_score=75.0,
            stress_index=100.0,
            risk_level="high",
        )
        assert features.stress_index == 100.0

    def test_risk_level_valid_values(self):
        """Test risk level accepts valid values."""
        from voice_man.models.forensic.audio_features import StressFeatures

        for risk_level in ["low", "medium", "high"]:
            features = StressFeatures(
                shimmer_percent=3.0,
                hnr_db=15.0,
                formant_stability_score=75.0,
                stress_index=45.0,
                risk_level=risk_level,
            )
            assert features.risk_level == risk_level

    def test_risk_level_invalid_value(self):
        """Test risk level rejects invalid values."""
        from voice_man.models.forensic.audio_features import StressFeatures

        with pytest.raises(ValidationError):
            StressFeatures(
                shimmer_percent=3.0,
                hnr_db=15.0,
                formant_stability_score=75.0,
                stress_index=45.0,
                risk_level="invalid",
            )


class TestEscalationZone:
    """Tests for EscalationZone model."""

    def test_valid_escalation_zone(self):
        """Test creating EscalationZone with valid data."""
        from voice_man.models.forensic.audio_features import EscalationZone

        zone = EscalationZone(
            start_time=10.0,
            end_time=15.0,
            intensity_score=0.8,
            volume_increase_db=6.0,
            pitch_increase_percent=25.0,
        )

        assert zone.start_time == 10.0
        assert zone.end_time == 15.0
        assert zone.intensity_score == 0.8
        assert zone.volume_increase_db == 6.0
        assert zone.pitch_increase_percent == 25.0

    def test_escalation_time_validation(self):
        """Test end_time must be greater than start_time."""
        from voice_man.models.forensic.audio_features import EscalationZone

        with pytest.raises(ValidationError):
            EscalationZone(
                start_time=15.0,
                end_time=10.0,
                intensity_score=0.8,
                volume_increase_db=6.0,
                pitch_increase_percent=25.0,
            )

    def test_intensity_score_valid_range(self):
        """Test intensity score accepts valid range 0-1."""
        from voice_man.models.forensic.audio_features import EscalationZone

        zone = EscalationZone(
            start_time=10.0,
            end_time=15.0,
            intensity_score=0.0,
            volume_increase_db=0.0,
            pitch_increase_percent=0.0,
        )
        assert zone.intensity_score == 0.0

        zone = EscalationZone(
            start_time=10.0,
            end_time=15.0,
            intensity_score=1.0,
            volume_increase_db=10.0,
            pitch_increase_percent=50.0,
        )
        assert zone.intensity_score == 1.0


class TestAudioFeatureAnalysis:
    """Tests for AudioFeatureAnalysis model."""

    def test_valid_audio_feature_analysis(self):
        """Test creating AudioFeatureAnalysis with valid data."""
        from voice_man.models.forensic.audio_features import (
            AudioFeatureAnalysis,
            VolumeFeatures,
            PitchFeatures,
            SpeechRateFeatures,
            StressFeatures,
            EscalationZone,
        )

        volume = VolumeFeatures(
            rms_amplitude=0.5,
            rms_db=-6.0,
            peak_amplitude=0.9,
            peak_db=-0.9,
            dynamic_range_db=20.0,
            volume_change_rate_db_per_sec=5.0,
        )

        pitch = PitchFeatures(
            f0_mean_hz=150.0,
            f0_std_hz=25.0,
            f0_min_hz=80.0,
            f0_max_hz=300.0,
            f0_range_semitones=12.0,
            jitter_percent=1.5,
        )

        speech_rate = SpeechRateFeatures(
            words_per_minute=120.0,
            speech_ratio=0.7,
            silence_ratio=0.3,
            pause_count=5,
            average_pause_duration=0.5,
            pauses=[],
        )

        stress = StressFeatures(
            shimmer_percent=3.0,
            hnr_db=15.0,
            formant_stability_score=75.0,
            stress_index=45.0,
            risk_level="medium",
        )

        escalation = EscalationZone(
            start_time=10.0,
            end_time=15.0,
            intensity_score=0.8,
            volume_increase_db=6.0,
            pitch_increase_percent=25.0,
        )

        analysis = AudioFeatureAnalysis(
            file_path="/path/to/audio.wav",
            duration_seconds=60.0,
            sample_rate=16000,
            volume_features=volume,
            pitch_features=pitch,
            speech_rate_features=speech_rate,
            stress_features=stress,
            escalation_zones=[escalation],
        )

        assert analysis.file_path == "/path/to/audio.wav"
        assert analysis.duration_seconds == 60.0
        assert analysis.sample_rate == 16000
        assert analysis.volume_features == volume
        assert analysis.pitch_features == pitch
        assert analysis.speech_rate_features == speech_rate
        assert analysis.stress_features == stress
        assert len(analysis.escalation_zones) == 1

    def test_audio_feature_analysis_optional_escalation_zones(self):
        """Test AudioFeatureAnalysis with empty escalation zones."""
        from voice_man.models.forensic.audio_features import (
            AudioFeatureAnalysis,
            VolumeFeatures,
            PitchFeatures,
            SpeechRateFeatures,
            StressFeatures,
        )

        volume = VolumeFeatures(
            rms_amplitude=0.5,
            rms_db=-6.0,
            peak_amplitude=0.9,
            peak_db=-0.9,
            dynamic_range_db=20.0,
            volume_change_rate_db_per_sec=5.0,
        )

        pitch = PitchFeatures(
            f0_mean_hz=150.0,
            f0_std_hz=25.0,
            f0_min_hz=80.0,
            f0_max_hz=300.0,
            f0_range_semitones=12.0,
            jitter_percent=1.5,
        )

        speech_rate = SpeechRateFeatures(
            words_per_minute=120.0,
            speech_ratio=0.7,
            silence_ratio=0.3,
            pause_count=0,
            average_pause_duration=0.0,
            pauses=[],
        )

        stress = StressFeatures(
            shimmer_percent=3.0,
            hnr_db=15.0,
            formant_stability_score=75.0,
            stress_index=45.0,
            risk_level="medium",
        )

        analysis = AudioFeatureAnalysis(
            file_path="/path/to/audio.wav",
            duration_seconds=60.0,
            sample_rate=16000,
            volume_features=volume,
            pitch_features=pitch,
            speech_rate_features=speech_rate,
            stress_features=stress,
            escalation_zones=[],
        )

        assert len(analysis.escalation_zones) == 0

    def test_audio_feature_analysis_sample_rate_validation(self):
        """Test sample rate must be positive."""
        from voice_man.models.forensic.audio_features import (
            AudioFeatureAnalysis,
            VolumeFeatures,
            PitchFeatures,
            SpeechRateFeatures,
            StressFeatures,
        )

        volume = VolumeFeatures(
            rms_amplitude=0.5,
            rms_db=-6.0,
            peak_amplitude=0.9,
            peak_db=-0.9,
            dynamic_range_db=20.0,
            volume_change_rate_db_per_sec=5.0,
        )

        pitch = PitchFeatures(
            f0_mean_hz=150.0,
            f0_std_hz=25.0,
            f0_min_hz=80.0,
            f0_max_hz=300.0,
            f0_range_semitones=12.0,
            jitter_percent=1.5,
        )

        speech_rate = SpeechRateFeatures(
            words_per_minute=120.0,
            speech_ratio=0.7,
            silence_ratio=0.3,
            pause_count=0,
            average_pause_duration=0.0,
            pauses=[],
        )

        stress = StressFeatures(
            shimmer_percent=3.0,
            hnr_db=15.0,
            formant_stability_score=75.0,
            stress_index=45.0,
            risk_level="medium",
        )

        with pytest.raises(ValidationError):
            AudioFeatureAnalysis(
                file_path="/path/to/audio.wav",
                duration_seconds=60.0,
                sample_rate=0,  # Invalid
                volume_features=volume,
                pitch_features=pitch,
                speech_rate_features=speech_rate,
                stress_features=stress,
                escalation_zones=[],
            )
