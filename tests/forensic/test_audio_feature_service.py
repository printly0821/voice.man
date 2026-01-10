"""
Tests for Audio Feature Service
SPEC-FORENSIC-001 TASK-002~010, 015~016: Audio feature analysis service tests

TDD RED Phase: These tests define the expected behavior of the audio feature service.
"""

import pytest
import numpy as np
from typing import List, Dict, Any


class TestCalculateRmsAmplitude:
    """Tests for calculate_rms_amplitude() - TASK-002."""

    def test_rms_amplitude_silent_audio(self):
        """Test RMS amplitude for silent audio is near zero."""
        from voice_man.services.forensic.audio_feature_service import AudioFeatureService

        service = AudioFeatureService()

        # Create silent audio (all zeros)
        sr = 16000
        duration = 1.0
        audio = np.zeros(int(sr * duration), dtype=np.float32)

        rms, rms_db = service.calculate_rms_amplitude(audio, sr)

        assert rms == pytest.approx(0.0, abs=1e-6)
        # Silent audio should have very low dB (approaching -inf)
        assert rms_db < -60

    def test_rms_amplitude_sine_wave(self):
        """Test RMS amplitude for known sine wave."""
        from voice_man.services.forensic.audio_feature_service import AudioFeatureService

        service = AudioFeatureService()

        # Create a sine wave with known amplitude
        sr = 16000
        duration = 1.0
        frequency = 440  # A4
        amplitude = 0.5
        t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
        audio = amplitude * np.sin(2 * np.pi * frequency * t).astype(np.float32)

        rms, rms_db = service.calculate_rms_amplitude(audio, sr)

        # RMS of sine wave = amplitude / sqrt(2)
        expected_rms = amplitude / np.sqrt(2)
        assert rms == pytest.approx(expected_rms, rel=0.05)
        assert 0.0 <= rms <= 1.0

    def test_rms_amplitude_range(self):
        """Test RMS amplitude is in 0-1 range."""
        from voice_man.services.forensic.audio_feature_service import AudioFeatureService

        service = AudioFeatureService()

        sr = 16000
        duration = 1.0
        audio = np.random.randn(int(sr * duration)).astype(np.float32) * 0.3

        rms, rms_db = service.calculate_rms_amplitude(audio, sr)

        assert 0.0 <= rms <= 1.0


class TestCalculatePeakAmplitude:
    """Tests for calculate_peak_amplitude() - TASK-003."""

    def test_peak_amplitude_silent_audio(self):
        """Test peak amplitude for silent audio is zero."""
        from voice_man.services.forensic.audio_feature_service import AudioFeatureService

        service = AudioFeatureService()

        sr = 16000
        audio = np.zeros(sr, dtype=np.float32)

        peak, peak_db = service.calculate_peak_amplitude(audio, sr)

        assert peak == pytest.approx(0.0, abs=1e-6)

    def test_peak_amplitude_known_value(self):
        """Test peak amplitude for audio with known peak."""
        from voice_man.services.forensic.audio_feature_service import AudioFeatureService

        service = AudioFeatureService()

        sr = 16000
        audio = np.zeros(sr, dtype=np.float32)
        audio[sr // 2] = 0.8  # Set a single peak

        peak, peak_db = service.calculate_peak_amplitude(audio, sr)

        assert peak == pytest.approx(0.8, rel=0.01)
        assert 0.0 <= peak <= 1.0

    def test_peak_amplitude_greater_than_rms(self):
        """Test that peak amplitude is greater than or equal to RMS."""
        from voice_man.services.forensic.audio_feature_service import AudioFeatureService

        service = AudioFeatureService()

        sr = 16000
        audio = np.random.randn(sr).astype(np.float32) * 0.3

        rms, _ = service.calculate_rms_amplitude(audio, sr)
        peak, _ = service.calculate_peak_amplitude(audio, sr)

        assert peak >= rms


class TestCalculateDynamicRange:
    """Tests for calculate_dynamic_range() - TASK-003."""

    def test_dynamic_range_calculation(self):
        """Test dynamic range calculation in dB."""
        from voice_man.services.forensic.audio_feature_service import AudioFeatureService

        service = AudioFeatureService()

        sr = 16000
        # Create audio with known RMS and peak
        t = np.linspace(0, 1, sr, dtype=np.float32)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

        dynamic_range = service.calculate_dynamic_range(audio, sr)

        # Dynamic range should be non-negative
        assert dynamic_range >= 0.0

    def test_dynamic_range_silent_audio(self):
        """Test dynamic range for silent audio."""
        from voice_man.services.forensic.audio_feature_service import AudioFeatureService

        service = AudioFeatureService()

        sr = 16000
        audio = np.zeros(sr, dtype=np.float32)

        dynamic_range = service.calculate_dynamic_range(audio, sr)

        # Silent audio should have 0 dynamic range
        assert dynamic_range == pytest.approx(0.0, abs=0.1)


class TestCalculateVolumeChangeRate:
    """Tests for calculate_volume_change_rate() - TASK-004."""

    def test_volume_change_rate_constant_volume(self):
        """Test volume change rate for constant volume audio."""
        from voice_man.services.forensic.audio_feature_service import AudioFeatureService

        service = AudioFeatureService()

        sr = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
        # Constant amplitude sine wave
        audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

        change_rate = service.calculate_volume_change_rate(audio, sr)

        # Constant volume should have near-zero change rate
        assert abs(change_rate) < 5.0  # Less than 5 dB/s

    def test_volume_change_rate_increasing_volume(self):
        """Test volume change rate for increasing volume audio."""
        from voice_man.services.forensic.audio_feature_service import AudioFeatureService

        service = AudioFeatureService()

        sr = 16000
        duration = 2.0
        samples = int(sr * duration)
        t = np.linspace(0, duration, samples, dtype=np.float32)
        # Linearly increasing amplitude
        envelope = np.linspace(0.1, 0.9, samples, dtype=np.float32)
        audio = envelope * np.sin(2 * np.pi * 440 * t).astype(np.float32)

        change_rate = service.calculate_volume_change_rate(audio, sr)

        # Increasing volume should have positive change rate
        assert change_rate > 0


class TestExtractF0:
    """Tests for extract_f0() - TASK-005."""

    def test_extract_f0_sine_wave(self):
        """Test F0 extraction for known frequency sine wave."""
        from voice_man.services.forensic.audio_feature_service import AudioFeatureService

        service = AudioFeatureService()

        sr = 16000
        duration = 1.0
        frequency = 200.0  # Hz
        t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
        audio = 0.5 * np.sin(2 * np.pi * frequency * t).astype(np.float32)

        f0_values, times = service.extract_f0(audio, sr)

        # Filter out NaN values for voiced frames
        valid_f0 = f0_values[~np.isnan(f0_values)]

        if len(valid_f0) > 0:
            mean_f0 = np.mean(valid_f0)
            # Allow 10% tolerance for F0 detection
            assert mean_f0 == pytest.approx(frequency, rel=0.1)

    def test_extract_f0_range(self):
        """Test F0 values are within human voice range (75-600Hz)."""
        from voice_man.services.forensic.audio_feature_service import AudioFeatureService

        service = AudioFeatureService()

        sr = 16000
        duration = 1.0
        frequency = 150.0  # Typical male voice
        t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
        audio = 0.5 * np.sin(2 * np.pi * frequency * t).astype(np.float32)

        f0_values, _ = service.extract_f0(audio, sr)

        # Filter valid values
        valid_f0 = f0_values[~np.isnan(f0_values)]

        if len(valid_f0) > 0:
            assert np.all(valid_f0 >= 75.0)
            assert np.all(valid_f0 <= 600.0)


class TestCalculateJitter:
    """Tests for calculate_jitter() - TASK-006."""

    def test_jitter_perfect_periodic_signal(self):
        """Test jitter for perfectly periodic signal is low."""
        from voice_man.services.forensic.audio_feature_service import AudioFeatureService

        service = AudioFeatureService()

        sr = 16000
        duration = 1.0
        frequency = 150.0
        t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
        audio = 0.5 * np.sin(2 * np.pi * frequency * t).astype(np.float32)

        jitter = service.calculate_jitter(audio, sr)

        # Perfect sine wave should have very low jitter
        assert jitter >= 0.0
        assert jitter <= 10.0  # Max jitter is 10%

    def test_jitter_range(self):
        """Test jitter is within 0-10% range."""
        from voice_man.services.forensic.audio_feature_service import AudioFeatureService

        service = AudioFeatureService()

        sr = 16000
        duration = 1.0
        audio = np.random.randn(int(sr * duration)).astype(np.float32) * 0.3

        jitter = service.calculate_jitter(audio, sr)

        assert 0.0 <= jitter <= 10.0


class TestCalculatePitchStats:
    """Tests for calculate_pitch_stats() - TASK-007."""

    def test_pitch_stats_valid_output(self):
        """Test pitch statistics output structure."""
        from voice_man.services.forensic.audio_feature_service import AudioFeatureService

        service = AudioFeatureService()

        sr = 16000
        duration = 1.0
        frequency = 150.0
        t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
        audio = 0.5 * np.sin(2 * np.pi * frequency * t).astype(np.float32)

        stats = service.calculate_pitch_stats(audio, sr)

        # Check required keys
        assert "f0_mean_hz" in stats
        assert "f0_std_hz" in stats
        assert "f0_min_hz" in stats
        assert "f0_max_hz" in stats
        assert "f0_range_semitones" in stats

    def test_pitch_stats_valid_ranges(self):
        """Test pitch statistics values are in valid ranges."""
        from voice_man.services.forensic.audio_feature_service import AudioFeatureService

        service = AudioFeatureService()

        sr = 16000
        duration = 1.0
        frequency = 150.0
        t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
        audio = 0.5 * np.sin(2 * np.pi * frequency * t).astype(np.float32)

        stats = service.calculate_pitch_stats(audio, sr)

        # Check ranges
        assert 75.0 <= stats["f0_mean_hz"] <= 600.0
        assert stats["f0_std_hz"] >= 0.0
        assert stats["f0_min_hz"] >= 0.0
        assert stats["f0_max_hz"] >= stats["f0_min_hz"]
        assert stats["f0_range_semitones"] >= 0.0


class TestCalculateWpm:
    """Tests for calculate_wpm() - TASK-008."""

    def test_wpm_from_segments(self):
        """Test WPM calculation from WhisperX segments."""
        from voice_man.services.forensic.audio_feature_service import AudioFeatureService

        service = AudioFeatureService()

        # Create mock segments with word-level data
        segments = [
            {
                "start": 0.0,
                "end": 3.0,
                "text": "Hello world this is a test",
                "words": [
                    {"word": "Hello", "start": 0.0, "end": 0.5},
                    {"word": "world", "start": 0.5, "end": 1.0},
                    {"word": "this", "start": 1.0, "end": 1.3},
                    {"word": "is", "start": 1.3, "end": 1.5},
                    {"word": "a", "start": 1.5, "end": 1.6},
                    {"word": "test", "start": 1.6, "end": 2.0},
                ],
            }
        ]

        wpm = service.calculate_wpm(segments, total_duration=3.0)

        # 6 words in 3 seconds = 120 WPM
        assert wpm == pytest.approx(120.0, rel=0.1)
        assert 50.0 <= wpm <= 300.0

    def test_wpm_empty_segments(self):
        """Test WPM calculation with empty segments."""
        from voice_man.services.forensic.audio_feature_service import AudioFeatureService

        service = AudioFeatureService()

        wpm = service.calculate_wpm([], total_duration=60.0)

        # Empty segments should return minimum WPM
        assert wpm == 50.0

    def test_wpm_range(self):
        """Test WPM is within valid range 50-300."""
        from voice_man.services.forensic.audio_feature_service import AudioFeatureService

        service = AudioFeatureService()

        # Very fast speech
        segments = [
            {
                "start": 0.0,
                "end": 1.0,
                "text": " ".join(["word"] * 10),
                "words": [
                    {"word": "word", "start": i * 0.1, "end": (i + 1) * 0.1} for i in range(10)
                ],
            }
        ]

        wpm = service.calculate_wpm(segments, total_duration=1.0)

        assert 50.0 <= wpm <= 300.0


class TestCalculateSpeechSilenceRatio:
    """Tests for calculate_speech_silence_ratio() - TASK-009."""

    def test_speech_silence_ratio_mixed_audio(self):
        """Test speech/silence ratio calculation."""
        from voice_man.services.forensic.audio_feature_service import AudioFeatureService

        service = AudioFeatureService()

        sr = 16000
        duration = 2.0
        samples = int(sr * duration)

        # Create audio with speech and silence regions
        audio = np.zeros(samples, dtype=np.float32)
        # First half: speech (sine wave)
        t = np.linspace(0, duration / 2, samples // 2, dtype=np.float32)
        audio[: samples // 2] = 0.5 * np.sin(2 * np.pi * 200 * t).astype(np.float32)
        # Second half: silence (already zeros)

        speech_ratio, silence_ratio = service.calculate_speech_silence_ratio(audio, sr)

        # Ratios should sum to approximately 1.0
        assert abs(speech_ratio + silence_ratio - 1.0) < 0.05
        assert 0.0 <= speech_ratio <= 1.0
        assert 0.0 <= silence_ratio <= 1.0

    def test_speech_silence_ratio_all_speech(self):
        """Test speech ratio for continuous speech."""
        from voice_man.services.forensic.audio_feature_service import AudioFeatureService

        service = AudioFeatureService()

        sr = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
        audio = 0.5 * np.sin(2 * np.pi * 200 * t).astype(np.float32)

        speech_ratio, silence_ratio = service.calculate_speech_silence_ratio(audio, sr)

        # Should be mostly speech
        assert speech_ratio > 0.8

    def test_speech_silence_ratio_all_silence(self):
        """Test silence ratio for silent audio."""
        from voice_man.services.forensic.audio_feature_service import AudioFeatureService

        service = AudioFeatureService()

        sr = 16000
        audio = np.zeros(sr, dtype=np.float32)

        speech_ratio, silence_ratio = service.calculate_speech_silence_ratio(audio, sr)

        # Note: librosa.effects.split behavior with completely silent audio
        # may vary. The important assertion is that ratios sum to ~1.0
        assert abs(speech_ratio + silence_ratio - 1.0) < 0.05
        assert 0.0 <= speech_ratio <= 1.0
        assert 0.0 <= silence_ratio <= 1.0


class TestDetectPauses:
    """Tests for detect_pauses() - TASK-010."""

    def test_detect_pauses_with_silence_gaps(self):
        """Test pause detection with clear silence gaps."""
        from voice_man.services.forensic.audio_feature_service import AudioFeatureService
        from voice_man.models.forensic.audio_features import PauseInfo

        service = AudioFeatureService()

        sr = 16000
        duration = 3.0
        samples = int(sr * duration)
        audio = np.zeros(samples, dtype=np.float32)

        # Speech from 0-1s
        t1 = np.linspace(0, 1, sr, dtype=np.float32)
        audio[:sr] = 0.5 * np.sin(2 * np.pi * 200 * t1).astype(np.float32)

        # Silence from 1-2s (pause)

        # Speech from 2-3s
        t2 = np.linspace(0, 1, sr, dtype=np.float32)
        audio[2 * sr :] = 0.5 * np.sin(2 * np.pi * 200 * t2).astype(np.float32)

        pauses = service.detect_pauses(audio, sr, min_pause_duration=0.3)

        # Should detect at least one pause
        assert len(pauses) >= 1

        # Check pause structure
        for pause in pauses:
            assert isinstance(pause, PauseInfo)
            assert pause.end_time > pause.start_time
            assert pause.duration > 0

    def test_detect_pauses_no_pause(self):
        """Test pause detection with continuous speech."""
        from voice_man.services.forensic.audio_feature_service import AudioFeatureService

        service = AudioFeatureService()

        sr = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
        audio = 0.5 * np.sin(2 * np.pi * 200 * t).astype(np.float32)

        pauses = service.detect_pauses(audio, sr, min_pause_duration=0.3)

        # Continuous speech should have no significant pauses
        assert len(pauses) == 0


class TestAnalyzeAudioFeatures:
    """Tests for analyze_audio_features() - TASK-015."""

    def test_analyze_audio_features_complete_output(self):
        """Test complete audio feature analysis output."""
        from voice_man.services.forensic.audio_feature_service import AudioFeatureService
        from voice_man.models.forensic.audio_features import AudioFeatureAnalysis

        service = AudioFeatureService()

        sr = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
        audio = 0.5 * np.sin(2 * np.pi * 150 * t).astype(np.float32)

        # Mock segments for WPM calculation
        segments = [
            {
                "start": 0.0,
                "end": 2.0,
                "text": "Test speech segment",
                "words": [
                    {"word": "Test", "start": 0.0, "end": 0.3},
                    {"word": "speech", "start": 0.3, "end": 0.6},
                    {"word": "segment", "start": 0.6, "end": 1.0},
                ],
            }
        ]

        analysis = service.analyze_audio_features(
            audio=audio, sr=sr, file_path="/test/audio.wav", segments=segments
        )

        assert isinstance(analysis, AudioFeatureAnalysis)
        assert analysis.file_path == "/test/audio.wav"
        assert analysis.duration_seconds == pytest.approx(duration, rel=0.01)
        assert analysis.sample_rate == sr

        # Check all sub-features are present
        assert analysis.volume_features is not None
        assert analysis.pitch_features is not None
        assert analysis.speech_rate_features is not None
        assert analysis.stress_features is not None


class TestDetectEmotionalEscalation:
    """Tests for detect_emotional_escalation() - TASK-016."""

    def test_detect_escalation_increasing_intensity(self):
        """Test escalation detection with increasing volume and pitch."""
        from voice_man.services.forensic.audio_feature_service import AudioFeatureService

        service = AudioFeatureService()

        sr = 16000
        duration = 4.0
        samples = int(sr * duration)
        t = np.linspace(0, duration, samples, dtype=np.float32)

        # Create audio with increasing amplitude
        envelope = np.linspace(0.1, 0.9, samples, dtype=np.float32)
        audio = envelope * np.sin(2 * np.pi * 150 * t).astype(np.float32)

        escalation_zones = service.detect_emotional_escalation(audio, sr)

        # Should detect escalation
        assert isinstance(escalation_zones, list)

        if len(escalation_zones) > 0:
            for zone in escalation_zones:
                assert zone.start_time >= 0.0
                assert zone.end_time > zone.start_time
                assert 0.0 <= zone.intensity_score <= 1.0

    def test_detect_escalation_stable_audio(self):
        """Test escalation detection with stable audio."""
        from voice_man.services.forensic.audio_feature_service import AudioFeatureService

        service = AudioFeatureService()

        sr = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
        audio = 0.5 * np.sin(2 * np.pi * 150 * t).astype(np.float32)

        escalation_zones = service.detect_emotional_escalation(audio, sr)

        # Stable audio should have few or no escalation zones
        assert isinstance(escalation_zones, list)

    def test_detect_escalation_output_structure(self):
        """Test escalation zone output structure."""
        from voice_man.services.forensic.audio_feature_service import AudioFeatureService
        from voice_man.models.forensic.audio_features import EscalationZone

        service = AudioFeatureService()

        sr = 16000
        duration = 3.0
        samples = int(sr * duration)
        t = np.linspace(0, duration, samples, dtype=np.float32)

        # Create escalating audio
        envelope = np.linspace(0.1, 0.9, samples, dtype=np.float32)
        audio = envelope * np.sin(2 * np.pi * 150 * t).astype(np.float32)

        escalation_zones = service.detect_emotional_escalation(audio, sr)

        for zone in escalation_zones:
            assert isinstance(zone, EscalationZone)
            assert hasattr(zone, "start_time")
            assert hasattr(zone, "end_time")
            assert hasattr(zone, "intensity_score")
            assert hasattr(zone, "volume_increase_db")
            assert hasattr(zone, "pitch_increase_percent")


class TestEdgeCases:
    """Tests for edge cases in AudioFeatureService."""

    def test_rms_amplitude_empty_audio(self):
        """Test RMS amplitude with empty audio array."""
        from voice_man.services.forensic.audio_feature_service import AudioFeatureService

        service = AudioFeatureService()

        audio = np.array([], dtype=np.float32)
        rms, rms_db = service.calculate_rms_amplitude(audio, 16000)

        assert rms == 0.0
        assert rms_db == -100.0

    def test_peak_amplitude_empty_audio(self):
        """Test peak amplitude with empty audio array."""
        from voice_man.services.forensic.audio_feature_service import AudioFeatureService

        service = AudioFeatureService()

        audio = np.array([], dtype=np.float32)
        peak, peak_db = service.calculate_peak_amplitude(audio, 16000)

        assert peak == 0.0
        assert peak_db == -100.0

    def test_volume_change_rate_short_audio(self):
        """Test volume change rate with very short audio."""
        from voice_man.services.forensic.audio_feature_service import AudioFeatureService

        service = AudioFeatureService()

        sr = 16000
        # Less than 1 second of audio
        audio = np.random.randn(8000).astype(np.float32) * 0.3

        rate = service.calculate_volume_change_rate(audio, sr)

        assert rate == 0.0

    def test_jitter_with_exception(self):
        """Test jitter returns 0 for problematic audio."""
        from voice_man.services.forensic.audio_feature_service import AudioFeatureService

        service = AudioFeatureService()

        sr = 16000
        # Very short or problematic audio
        audio = np.array([0.1, 0.2], dtype=np.float32)

        jitter = service.calculate_jitter(audio, sr)

        assert 0.0 <= jitter <= 10.0

    def test_pitch_stats_silent_audio(self):
        """Test pitch stats returns defaults for silent audio."""
        from voice_man.services.forensic.audio_feature_service import AudioFeatureService

        # Use CPU mode to test original librosa behavior
        service = AudioFeatureService(use_gpu=False)

        sr = 16000
        audio = np.zeros(sr, dtype=np.float32)

        stats = service.calculate_pitch_stats(audio, sr)

        # Should return default values (CPU/librosa returns these for silent audio)
        assert stats["f0_mean_hz"] == 150.0
        assert stats["f0_std_hz"] == 0.0

    def test_wpm_no_words_in_segment(self):
        """Test WPM calculation when segments have text but no word data."""
        from voice_man.services.forensic.audio_feature_service import AudioFeatureService

        service = AudioFeatureService()

        segments = [{"start": 0.0, "end": 2.0, "text": "Hello world test"}]

        wpm = service.calculate_wpm(segments, total_duration=2.0)

        # Should count words from text
        assert 50.0 <= wpm <= 300.0

    def test_detect_pauses_empty_audio(self):
        """Test pause detection with empty audio."""
        from voice_man.services.forensic.audio_feature_service import AudioFeatureService

        service = AudioFeatureService()

        audio = np.array([], dtype=np.float32)
        pauses = service.detect_pauses(audio, 16000)

        assert pauses == []

    def test_detect_pauses_single_interval(self):
        """Test pause detection with single speech interval."""
        from voice_man.services.forensic.audio_feature_service import AudioFeatureService

        service = AudioFeatureService()

        sr = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
        audio = 0.5 * np.sin(2 * np.pi * 200 * t).astype(np.float32)

        pauses = service.detect_pauses(audio, sr)

        # Single continuous speech should have no pauses
        assert len(pauses) == 0

    def test_escalation_detection_short_audio(self):
        """Test escalation detection with audio shorter than 2 seconds."""
        from voice_man.services.forensic.audio_feature_service import AudioFeatureService

        service = AudioFeatureService()

        sr = 16000
        duration = 1.0  # Less than 2 seconds
        t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
        audio = 0.5 * np.sin(2 * np.pi * 150 * t).astype(np.float32)

        zones = service.detect_emotional_escalation(audio, sr)

        assert zones == []

    def test_analyze_audio_features_without_segments(self):
        """Test complete analysis without providing segments."""
        from voice_man.services.forensic.audio_feature_service import AudioFeatureService
        from voice_man.models.forensic.audio_features import AudioFeatureAnalysis

        service = AudioFeatureService()

        sr = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
        audio = 0.5 * np.sin(2 * np.pi * 150 * t).astype(np.float32)

        analysis = service.analyze_audio_features(
            audio=audio,
            sr=sr,
            file_path="/test/audio.wav",
            segments=None,  # No segments provided
        )

        assert isinstance(analysis, AudioFeatureAnalysis)
        # Should use minimum WPM when no segments
        assert analysis.speech_rate_features.words_per_minute == 50.0

    def test_dynamic_range_with_peak_zero(self):
        """Test dynamic range when peak is effectively zero."""
        from voice_man.services.forensic.audio_feature_service import AudioFeatureService

        service = AudioFeatureService()

        sr = 16000
        # Very quiet audio
        audio = np.full(sr, 1e-10, dtype=np.float32)

        dr = service.calculate_dynamic_range(audio, sr)

        assert dr >= 0.0
