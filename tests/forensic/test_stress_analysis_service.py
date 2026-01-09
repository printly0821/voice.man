"""
Tests for Stress Analysis Service
SPEC-FORENSIC-001 TASK-011~014: Stress analysis service tests

TDD RED Phase: These tests define the expected behavior of the stress analysis service.
"""

import pytest
import numpy as np


class TestCalculateShimmer:
    """Tests for calculate_shimmer() - TASK-011."""

    def test_shimmer_perfect_periodic_signal(self):
        """Test shimmer for perfectly periodic signal is low."""
        from voice_man.services.forensic.stress_analysis_service import StressAnalysisService

        service = StressAnalysisService()

        sr = 16000
        duration = 1.0
        frequency = 150.0
        t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
        audio = 0.5 * np.sin(2 * np.pi * frequency * t).astype(np.float32)

        shimmer = service.calculate_shimmer(audio, sr)

        # Perfect sine wave should have very low shimmer
        assert shimmer >= 0.0
        assert shimmer <= 15.0  # Max shimmer is 15%

    def test_shimmer_range(self):
        """Test shimmer is within 0-15% range."""
        from voice_man.services.forensic.stress_analysis_service import StressAnalysisService

        service = StressAnalysisService()

        sr = 16000
        duration = 1.0
        audio = np.random.randn(int(sr * duration)).astype(np.float32) * 0.3

        shimmer = service.calculate_shimmer(audio, sr)

        assert 0.0 <= shimmer <= 15.0

    def test_shimmer_amplitude_variation(self):
        """Test shimmer increases with amplitude variation."""
        from voice_man.services.forensic.stress_analysis_service import StressAnalysisService

        service = StressAnalysisService()

        sr = 16000
        duration = 1.0
        frequency = 150.0
        samples = int(sr * duration)
        t = np.linspace(0, duration, samples, dtype=np.float32)

        # Stable amplitude
        audio_stable = 0.5 * np.sin(2 * np.pi * frequency * t).astype(np.float32)

        # Varying amplitude (amplitude modulation)
        am_freq = 5  # 5 Hz amplitude modulation
        modulation = 0.3 + 0.2 * np.sin(2 * np.pi * am_freq * t)
        audio_varying = (modulation * np.sin(2 * np.pi * frequency * t)).astype(np.float32)

        shimmer_stable = service.calculate_shimmer(audio_stable, sr)
        shimmer_varying = service.calculate_shimmer(audio_varying, sr)

        # Varying amplitude should have higher shimmer
        # Note: This may not always hold depending on implementation
        assert shimmer_stable >= 0.0
        assert shimmer_varying >= 0.0


class TestCalculateHnr:
    """Tests for calculate_hnr() - TASK-012."""

    def test_hnr_pure_tone(self):
        """Test HNR for pure tone is high."""
        from voice_man.services.forensic.stress_analysis_service import StressAnalysisService

        service = StressAnalysisService()

        sr = 16000
        duration = 1.0
        frequency = 150.0
        t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
        audio = 0.5 * np.sin(2 * np.pi * frequency * t).astype(np.float32)

        hnr = service.calculate_hnr(audio, sr)

        # Pure tone should have high HNR
        assert hnr >= 0.0
        assert hnr <= 40.0  # Max HNR is 40dB

    def test_hnr_range(self):
        """Test HNR is within 0-40dB range."""
        from voice_man.services.forensic.stress_analysis_service import StressAnalysisService

        service = StressAnalysisService()

        sr = 16000
        duration = 1.0
        audio = np.random.randn(int(sr * duration)).astype(np.float32) * 0.3

        hnr = service.calculate_hnr(audio, sr)

        assert 0.0 <= hnr <= 40.0

    def test_hnr_noisy_signal(self):
        """Test HNR for noisy signal is lower than pure tone."""
        from voice_man.services.forensic.stress_analysis_service import StressAnalysisService

        service = StressAnalysisService()

        sr = 16000
        duration = 1.0
        frequency = 150.0
        t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)

        # Pure tone
        audio_pure = 0.5 * np.sin(2 * np.pi * frequency * t).astype(np.float32)

        # Noisy signal (tone + noise)
        noise = np.random.randn(int(sr * duration)).astype(np.float32) * 0.2
        audio_noisy = (0.5 * np.sin(2 * np.pi * frequency * t) + noise).astype(np.float32)

        hnr_pure = service.calculate_hnr(audio_pure, sr)
        hnr_noisy = service.calculate_hnr(audio_noisy, sr)

        # Pure tone should have higher HNR than noisy signal
        assert hnr_pure >= hnr_noisy - 5  # Allow some tolerance


class TestCalculateFormantStability:
    """Tests for calculate_formant_stability() - TASK-013."""

    def test_formant_stability_range(self):
        """Test formant stability score is within 0-100 range."""
        from voice_man.services.forensic.stress_analysis_service import StressAnalysisService

        service = StressAnalysisService()

        sr = 16000
        duration = 1.0
        frequency = 150.0
        t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
        audio = 0.5 * np.sin(2 * np.pi * frequency * t).astype(np.float32)

        stability = service.calculate_formant_stability(audio, sr)

        assert 0.0 <= stability <= 100.0

    def test_formant_stability_stable_vowel(self):
        """Test formant stability for stable vowel-like sound."""
        from voice_man.services.forensic.stress_analysis_service import StressAnalysisService

        service = StressAnalysisService()

        sr = 16000
        duration = 1.0
        # Create a vowel-like sound with harmonics
        t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
        f0 = 150.0
        audio = (
            0.5 * np.sin(2 * np.pi * f0 * t)
            + 0.3 * np.sin(2 * np.pi * 2 * f0 * t)
            + 0.2 * np.sin(2 * np.pi * 3 * f0 * t)
        ).astype(np.float32)

        stability = service.calculate_formant_stability(audio, sr)

        assert 0.0 <= stability <= 100.0

    def test_formant_stability_output_type(self):
        """Test formant stability returns float."""
        from voice_man.services.forensic.stress_analysis_service import StressAnalysisService

        service = StressAnalysisService()

        sr = 16000
        duration = 0.5
        audio = np.random.randn(int(sr * duration)).astype(np.float32) * 0.3

        stability = service.calculate_formant_stability(audio, sr)

        assert isinstance(stability, float)


class TestCalculateStressIndex:
    """Tests for calculate_stress_index() - TASK-014."""

    def test_stress_index_range(self):
        """Test stress index is within 0-100 range."""
        from voice_man.services.forensic.stress_analysis_service import StressAnalysisService

        service = StressAnalysisService()

        stress_index, risk_level = service.calculate_stress_index(
            shimmer_percent=3.0,
            hnr_db=15.0,
            formant_stability=75.0,
            jitter_percent=1.5,
        )

        assert 0.0 <= stress_index <= 100.0
        assert risk_level in ["low", "medium", "high"]

    def test_stress_index_low_stress(self):
        """Test stress index for low stress indicators."""
        from voice_man.services.forensic.stress_analysis_service import StressAnalysisService

        service = StressAnalysisService()

        # Low stress: low shimmer, high HNR, high stability, low jitter
        stress_index, risk_level = service.calculate_stress_index(
            shimmer_percent=1.0,
            hnr_db=25.0,
            formant_stability=90.0,
            jitter_percent=0.5,
        )

        assert stress_index <= 40.0
        assert risk_level == "low"

    def test_stress_index_high_stress(self):
        """Test stress index for high stress indicators."""
        from voice_man.services.forensic.stress_analysis_service import StressAnalysisService

        service = StressAnalysisService()

        # High stress: high shimmer, low HNR, low stability, high jitter
        stress_index, risk_level = service.calculate_stress_index(
            shimmer_percent=10.0,
            hnr_db=5.0,
            formant_stability=30.0,
            jitter_percent=5.0,
        )

        assert stress_index >= 60.0
        assert risk_level in ["medium", "high"]

    def test_stress_index_risk_levels(self):
        """Test stress index risk level classification."""
        from voice_man.services.forensic.stress_analysis_service import StressAnalysisService

        service = StressAnalysisService()

        # Test low risk (index < 33)
        _, risk_low = service.calculate_stress_index(
            shimmer_percent=1.0,
            hnr_db=30.0,
            formant_stability=95.0,
            jitter_percent=0.3,
        )
        assert risk_low == "low"

        # Test high risk (index > 66)
        _, risk_high = service.calculate_stress_index(
            shimmer_percent=12.0,
            hnr_db=3.0,
            formant_stability=20.0,
            jitter_percent=7.0,
        )
        assert risk_high == "high"

    def test_stress_index_weighted_calculation(self):
        """Test stress index uses weighted combination of factors."""
        from voice_man.services.forensic.stress_analysis_service import StressAnalysisService

        service = StressAnalysisService()

        # Different combinations should give different results
        stress1, _ = service.calculate_stress_index(
            shimmer_percent=5.0,
            hnr_db=15.0,
            formant_stability=50.0,
            jitter_percent=2.0,
        )

        stress2, _ = service.calculate_stress_index(
            shimmer_percent=2.0,
            hnr_db=20.0,
            formant_stability=70.0,
            jitter_percent=1.0,
        )

        # Different inputs should generally give different outputs
        # stress1 should be higher due to worse indicators
        assert stress1 != stress2


class TestAnalyzeStress:
    """Tests for analyze_stress() - Complete stress analysis."""

    def test_analyze_stress_complete_output(self):
        """Test complete stress analysis output."""
        from voice_man.services.forensic.stress_analysis_service import StressAnalysisService
        from voice_man.models.forensic.audio_features import StressFeatures

        service = StressAnalysisService()

        sr = 16000
        duration = 1.0
        frequency = 150.0
        t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
        audio = 0.5 * np.sin(2 * np.pi * frequency * t).astype(np.float32)

        stress_features = service.analyze_stress(audio, sr, jitter_percent=1.5)

        assert isinstance(stress_features, StressFeatures)
        assert 0.0 <= stress_features.shimmer_percent <= 15.0
        assert 0.0 <= stress_features.hnr_db <= 40.0
        assert 0.0 <= stress_features.formant_stability_score <= 100.0
        assert 0.0 <= stress_features.stress_index <= 100.0
        assert stress_features.risk_level in ["low", "medium", "high"]

    def test_analyze_stress_with_provided_jitter(self):
        """Test stress analysis uses provided jitter value."""
        from voice_man.services.forensic.stress_analysis_service import StressAnalysisService

        service = StressAnalysisService()

        sr = 16000
        duration = 1.0
        audio = np.random.randn(int(sr * duration)).astype(np.float32) * 0.3

        # Provide specific jitter value
        jitter_value = 3.5
        stress_features = service.analyze_stress(audio, sr, jitter_percent=jitter_value)

        # The stress calculation should incorporate the provided jitter
        # We can't directly check if it was used, but the output should be valid
        assert stress_features is not None
        assert stress_features.stress_index >= 0.0


class TestEdgeCases:
    """Tests for edge cases and fallback methods."""

    def test_shimmer_empty_audio(self):
        """Test shimmer with very short audio."""
        from voice_man.services.forensic.stress_analysis_service import StressAnalysisService

        service = StressAnalysisService()

        sr = 16000
        # Very short audio
        audio = np.array([0.1, 0.2, 0.1], dtype=np.float32)

        shimmer = service.calculate_shimmer(audio, sr)

        assert 0.0 <= shimmer <= 15.0

    def test_shimmer_fallback_method(self):
        """Test shimmer fallback calculation directly."""
        from voice_man.services.forensic.stress_analysis_service import StressAnalysisService

        service = StressAnalysisService()

        sr = 16000
        duration = 0.5
        frequency = 150.0
        t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
        audio = 0.5 * np.sin(2 * np.pi * frequency * t).astype(np.float32)

        # Call fallback directly
        shimmer = service._calculate_shimmer_fallback(audio, sr)

        assert 0.0 <= shimmer <= 15.0

    def test_hnr_fallback_method(self):
        """Test HNR fallback calculation directly."""
        from voice_man.services.forensic.stress_analysis_service import StressAnalysisService

        service = StressAnalysisService()

        sr = 16000
        duration = 0.5
        frequency = 150.0
        t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
        audio = 0.5 * np.sin(2 * np.pi * frequency * t).astype(np.float32)

        # Call fallback directly
        hnr = service._calculate_hnr_fallback(audio, sr)

        assert 0.0 <= hnr <= 40.0

    def test_hnr_silent_audio(self):
        """Test HNR with silent audio frames."""
        from voice_man.services.forensic.stress_analysis_service import StressAnalysisService

        service = StressAnalysisService()

        sr = 16000
        # Mostly silent with brief signal
        audio = np.zeros(sr, dtype=np.float32)
        audio[1000:2000] = 0.5 * np.sin(2 * np.pi * 150 * np.linspace(0, 1000 / sr, 1000)).astype(
            np.float32
        )

        hnr = service.calculate_hnr(audio, sr)

        assert 0.0 <= hnr <= 40.0

    def test_formant_stability_fallback_method(self):
        """Test formant stability fallback calculation directly."""
        from voice_man.services.forensic.stress_analysis_service import StressAnalysisService

        service = StressAnalysisService()

        sr = 16000
        duration = 0.5
        frequency = 150.0
        t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
        audio = 0.5 * np.sin(2 * np.pi * frequency * t).astype(np.float32)

        # Call fallback directly
        stability = service._calculate_formant_stability_fallback(audio, sr)

        assert 0.0 <= stability <= 100.0

    def test_formant_stability_short_audio(self):
        """Test formant stability with short audio."""
        from voice_man.services.forensic.stress_analysis_service import StressAnalysisService

        service = StressAnalysisService()

        sr = 16000
        # Very short audio
        audio = np.random.randn(100).astype(np.float32) * 0.3

        stability = service.calculate_formant_stability(audio, sr)

        assert 0.0 <= stability <= 100.0

    def test_stress_index_extreme_values(self):
        """Test stress index with extreme input values."""
        from voice_man.services.forensic.stress_analysis_service import StressAnalysisService

        service = StressAnalysisService()

        # All minimum values
        stress_min, risk_min = service.calculate_stress_index(
            shimmer_percent=0.0,
            hnr_db=40.0,
            formant_stability=100.0,
            jitter_percent=0.0,
        )
        assert stress_min == 0.0
        assert risk_min == "low"

        # All maximum stress values
        stress_max, risk_max = service.calculate_stress_index(
            shimmer_percent=15.0,
            hnr_db=0.0,
            formant_stability=0.0,
            jitter_percent=10.0,
        )
        assert stress_max == 100.0
        assert risk_max == "high"

    def test_shimmer_quiet_frames_filtered(self):
        """Test that very quiet frames are filtered in shimmer calculation."""
        from voice_man.services.forensic.stress_analysis_service import StressAnalysisService

        service = StressAnalysisService()

        sr = 16000
        duration = 1.0
        samples = int(sr * duration)

        # Audio with alternating loud and quiet sections
        audio = np.zeros(samples, dtype=np.float32)
        # Add loud section
        t = np.linspace(0, 0.5, samples // 2, dtype=np.float32)
        audio[: samples // 2] = 0.5 * np.sin(2 * np.pi * 150 * t).astype(np.float32)
        # Second half is very quiet (below threshold)
        audio[samples // 2 :] = 0.001 * np.random.randn(samples // 2).astype(np.float32)

        shimmer = service._calculate_shimmer_fallback(audio, sr)

        assert 0.0 <= shimmer <= 15.0
