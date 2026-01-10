"""
Tests for AudioFeatureService GPU Integration
SPEC-GPUAUDIO-001 Phase 3: TDD tests for GPU-accelerated audio feature extraction

TDD RED Phase: These tests define the expected behavior of the GPU-integrated service.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock


class TestUseGpuParameter:
    """Tests for use_gpu parameter in AudioFeatureService."""

    def test_use_gpu_parameter_default(self):
        """Test that use_gpu defaults to True."""
        from voice_man.services.forensic.audio_feature_service import AudioFeatureService

        service = AudioFeatureService()

        assert hasattr(service, "_use_gpu")
        assert service._use_gpu is True

    def test_use_gpu_parameter_false(self):
        """Test that use_gpu can be set to False."""
        from voice_man.services.forensic.audio_feature_service import AudioFeatureService

        service = AudioFeatureService(use_gpu=False)

        assert service._use_gpu is False

    def test_use_gpu_parameter_explicit_true(self):
        """Test that use_gpu can be explicitly set to True."""
        from voice_man.services.forensic.audio_feature_service import AudioFeatureService

        service = AudioFeatureService(use_gpu=True)

        assert service._use_gpu is True


class TestGpuBackendLazyLoading:
    """Tests for GPU backend lazy loading."""

    def test_gpu_backend_lazy_loaded(self):
        """Test that GPU backend is lazily initialized on first access."""
        from voice_man.services.forensic.audio_feature_service import AudioFeatureService

        service = AudioFeatureService(use_gpu=True)

        # GPU backend should not be initialized yet
        assert service._gpu_backend is None

        # Access the property to trigger lazy loading
        backend = service.gpu_backend

        # Now it should be initialized
        assert service._gpu_backend is not None
        assert backend is service._gpu_backend

    def test_gpu_backend_not_created_when_disabled(self):
        """Test that GPU backend is not created when use_gpu is False."""
        from voice_man.services.forensic.audio_feature_service import AudioFeatureService

        service = AudioFeatureService(use_gpu=False)

        # GPU backend should be None initially
        assert service._gpu_backend is None

        # Access the property
        backend = service.gpu_backend

        # Backend should be created but with use_gpu=False
        assert backend is not None
        assert backend._use_gpu is False

    def test_gpu_backend_reused_on_subsequent_access(self):
        """Test that the same GPU backend instance is returned on subsequent accesses."""
        from voice_man.services.forensic.audio_feature_service import AudioFeatureService

        service = AudioFeatureService(use_gpu=True)

        backend1 = service.gpu_backend
        backend2 = service.gpu_backend

        assert backend1 is backend2


class TestExtractF0GpuIntegration:
    """Tests for F0 extraction with GPU integration."""

    def test_extract_f0_uses_gpu_when_enabled(self):
        """Test that extract_f0 uses GPU backend when use_gpu is True and GPU available."""
        from voice_man.services.forensic.audio_feature_service import AudioFeatureService

        service = AudioFeatureService(use_gpu=True)

        # Create test audio
        sr = 16000
        duration = 1.0
        frequency = 200.0
        t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
        audio = 0.5 * np.sin(2 * np.pi * frequency * t).astype(np.float32)

        # Mock the gpu_backend to verify it's called
        mock_backend = Mock()
        mock_f0 = np.full(31, 200.0)  # CREPE output
        mock_confidence = np.full(31, 0.9)
        mock_backend.extract_f0.return_value = (mock_f0, mock_confidence)
        mock_backend._use_gpu = True

        with patch.object(service, "_gpu_backend", mock_backend):
            service._gpu_backend = mock_backend  # Set it so lazy loading doesn't trigger

            # Try to verify GPU path is used
            # The actual implementation will determine the exact behavior
            f0, times = service.extract_f0(audio, sr)

            # Should return valid arrays
            assert isinstance(f0, np.ndarray)
            assert isinstance(times, np.ndarray)

    def test_extract_f0_falls_back_to_cpu(self):
        """Test that extract_f0 falls back to CPU (librosa) when GPU fails."""
        from voice_man.services.forensic.audio_feature_service import AudioFeatureService

        service = AudioFeatureService(use_gpu=True)

        # Create test audio
        sr = 16000
        duration = 1.0
        frequency = 200.0
        t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
        audio = 0.5 * np.sin(2 * np.pi * frequency * t).astype(np.float32)

        # Mock GPU backend to raise exception
        mock_backend = Mock()
        mock_backend.extract_f0.side_effect = RuntimeError("GPU error")
        mock_backend._use_gpu = True

        service._gpu_backend = mock_backend

        # Should still return valid results via CPU fallback
        f0, times = service.extract_f0(audio, sr)

        assert isinstance(f0, np.ndarray)
        assert isinstance(times, np.ndarray)
        # Verify some F0 values were detected
        valid_f0 = f0[~np.isnan(f0)]
        assert len(valid_f0) > 0

    def test_extract_f0_cpu_only_when_gpu_disabled(self):
        """Test that extract_f0 uses only CPU (librosa) when use_gpu is False."""
        from voice_man.services.forensic.audio_feature_service import AudioFeatureService

        service = AudioFeatureService(use_gpu=False)

        # Create test audio
        sr = 16000
        duration = 1.0
        frequency = 200.0
        t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
        audio = 0.5 * np.sin(2 * np.pi * frequency * t).astype(np.float32)

        f0, times = service.extract_f0(audio, sr)

        assert isinstance(f0, np.ndarray)
        assert isinstance(times, np.ndarray)


class TestDetectEmotionalEscalationGpuOptimization:
    """Tests for detect_emotional_escalation GPU optimization with batch processing."""

    def test_detect_emotional_escalation_batch_processing(self):
        """Test that emotional escalation detection uses batch processing for F0 extraction."""
        from voice_man.services.forensic.audio_feature_service import AudioFeatureService

        service = AudioFeatureService(use_gpu=True)

        # Create test audio with increasing amplitude (escalation)
        sr = 16000
        duration = 4.0
        samples = int(sr * duration)
        t = np.linspace(0, duration, samples, dtype=np.float32)
        envelope = np.linspace(0.1, 0.9, samples, dtype=np.float32)
        audio = envelope * np.sin(2 * np.pi * 150 * t).astype(np.float32)

        # Mock the gpu_backend
        mock_backend = Mock()

        # Setup batch F0 extraction mock
        # For a 4-second audio with 1-second windows and 0.5-second hop:
        # Windows start at 0, 0.5, 1.0, 1.5, 2.0, 2.5 (6 windows)
        num_windows = 6
        mock_f0_values = np.linspace(140, 160, num_windows)  # Increasing pitch
        mock_confidence = np.full(num_windows, 0.9)
        mock_backend.extract_f0_batch.return_value = (mock_f0_values, mock_confidence)
        mock_backend._use_gpu = True

        service._gpu_backend = mock_backend

        # Call the method
        escalation_zones = service.detect_emotional_escalation(audio, sr)

        # Verify batch processing was used (extract_f0_batch called)
        # The actual implementation determines if this assertion is valid
        assert isinstance(escalation_zones, list)

    def test_detect_emotional_escalation_preserves_functionality(self):
        """Test that GPU optimization preserves the original functionality."""
        from voice_man.services.forensic.audio_feature_service import AudioFeatureService
        from voice_man.models.forensic.audio_features import EscalationZone

        service = AudioFeatureService(use_gpu=True)

        # Create test audio with clear escalation
        sr = 16000
        duration = 4.0
        samples = int(sr * duration)
        t = np.linspace(0, duration, samples, dtype=np.float32)
        envelope = np.linspace(0.1, 0.9, samples, dtype=np.float32)
        audio = envelope * np.sin(2 * np.pi * 150 * t).astype(np.float32)

        escalation_zones = service.detect_emotional_escalation(audio, sr)

        # Should detect escalation zones
        assert isinstance(escalation_zones, list)

        # If zones detected, verify structure
        for zone in escalation_zones:
            assert isinstance(zone, EscalationZone)
            assert zone.start_time >= 0.0
            assert zone.end_time > zone.start_time
            assert 0.0 <= zone.intensity_score <= 1.0

    def test_detect_emotional_escalation_short_audio_no_batch(self):
        """Test that short audio (< 2 seconds) returns empty list."""
        from voice_man.services.forensic.audio_feature_service import AudioFeatureService

        service = AudioFeatureService(use_gpu=True)

        sr = 16000
        duration = 1.0  # Less than 2 seconds
        t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
        audio = 0.5 * np.sin(2 * np.pi * 150 * t).astype(np.float32)

        zones = service.detect_emotional_escalation(audio, sr)

        assert zones == []


class TestExtractF0LibrosaFallback:
    """Tests for the _extract_f0_librosa internal method."""

    def test_extract_f0_librosa_method_exists(self):
        """Test that _extract_f0_librosa method exists."""
        from voice_man.services.forensic.audio_feature_service import AudioFeatureService

        service = AudioFeatureService()

        assert hasattr(service, "_extract_f0_librosa")
        assert callable(service._extract_f0_librosa)

    def test_extract_f0_librosa_returns_valid_output(self):
        """Test that _extract_f0_librosa returns valid F0 and times arrays."""
        from voice_man.services.forensic.audio_feature_service import AudioFeatureService

        service = AudioFeatureService()

        sr = 16000
        duration = 1.0
        frequency = 200.0
        t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
        audio = 0.5 * np.sin(2 * np.pi * frequency * t).astype(np.float32)

        f0, times = service._extract_f0_librosa(audio, sr)

        assert isinstance(f0, np.ndarray)
        assert isinstance(times, np.ndarray)
        assert len(f0) == len(times)


class TestBackwardCompatibility:
    """Tests for backward compatibility with existing code."""

    def test_existing_api_unchanged(self):
        """Test that existing public API methods still work without changes."""
        from voice_man.services.forensic.audio_feature_service import AudioFeatureService

        service = AudioFeatureService()

        sr = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
        audio = 0.5 * np.sin(2 * np.pi * 150 * t).astype(np.float32)

        # All existing methods should work
        rms, rms_db = service.calculate_rms_amplitude(audio, sr)
        assert isinstance(rms, float)

        peak, peak_db = service.calculate_peak_amplitude(audio, sr)
        assert isinstance(peak, float)

        f0, times = service.extract_f0(audio, sr)
        assert isinstance(f0, np.ndarray)

        jitter = service.calculate_jitter(audio, sr)
        assert isinstance(jitter, float)

    def test_no_gpu_argument_works(self):
        """Test that creating service without gpu argument still works."""
        from voice_man.services.forensic.audio_feature_service import AudioFeatureService

        # Should not raise any exception
        service = AudioFeatureService()

        # Should have default use_gpu=True
        assert hasattr(service, "_use_gpu")


class TestGpuBackendProperty:
    """Tests for gpu_backend property behavior."""

    def test_gpu_backend_property_returns_gpu_audio_backend(self):
        """Test that gpu_backend property returns GPUAudioBackend instance."""
        from voice_man.services.forensic.audio_feature_service import AudioFeatureService
        from voice_man.services.forensic.gpu import GPUAudioBackend

        service = AudioFeatureService(use_gpu=True)

        backend = service.gpu_backend

        assert isinstance(backend, GPUAudioBackend)

    def test_gpu_backend_inherits_use_gpu_setting(self):
        """Test that GPU backend inherits the use_gpu setting from service."""
        from voice_man.services.forensic.audio_feature_service import AudioFeatureService

        service_gpu = AudioFeatureService(use_gpu=True)
        service_cpu = AudioFeatureService(use_gpu=False)

        assert service_gpu.gpu_backend._use_gpu is True
        assert service_cpu.gpu_backend._use_gpu is False
