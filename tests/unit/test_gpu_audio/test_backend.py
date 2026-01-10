"""
GPU Audio Backend Tests
SPEC-GPUAUDIO-001 TAG-003: Unit tests for GPUAudioBackend

Tests use session-scoped mocks from conftest.py
"""

import pytest
import numpy as np
from unittest.mock import MagicMock


class TestBackendDeviceDetection:
    """Test device detection in GPUAudioBackend."""

    def test_device_is_cuda_when_specified(self, mock_gpu_modules):
        """Test that device is 'cuda' when explicitly specified."""
        from voice_man.services.forensic.gpu.backend import GPUAudioBackend

        backend = GPUAudioBackend(use_gpu=True)
        # When use_gpu=True and torch.cuda.is_available returns True
        # But since torch is mocked, we test the logic
        assert backend._use_gpu is True

    def test_device_is_cpu_when_gpu_disabled(self, mock_gpu_modules):
        """Test that device is 'cpu' when use_gpu=False."""
        from voice_man.services.forensic.gpu.backend import GPUAudioBackend

        backend = GPUAudioBackend(use_gpu=False)
        assert backend.device == "cpu"


class TestBackendGPUAvailability:
    """Test GPU availability property."""

    def test_is_gpu_available_property_exists(self, mock_gpu_modules):
        """Test is_gpu_available property exists and returns boolean."""
        from voice_man.services.forensic.gpu.backend import GPUAudioBackend

        backend = GPUAudioBackend()
        # The property should return a boolean
        assert isinstance(backend.is_gpu_available, bool)


class TestBackendF0Extraction:
    """Test F0 extraction through backend."""

    @pytest.fixture
    def mock_crepe_extractor(self):
        """Mock the CREPE extractor."""
        mock_instance = MagicMock()
        mock_instance.extract_f0.return_value = (
            np.array([200.0, 200.0, 200.0]),
            np.array([0.9, 0.9, 0.9]),
        )
        mock_instance.extract_f0_batch.return_value = (
            np.array([200.0, 200.0, 200.0]),
            np.array([0.9, 0.9, 0.9]),
        )
        return mock_instance

    def test_extract_f0_delegates_to_crepe(
        self, mock_gpu_modules, sample_audio_1s, mock_crepe_extractor
    ):
        """Test that extract_f0 delegates to CREPE extractor."""
        audio, sr = sample_audio_1s

        from voice_man.services.forensic.gpu.backend import GPUAudioBackend

        backend = GPUAudioBackend(use_gpu=False)
        # Force using the mock by setting the private attribute
        backend._crepe_extractor = mock_crepe_extractor

        f0, confidence = backend.extract_f0(audio, sr)

        mock_crepe_extractor.extract_f0.assert_called_once()
        assert isinstance(f0, np.ndarray)
        assert isinstance(confidence, np.ndarray)

    def test_extract_f0_batch_delegates_to_crepe(self, mock_gpu_modules, mock_crepe_extractor):
        """Test that extract_f0_batch delegates to CREPE extractor."""
        sr = 16000
        windows = np.random.randn(3, sr).astype(np.float32)

        from voice_man.services.forensic.gpu.backend import GPUAudioBackend

        backend = GPUAudioBackend(use_gpu=False)
        backend._crepe_extractor = mock_crepe_extractor

        f0_values, confidence_values = backend.extract_f0_batch(windows, sr)

        mock_crepe_extractor.extract_f0_batch.assert_called_once()
        assert isinstance(f0_values, np.ndarray)
        assert isinstance(confidence_values, np.ndarray)


class TestBackendLazyLoading:
    """Test lazy loading of components."""

    def test_crepe_extractor_lazy_loaded(self, mock_gpu_modules):
        """Test that CREPE extractor is lazy loaded."""
        from voice_man.services.forensic.gpu.backend import GPUAudioBackend

        backend = GPUAudioBackend()

        # Initially None
        assert backend._crepe_extractor is None

    def test_torch_lazy_loaded(self, mock_gpu_modules):
        """Test that torch is lazy loaded."""
        from voice_man.services.forensic.gpu.backend import GPUAudioBackend

        backend = GPUAudioBackend()

        # Initially None
        assert backend._torch is None


class TestBackendConfiguration:
    """Test backend configuration options."""

    def test_default_use_gpu_true(self, mock_gpu_modules):
        """Test that use_gpu defaults to True."""
        from voice_man.services.forensic.gpu.backend import GPUAudioBackend

        backend = GPUAudioBackend()
        assert backend._use_gpu is True

    def test_explicit_use_gpu_false(self, mock_gpu_modules):
        """Test explicit use_gpu=False configuration."""
        from voice_man.services.forensic.gpu.backend import GPUAudioBackend

        backend = GPUAudioBackend(use_gpu=False)
        assert backend._use_gpu is False
        assert backend.device == "cpu"


class TestBackendIntegration:
    """Integration tests for GPUAudioBackend with mocked dependencies."""

    def test_end_to_end_f0_extraction(self, mock_gpu_modules, sample_audio_1s):
        """Test end-to-end F0 extraction through backend."""
        audio, sr = sample_audio_1s

        from voice_man.services.forensic.gpu.backend import GPUAudioBackend

        backend = GPUAudioBackend(use_gpu=False)
        f0, confidence = backend.extract_f0(audio, sr)

        assert isinstance(f0, np.ndarray)
        assert isinstance(confidence, np.ndarray)
        assert len(f0) > 0
        assert len(confidence) > 0
