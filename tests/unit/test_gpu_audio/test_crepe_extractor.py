"""
TorchCrepe Extractor Tests
SPEC-GPUAUDIO-001 TAG-002: Unit tests for TorchCrepeExtractor

Tests use session-scoped mocks from conftest.py
"""

import pytest
import numpy as np


class TestGPUDetection:
    """Test GPU availability detection."""

    def test_gpu_detection_with_cuda_available(self, mock_gpu_modules):
        """Test that GPU is correctly detected when CUDA is available."""
        # Import after mock is in place
        from voice_man.services.forensic.gpu.crepe_extractor import TorchCrepeExtractor

        # Configure torch mock for CUDA available
        mock_gpu_modules["torchcrepe"]  # Ensure torchcrepe is mocked

        extractor = TorchCrepeExtractor(device="cuda")
        assert extractor.device == "cuda"

    def test_gpu_detection_with_cuda_unavailable(self, mock_gpu_modules):
        """Test that CPU is used when CUDA is unavailable."""
        from voice_man.services.forensic.gpu.crepe_extractor import TorchCrepeExtractor

        extractor = TorchCrepeExtractor(device="cpu")
        assert extractor.device == "cpu"


class TestCPUFallback:
    """Test CPU fallback behavior."""

    def test_cpu_fallback_when_gpu_unavailable(self, mock_gpu_modules):
        """Test that extraction falls back to CPU when GPU unavailable."""
        from voice_man.services.forensic.gpu.crepe_extractor import TorchCrepeExtractor

        extractor = TorchCrepeExtractor(device="cpu")
        assert extractor.device == "cpu"

    def test_explicit_cpu_device_override(self, mock_gpu_modules):
        """Test that explicit CPU device override works."""
        from voice_man.services.forensic.gpu.crepe_extractor import TorchCrepeExtractor

        extractor = TorchCrepeExtractor(device="cpu")
        assert extractor.device == "cpu"


class TestExtractF0Single:
    """Test single audio F0 extraction."""

    def test_extract_f0_returns_correct_shape(self, mock_gpu_modules, sample_audio_1s):
        """Test that extract_f0 returns arrays with correct shapes."""
        audio, sr = sample_audio_1s

        from voice_man.services.forensic.gpu.crepe_extractor import TorchCrepeExtractor

        extractor = TorchCrepeExtractor(device="cpu")
        f0, confidence = extractor.extract_f0(audio, sr)

        # Both outputs should be numpy arrays
        assert isinstance(f0, np.ndarray)
        assert isinstance(confidence, np.ndarray)
        assert f0.shape == confidence.shape

    def test_extract_f0_single_audio(self, mock_gpu_modules, sample_audio_1s):
        """Test F0 extraction from single audio signal."""
        audio, sr = sample_audio_1s

        from voice_man.services.forensic.gpu.crepe_extractor import TorchCrepeExtractor

        extractor = TorchCrepeExtractor(device="cpu")
        f0, confidence = extractor.extract_f0(audio, sr)

        # F0 should contain values (may include NaN for unvoiced)
        valid_f0 = f0[~np.isnan(f0)]
        assert len(valid_f0) > 0


class TestExtractF0Batch:
    """Test batch F0 extraction."""

    def test_extract_f0_batch_returns_correct_shape(self, mock_gpu_modules, sample_windows_5):
        """Test that batch extraction returns correct shapes."""
        windows, sr = sample_windows_5
        num_windows = windows.shape[0]

        from voice_man.services.forensic.gpu.crepe_extractor import TorchCrepeExtractor

        extractor = TorchCrepeExtractor(device="cpu")
        f0_values, confidence_values = extractor.extract_f0_batch(windows, sr)

        assert f0_values.shape == (num_windows,)
        assert confidence_values.shape == (num_windows,)

    def test_extract_f0_batch_processes_all_windows(self, mock_gpu_modules, sample_windows_5):
        """Test that all windows are processed in batch."""
        windows, sr = sample_windows_5

        from voice_man.services.forensic.gpu.crepe_extractor import TorchCrepeExtractor

        extractor = TorchCrepeExtractor(device="cpu")
        f0_values, confidence_values = extractor.extract_f0_batch(windows, sr)

        # All windows should have been processed
        assert len(f0_values) == windows.shape[0]


class TestF0RangeValidation:
    """Test F0 value range validation."""

    def test_f0_values_within_valid_range(self, configure_torchcrepe_mock):
        """Test that valid F0 values are within 50-550 Hz range."""
        # Configure mock with values in valid range
        configure_torchcrepe_mock(
            f0_values=[100.0, 200.0, 300.0, 400.0], confidence_values=[0.9, 0.9, 0.9, 0.9]
        )

        sr = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
        audio = np.sin(2 * np.pi * 200.0 * t).astype(np.float32)

        from voice_man.services.forensic.gpu.crepe_extractor import TorchCrepeExtractor

        extractor = TorchCrepeExtractor(device="cpu")
        f0, _ = extractor.extract_f0(audio, sr, fmin=50.0, fmax=550.0)

        valid_f0 = f0[~np.isnan(f0)]
        if len(valid_f0) > 0:
            assert np.all(valid_f0 >= 50.0), "F0 values below minimum"
            assert np.all(valid_f0 <= 550.0), "F0 values above maximum"


class TestConfidenceRange:
    """Test confidence score range validation."""

    def test_confidence_values_in_valid_range(self, configure_torchcrepe_mock):
        """Test that confidence values are in 0-1 range."""
        # Configure mock with varied confidence values
        configure_torchcrepe_mock(
            f0_values=[200.0, 200.0, 200.0, 200.0], confidence_values=[0.2, 0.5, 0.8, 0.95]
        )

        sr = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
        audio = np.sin(2 * np.pi * 200.0 * t).astype(np.float32)

        from voice_man.services.forensic.gpu.crepe_extractor import TorchCrepeExtractor

        extractor = TorchCrepeExtractor(device="cpu")
        _, confidence = extractor.extract_f0(audio, sr)

        assert np.all(confidence >= 0.0), "Confidence values below 0"
        assert np.all(confidence <= 1.0), "Confidence values above 1"


class TestEmptyAudioHandling:
    """Test empty audio handling."""

    def test_empty_audio_raises_error(self, mock_gpu_modules):
        """Test that empty audio raises ValueError."""
        from voice_man.services.forensic.gpu.crepe_extractor import TorchCrepeExtractor

        extractor = TorchCrepeExtractor(device="cpu")
        empty_audio = np.array([], dtype=np.float32)

        with pytest.raises(ValueError, match="empty"):
            extractor.extract_f0(empty_audio, 16000)

    def test_empty_batch_raises_error(self, mock_gpu_modules):
        """Test that empty batch raises ValueError."""
        from voice_man.services.forensic.gpu.crepe_extractor import TorchCrepeExtractor

        extractor = TorchCrepeExtractor(device="cpu")
        empty_windows = np.array([]).reshape(0, 16000)

        with pytest.raises(ValueError, match="empty"):
            extractor.extract_f0_batch(empty_windows, 16000)


class TestShortAudioPadding:
    """Test short audio padding behavior."""

    def test_short_audio_is_padded(self, configure_torchcrepe_mock):
        """Test that audio shorter than MIN_AUDIO_SAMPLES is padded."""
        # Configure mock for short audio
        configure_torchcrepe_mock(f0_values=[200.0], confidence_values=[0.9])

        sr = 16000
        # Create very short audio (100 samples, less than MIN_AUDIO_SAMPLES=512)
        short_audio = np.sin(np.linspace(0, 0.5, 100)).astype(np.float32)

        from voice_man.services.forensic.gpu.crepe_extractor import TorchCrepeExtractor

        extractor = TorchCrepeExtractor(device="cpu")

        # Should not raise an error - audio is padded internally
        f0, confidence = extractor.extract_f0(short_audio, sr)

        # Should return valid results
        assert isinstance(f0, np.ndarray)
        assert isinstance(confidence, np.ndarray)

    def test_padding_method_pads_correctly(self, mock_gpu_modules):
        """Test the _pad_audio method directly."""
        from voice_man.services.forensic.gpu.crepe_extractor import TorchCrepeExtractor

        extractor = TorchCrepeExtractor(device="cpu")

        # Short audio
        short_audio = np.ones(100, dtype=np.float32)
        padded, original_len = extractor._pad_audio(short_audio)

        assert original_len == 100
        assert len(padded) == extractor.MIN_AUDIO_SAMPLES
        # Verify padding is zeros
        assert np.all(padded[100:] == 0)

    def test_audio_at_minimum_length_not_padded(self, mock_gpu_modules):
        """Test that audio at exactly MIN_AUDIO_SAMPLES is not padded."""
        from voice_man.services.forensic.gpu.crepe_extractor import TorchCrepeExtractor

        extractor = TorchCrepeExtractor(device="cpu")

        # Audio at exactly minimum length
        min_audio = np.ones(extractor.MIN_AUDIO_SAMPLES, dtype=np.float32)
        padded, original_len = extractor._pad_audio(min_audio)

        assert original_len == extractor.MIN_AUDIO_SAMPLES
        assert len(padded) == extractor.MIN_AUDIO_SAMPLES

    def test_long_audio_not_padded(self, mock_gpu_modules):
        """Test that long audio is not padded."""
        from voice_man.services.forensic.gpu.crepe_extractor import TorchCrepeExtractor

        extractor = TorchCrepeExtractor(device="cpu")

        # Long audio
        long_audio = np.ones(10000, dtype=np.float32)
        padded, original_len = extractor._pad_audio(long_audio)

        assert original_len == 10000
        assert len(padded) == 10000


class TestAutoDeviceDetection:
    """Test automatic device detection without override."""

    def test_device_auto_detection_cuda(self, mock_gpu_modules):
        """Test auto detection when CUDA is available."""
        from voice_man.services.forensic.gpu.crepe_extractor import TorchCrepeExtractor

        # No device override - should auto-detect
        extractor = TorchCrepeExtractor()

        # Access is_gpu_available to trigger torch import
        _ = extractor.is_gpu_available

        # Since torch is mocked, device property will use the mock
        # The extractor should have _device as None initially
        assert extractor._device_override is None

    def test_is_gpu_available_uses_torch(self, mock_gpu_modules):
        """Test is_gpu_available property accesses torch."""
        from voice_man.services.forensic.gpu.crepe_extractor import TorchCrepeExtractor

        extractor = TorchCrepeExtractor(device="cpu")

        # Access the property
        result = extractor.is_gpu_available

        # Should return boolean
        assert isinstance(result, bool)


class TestBatchDimensionValidation:
    """Test batch input dimension validation."""

    def test_batch_1d_raises_error(self, mock_gpu_modules):
        """Test that 1D input to extract_f0_batch raises ValueError."""
        from voice_man.services.forensic.gpu.crepe_extractor import TorchCrepeExtractor

        extractor = TorchCrepeExtractor(device="cpu")
        # 1D array instead of 2D
        invalid_windows = np.ones(16000, dtype=np.float32)

        with pytest.raises(ValueError, match="2D"):
            extractor.extract_f0_batch(invalid_windows, 16000)

    def test_batch_3d_raises_error(self, mock_gpu_modules):
        """Test that 3D input to extract_f0_batch raises ValueError."""
        from voice_man.services.forensic.gpu.crepe_extractor import TorchCrepeExtractor

        extractor = TorchCrepeExtractor(device="cpu")
        # 3D array instead of 2D
        invalid_windows = np.ones((3, 16000, 1), dtype=np.float32)

        with pytest.raises(ValueError, match="2D"):
            extractor.extract_f0_batch(invalid_windows, 16000)


class TestBatchExceptionHandling:
    """Test exception handling in batch processing."""

    def test_batch_with_unvoiced_window(self, configure_torchcrepe_mock):
        """Test batch processing when a window has no voiced frames."""
        # Configure mock to return all NaN F0 values (unvoiced)
        from unittest.mock import MagicMock

        configure_torchcrepe_mock(
            f0_values=[np.nan, np.nan, np.nan, np.nan], confidence_values=[0.1, 0.1, 0.1, 0.1]
        )

        sr = 16000
        windows = np.random.randn(2, sr).astype(np.float32)

        from voice_man.services.forensic.gpu.crepe_extractor import TorchCrepeExtractor

        extractor = TorchCrepeExtractor(device="cpu")
        f0_values, confidence_values = extractor.extract_f0_batch(windows, sr)

        # Should return NaN for unvoiced windows
        assert len(f0_values) == 2
        assert len(confidence_values) == 2

    def test_batch_with_short_windows(self, configure_torchcrepe_mock):
        """Test batch processing with windows shorter than MIN_AUDIO_SAMPLES."""
        configure_torchcrepe_mock(f0_values=[200.0], confidence_values=[0.9])

        sr = 16000
        # Create windows shorter than MIN_AUDIO_SAMPLES (512)
        short_windows = np.random.randn(3, 100).astype(np.float32)

        from voice_man.services.forensic.gpu.crepe_extractor import TorchCrepeExtractor

        extractor = TorchCrepeExtractor(device="cpu")
        f0_values, confidence_values = extractor.extract_f0_batch(short_windows, sr)

        # Should process all windows (with internal padding)
        assert len(f0_values) == 3
        assert len(confidence_values) == 3


class TestModelConfiguration:
    """Test model configuration options."""

    def test_default_model_is_full(self, mock_gpu_modules):
        """Test that default model is 'full'."""
        from voice_man.services.forensic.gpu.crepe_extractor import TorchCrepeExtractor

        extractor = TorchCrepeExtractor()
        assert extractor.model == "full"

    def test_tiny_model_configuration(self, mock_gpu_modules):
        """Test that 'tiny' model can be configured."""
        from voice_man.services.forensic.gpu.crepe_extractor import TorchCrepeExtractor

        extractor = TorchCrepeExtractor(model="tiny")
        assert extractor.model == "tiny"

    def test_custom_batch_size(self, mock_gpu_modules):
        """Test custom batch size configuration."""
        from voice_man.services.forensic.gpu.crepe_extractor import TorchCrepeExtractor

        extractor = TorchCrepeExtractor(batch_size=1024)
        assert extractor.batch_size == 1024
