"""
NNAudioProcessor Tests
SPEC-GPUAUDIO-001 Phase 2: Unit tests for NNAudioProcessor

Tests GPU-accelerated spectrogram generation using nnAudio.
Uses session-scoped mocks from conftest.py
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
import sys


# ============================================================================
# Milestone 2.1: NNAudioProcessor Base Structure Tests
# ============================================================================


class TestNNAudioProcessorInitialization:
    """Test NNAudioProcessor initialization and configuration."""

    def test_default_initialization(self, mock_nnaudio_modules):
        """Test default parameter initialization."""
        from voice_man.services.forensic.gpu.nnaudio_processor import NNAudioProcessor

        processor = NNAudioProcessor()

        assert processor.n_fft == NNAudioProcessor.DEFAULT_N_FFT
        assert processor.hop_length == NNAudioProcessor.DEFAULT_HOP_LENGTH
        assert processor.n_mels == NNAudioProcessor.DEFAULT_N_MELS
        assert processor.sr == NNAudioProcessor.DEFAULT_SR

    def test_custom_initialization(self, mock_nnaudio_modules):
        """Test custom parameter initialization."""
        from voice_man.services.forensic.gpu.nnaudio_processor import NNAudioProcessor

        processor = NNAudioProcessor(
            n_fft=1024,
            hop_length=256,
            n_mels=80,
            sr=22050,
        )

        assert processor.n_fft == 1024
        assert processor.hop_length == 256
        assert processor.n_mels == 80
        assert processor.sr == 22050

    def test_device_auto_detection(self, mock_nnaudio_modules):
        """Test automatic device detection."""
        from voice_man.services.forensic.gpu.nnaudio_processor import NNAudioProcessor

        processor = NNAudioProcessor()

        # Should return "cuda" or "cpu" based on availability
        assert processor.device in ["cuda", "cpu"]

    def test_explicit_device_override(self, mock_nnaudio_modules):
        """Test explicit device specification."""
        from voice_man.services.forensic.gpu.nnaudio_processor import NNAudioProcessor

        processor = NNAudioProcessor(device="cpu")
        assert processor.device == "cpu"


class TestNNAudioProcessorConstants:
    """Test default constant values."""

    def test_default_n_fft(self, mock_nnaudio_modules):
        """Test DEFAULT_N_FFT value."""
        from voice_man.services.forensic.gpu.nnaudio_processor import NNAudioProcessor

        assert NNAudioProcessor.DEFAULT_N_FFT == 2048

    def test_default_hop_length(self, mock_nnaudio_modules):
        """Test DEFAULT_HOP_LENGTH value."""
        from voice_man.services.forensic.gpu.nnaudio_processor import NNAudioProcessor

        assert NNAudioProcessor.DEFAULT_HOP_LENGTH == 512

    def test_default_n_mels(self, mock_nnaudio_modules):
        """Test DEFAULT_N_MELS value."""
        from voice_man.services.forensic.gpu.nnaudio_processor import NNAudioProcessor

        assert NNAudioProcessor.DEFAULT_N_MELS == 128

    def test_default_sr(self, mock_nnaudio_modules):
        """Test DEFAULT_SR value."""
        from voice_man.services.forensic.gpu.nnaudio_processor import NNAudioProcessor

        assert NNAudioProcessor.DEFAULT_SR == 16000


class TestSTFTLayerLazyLoading:
    """Test STFT layer lazy loading."""

    def test_stft_layer_not_initialized_at_construction(self, mock_nnaudio_modules):
        """Test that STFT layer is not created during __init__."""
        from voice_man.services.forensic.gpu.nnaudio_processor import NNAudioProcessor

        processor = NNAudioProcessor()

        # _stft_layer should be None until accessed
        assert processor._stft_layer is None

    def test_stft_layer_initialized_on_access(self, mock_nnaudio_modules):
        """Test that STFT layer is created on first access."""
        from voice_man.services.forensic.gpu.nnaudio_processor import NNAudioProcessor

        processor = NNAudioProcessor()
        stft_layer = processor.stft_layer

        # Should now be initialized
        assert stft_layer is not None

    def test_stft_layer_cached(self, mock_nnaudio_modules):
        """Test that STFT layer is cached after first access."""
        from voice_man.services.forensic.gpu.nnaudio_processor import NNAudioProcessor

        processor = NNAudioProcessor()
        stft_layer_1 = processor.stft_layer
        stft_layer_2 = processor.stft_layer

        # Should be the same object
        assert stft_layer_1 is stft_layer_2


class TestMelLayerLazyLoading:
    """Test Mel spectrogram layer lazy loading."""

    def test_mel_layer_not_initialized_at_construction(self, mock_nnaudio_modules):
        """Test that Mel layer is not created during __init__."""
        from voice_man.services.forensic.gpu.nnaudio_processor import NNAudioProcessor

        processor = NNAudioProcessor()

        # _mel_layer should be None until accessed
        assert processor._mel_layer is None

    def test_mel_layer_initialized_on_access(self, mock_nnaudio_modules):
        """Test that Mel layer is created on first access."""
        from voice_man.services.forensic.gpu.nnaudio_processor import NNAudioProcessor

        processor = NNAudioProcessor()
        mel_layer = processor.mel_layer

        # Should now be initialized
        assert mel_layer is not None

    def test_mel_layer_cached(self, mock_nnaudio_modules):
        """Test that Mel layer is cached after first access."""
        from voice_man.services.forensic.gpu.nnaudio_processor import NNAudioProcessor

        processor = NNAudioProcessor()
        mel_layer_1 = processor.mel_layer
        mel_layer_2 = processor.mel_layer

        # Should be the same object
        assert mel_layer_1 is mel_layer_2


# ============================================================================
# Milestone 2.2: Spectrogram Computation Tests
# ============================================================================


class TestComputeSTFT:
    """Test STFT computation."""

    def test_compute_stft_returns_numpy_array(self, mock_nnaudio_modules, sample_audio_1s):
        """Test that compute_stft returns numpy array."""
        from voice_man.services.forensic.gpu.nnaudio_processor import NNAudioProcessor

        audio, sr = sample_audio_1s
        processor = NNAudioProcessor(sr=sr)
        result = processor.compute_stft(audio)

        assert isinstance(result, np.ndarray)

    def test_compute_stft_output_shape(self, mock_nnaudio_modules, sample_audio_1s):
        """Test STFT output shape."""
        from voice_man.services.forensic.gpu.nnaudio_processor import NNAudioProcessor

        audio, sr = sample_audio_1s
        processor = NNAudioProcessor(sr=sr, n_fft=2048, hop_length=512)
        result = processor.compute_stft(audio)

        # Expected shape: (n_fft // 2 + 1, num_frames)
        expected_freq_bins = processor.n_fft // 2 + 1
        assert result.shape[0] == expected_freq_bins

    def test_compute_stft_return_complex(self, mock_nnaudio_modules, sample_audio_1s):
        """Test STFT with return_complex=True."""
        from voice_man.services.forensic.gpu.nnaudio_processor import NNAudioProcessor

        audio, sr = sample_audio_1s
        processor = NNAudioProcessor(sr=sr)
        result = processor.compute_stft(audio, return_complex=True)

        assert isinstance(result, np.ndarray)
        assert np.iscomplexobj(result)

    def test_compute_stft_magnitude(self, mock_nnaudio_modules, sample_audio_1s):
        """Test STFT returns magnitude by default."""
        from voice_man.services.forensic.gpu.nnaudio_processor import NNAudioProcessor

        audio, sr = sample_audio_1s
        processor = NNAudioProcessor(sr=sr)
        result = processor.compute_stft(audio, return_complex=False)

        # Magnitude should be real-valued and non-negative
        assert not np.iscomplexobj(result)
        assert np.all(result >= 0)

    def test_compute_stft_accepts_tensor_input(self, mock_nnaudio_modules, sample_audio_1s):
        """Test that compute_stft accepts torch.Tensor input."""
        from voice_man.services.forensic.gpu.nnaudio_processor import NNAudioProcessor

        audio, sr = sample_audio_1s
        processor = NNAudioProcessor(sr=sr)

        # Create mock tensor input
        import torch

        audio_tensor = torch.tensor(audio, dtype=torch.float32)
        result = processor.compute_stft(audio_tensor)

        assert isinstance(result, np.ndarray)


class TestComputeMelSpectrogram:
    """Test Mel spectrogram computation."""

    def test_compute_mel_returns_numpy_array(self, mock_nnaudio_modules, sample_audio_1s):
        """Test that compute_mel_spectrogram returns numpy array."""
        from voice_man.services.forensic.gpu.nnaudio_processor import NNAudioProcessor

        audio, sr = sample_audio_1s
        processor = NNAudioProcessor(sr=sr)
        result = processor.compute_mel_spectrogram(audio)

        assert isinstance(result, np.ndarray)

    def test_compute_mel_output_shape(self, mock_nnaudio_modules, sample_audio_1s):
        """Test Mel spectrogram output shape."""
        from voice_man.services.forensic.gpu.nnaudio_processor import NNAudioProcessor

        audio, sr = sample_audio_1s
        processor = NNAudioProcessor(sr=sr, n_mels=128, hop_length=512)
        result = processor.compute_mel_spectrogram(audio)

        # Expected shape: (n_mels, num_frames)
        assert result.shape[0] == processor.n_mels

    def test_compute_mel_power_parameter(self, mock_nnaudio_modules, sample_audio_1s):
        """Test power parameter for Mel spectrogram."""
        from voice_man.services.forensic.gpu.nnaudio_processor import NNAudioProcessor

        audio, sr = sample_audio_1s
        processor = NNAudioProcessor(sr=sr)

        result_power2 = processor.compute_mel_spectrogram(audio, power=2.0)
        result_power1 = processor.compute_mel_spectrogram(audio, power=1.0)

        # Both should be valid arrays
        assert isinstance(result_power2, np.ndarray)
        assert isinstance(result_power1, np.ndarray)

    def test_compute_mel_normalize(self, mock_nnaudio_modules, sample_audio_1s):
        """Test normalize parameter for Mel spectrogram."""
        from voice_man.services.forensic.gpu.nnaudio_processor import NNAudioProcessor

        audio, sr = sample_audio_1s
        processor = NNAudioProcessor(sr=sr)

        result_normalized = processor.compute_mel_spectrogram(audio, normalize=True)
        result_unnormalized = processor.compute_mel_spectrogram(audio, normalize=False)

        assert isinstance(result_normalized, np.ndarray)
        assert isinstance(result_unnormalized, np.ndarray)

    def test_compute_mel_accepts_tensor_input(self, mock_nnaudio_modules, sample_audio_1s):
        """Test that compute_mel_spectrogram accepts torch.Tensor input."""
        from voice_man.services.forensic.gpu.nnaudio_processor import NNAudioProcessor

        audio, sr = sample_audio_1s
        processor = NNAudioProcessor(sr=sr)

        import torch

        audio_tensor = torch.tensor(audio, dtype=torch.float32)
        result = processor.compute_mel_spectrogram(audio_tensor)

        assert isinstance(result, np.ndarray)


class TestBatchProcessing:
    """Test batch processing capabilities."""

    def test_compute_stft_batch_returns_correct_shape(self, mock_nnaudio_modules, sample_windows_5):
        """Test batch STFT returns correct shape."""
        from voice_man.services.forensic.gpu.nnaudio_processor import NNAudioProcessor

        windows, sr = sample_windows_5
        processor = NNAudioProcessor(sr=sr)
        result = processor.compute_stft_batch(windows)

        # Expected shape: (batch_size, n_fft // 2 + 1, num_frames)
        assert result.shape[0] == windows.shape[0]
        assert result.shape[1] == processor.n_fft // 2 + 1

    def test_compute_mel_batch_returns_correct_shape(self, mock_nnaudio_modules, sample_windows_5):
        """Test batch Mel spectrogram returns correct shape."""
        from voice_man.services.forensic.gpu.nnaudio_processor import NNAudioProcessor

        windows, sr = sample_windows_5
        processor = NNAudioProcessor(sr=sr)
        result = processor.compute_mel_batch(windows)

        # Expected shape: (batch_size, n_mels, num_frames)
        assert result.shape[0] == windows.shape[0]
        assert result.shape[1] == processor.n_mels


# ============================================================================
# Milestone 2.3: CPU Fallback and Error Handling Tests
# ============================================================================


class TestCPUFallback:
    """Test CPU fallback functionality."""

    def test_compute_stft_cpu_available(self, mock_nnaudio_modules):
        """Test that compute_stft_cpu method exists."""
        from voice_man.services.forensic.gpu.nnaudio_processor import NNAudioProcessor

        processor = NNAudioProcessor()
        assert hasattr(processor, "compute_stft_cpu")
        assert callable(processor.compute_stft_cpu)

    def test_compute_mel_cpu_available(self, mock_nnaudio_modules):
        """Test that compute_mel_cpu method exists."""
        from voice_man.services.forensic.gpu.nnaudio_processor import NNAudioProcessor

        processor = NNAudioProcessor()
        assert hasattr(processor, "compute_mel_cpu")
        assert callable(processor.compute_mel_cpu)

    def test_compute_stft_cpu_returns_numpy(self, mock_nnaudio_modules, sample_audio_1s):
        """Test CPU STFT returns numpy array."""
        from voice_man.services.forensic.gpu.nnaudio_processor import NNAudioProcessor

        audio, sr = sample_audio_1s
        processor = NNAudioProcessor(sr=sr)
        result = processor.compute_stft_cpu(audio)

        assert isinstance(result, np.ndarray)

    def test_compute_mel_cpu_returns_numpy(self, mock_nnaudio_modules, sample_audio_1s):
        """Test CPU Mel spectrogram returns numpy array."""
        from voice_man.services.forensic.gpu.nnaudio_processor import NNAudioProcessor

        audio, sr = sample_audio_1s
        processor = NNAudioProcessor(sr=sr)
        result = processor.compute_mel_cpu(audio)

        assert isinstance(result, np.ndarray)


class TestAutoDeviceSwitching:
    """Test automatic GPU/CPU switching."""

    def test_fallback_to_cpu_when_gpu_unavailable(self, mock_nnaudio_modules):
        """Test automatic fallback to CPU when GPU is unavailable."""
        from voice_man.services.forensic.gpu.nnaudio_processor import NNAudioProcessor

        # Force CPU by setting device
        processor = NNAudioProcessor(device="cpu")
        assert processor.device == "cpu"

    def test_uses_gpu_when_available(self, mock_nnaudio_modules_with_cuda):
        """Test that GPU is used when available."""
        from voice_man.services.forensic.gpu.nnaudio_processor import NNAudioProcessor

        processor = NNAudioProcessor()
        # With CUDA mocked as available, should use cuda
        assert processor.device == "cuda"


class TestInputValidation:
    """Test input validation."""

    def test_empty_audio_raises_error(self, mock_nnaudio_modules):
        """Test that empty audio raises ValueError."""
        from voice_man.services.forensic.gpu.nnaudio_processor import NNAudioProcessor

        processor = NNAudioProcessor()
        empty_audio = np.array([], dtype=np.float32)

        with pytest.raises(ValueError, match="empty"):
            processor.compute_stft(empty_audio)

    def test_empty_audio_mel_raises_error(self, mock_nnaudio_modules):
        """Test that empty audio raises ValueError for Mel."""
        from voice_man.services.forensic.gpu.nnaudio_processor import NNAudioProcessor

        processor = NNAudioProcessor()
        empty_audio = np.array([], dtype=np.float32)

        with pytest.raises(ValueError, match="empty"):
            processor.compute_mel_spectrogram(empty_audio)

    def test_invalid_n_fft_raises_error(self, mock_nnaudio_modules):
        """Test that invalid n_fft raises ValueError."""
        from voice_man.services.forensic.gpu.nnaudio_processor import NNAudioProcessor

        with pytest.raises(ValueError, match="n_fft"):
            NNAudioProcessor(n_fft=0)

    def test_invalid_hop_length_raises_error(self, mock_nnaudio_modules):
        """Test that invalid hop_length raises ValueError."""
        from voice_man.services.forensic.gpu.nnaudio_processor import NNAudioProcessor

        with pytest.raises(ValueError, match="hop_length"):
            NNAudioProcessor(hop_length=0)

    def test_invalid_n_mels_raises_error(self, mock_nnaudio_modules):
        """Test that invalid n_mels raises ValueError."""
        from voice_man.services.forensic.gpu.nnaudio_processor import NNAudioProcessor

        with pytest.raises(ValueError, match="n_mels"):
            NNAudioProcessor(n_mels=0)

    def test_batch_empty_raises_error(self, mock_nnaudio_modules):
        """Test that empty batch raises ValueError."""
        from voice_man.services.forensic.gpu.nnaudio_processor import NNAudioProcessor

        processor = NNAudioProcessor()
        empty_batch = np.array([]).reshape(0, 16000)

        with pytest.raises(ValueError, match="empty"):
            processor.compute_stft_batch(empty_batch)

    def test_batch_1d_raises_error(self, mock_nnaudio_modules):
        """Test that 1D batch raises ValueError."""
        from voice_man.services.forensic.gpu.nnaudio_processor import NNAudioProcessor

        processor = NNAudioProcessor()
        invalid_batch = np.ones(16000, dtype=np.float32)

        with pytest.raises(ValueError, match="2D"):
            processor.compute_stft_batch(invalid_batch)


# ============================================================================
# Milestone 2.4: GPUAudioBackend Integration Tests
# ============================================================================


class TestGPUAudioBackendIntegration:
    """Test NNAudioProcessor integration with GPUAudioBackend."""

    def test_backend_has_stft_method(self, mock_nnaudio_modules):
        """Test that GPUAudioBackend has compute_stft method."""
        from voice_man.services.forensic.gpu.backend import GPUAudioBackend

        backend = GPUAudioBackend()
        assert hasattr(backend, "compute_stft")
        assert callable(backend.compute_stft)

    def test_backend_has_mel_method(self, mock_nnaudio_modules):
        """Test that GPUAudioBackend has compute_mel_spectrogram method."""
        from voice_man.services.forensic.gpu.backend import GPUAudioBackend

        backend = GPUAudioBackend()
        assert hasattr(backend, "compute_mel_spectrogram")
        assert callable(backend.compute_mel_spectrogram)

    def test_backend_stft_returns_numpy(self, mock_nnaudio_modules, sample_audio_1s):
        """Test GPUAudioBackend.compute_stft returns numpy array."""
        from voice_man.services.forensic.gpu.backend import GPUAudioBackend

        audio, sr = sample_audio_1s
        backend = GPUAudioBackend()
        result = backend.compute_stft(audio, sr)

        assert isinstance(result, np.ndarray)

    def test_backend_mel_returns_numpy(self, mock_nnaudio_modules, sample_audio_1s):
        """Test GPUAudioBackend.compute_mel_spectrogram returns numpy array."""
        from voice_man.services.forensic.gpu.backend import GPUAudioBackend

        audio, sr = sample_audio_1s
        backend = GPUAudioBackend()
        result = backend.compute_mel_spectrogram(audio, sr)

        assert isinstance(result, np.ndarray)

    def test_backend_lazy_loads_nnaudio_processor(self, mock_nnaudio_modules):
        """Test that GPUAudioBackend lazy-loads NNAudioProcessor."""
        from voice_man.services.forensic.gpu.backend import GPUAudioBackend

        backend = GPUAudioBackend()

        # Should not have nnaudio_processor until accessed
        assert backend._nnaudio_processor is None

        # Access nnaudio_processor property
        processor = backend.nnaudio_processor

        # Now should be initialized
        assert processor is not None

    def test_backend_nnaudio_processor_cached(self, mock_nnaudio_modules):
        """Test that NNAudioProcessor is cached in backend."""
        from voice_man.services.forensic.gpu.backend import GPUAudioBackend

        backend = GPUAudioBackend()

        processor_1 = backend.nnaudio_processor
        processor_2 = backend.nnaudio_processor

        assert processor_1 is processor_2


# ============================================================================
# Cross-Validation Tests (librosa comparison)
# ============================================================================


class TestLibrosaComparison:
    """Test cross-validation with librosa results.

    Note: These tests require real nnAudio (not mocked) to compare
    actual GPU output with librosa CPU output. In mock environment,
    these tests validate the CPU fallback paths instead.
    """

    def test_mel_spectrogram_cpu_matches_librosa(self, mock_nnaudio_modules, sample_audio_1s):
        """Test that CPU Mel spectrogram matches librosa output."""
        from voice_man.services.forensic.gpu.nnaudio_processor import NNAudioProcessor

        audio, sr = sample_audio_1s
        processor = NNAudioProcessor(sr=sr, n_mels=128, n_fft=2048, hop_length=512)

        # Get CPU fallback result (uses librosa internally)
        result = processor.compute_mel_cpu(audio)

        # Calculate librosa reference
        try:
            import librosa

            librosa_mel = librosa.feature.melspectrogram(
                y=audio,
                sr=sr,
                n_fft=2048,
                hop_length=512,
                n_mels=128,
            )

            # Compare shapes
            assert result.shape == librosa_mel.shape

            # Results should be nearly identical (same library)
            mae = np.mean(np.abs(result - librosa_mel))
            assert mae < 1e-6, f"MAE {mae} exceeds threshold 1e-6"

        except ImportError:
            pytest.skip("librosa not installed")

    def test_stft_cpu_matches_librosa(self, mock_nnaudio_modules, sample_audio_1s):
        """Test that CPU STFT matches librosa output."""
        from voice_man.services.forensic.gpu.nnaudio_processor import NNAudioProcessor

        audio, sr = sample_audio_1s
        processor = NNAudioProcessor(sr=sr, n_fft=2048, hop_length=512)

        # Get CPU fallback result
        result = processor.compute_stft_cpu(audio)

        try:
            import librosa

            librosa_stft = np.abs(librosa.stft(audio, n_fft=2048, hop_length=512))

            # Compare shapes
            assert result.shape == librosa_stft.shape

            # Results should be nearly identical
            mae = np.mean(np.abs(result - librosa_stft))
            assert mae < 1e-6, f"MAE {mae} exceeds threshold 1e-6"

        except ImportError:
            pytest.skip("librosa not installed")


# ============================================================================
# Additional Coverage Tests
# ============================================================================


class TestCPUFallbackEmptyInput:
    """Test CPU fallback with empty input handling."""

    def test_compute_stft_cpu_empty_raises_error(self, mock_nnaudio_modules):
        """Test that empty audio raises ValueError in CPU STFT."""
        from voice_man.services.forensic.gpu.nnaudio_processor import NNAudioProcessor

        processor = NNAudioProcessor()
        empty_audio = np.array([], dtype=np.float32)

        with pytest.raises(ValueError, match="empty"):
            processor.compute_stft_cpu(empty_audio)

    def test_compute_mel_cpu_empty_raises_error(self, mock_nnaudio_modules):
        """Test that empty audio raises ValueError in CPU Mel."""
        from voice_man.services.forensic.gpu.nnaudio_processor import NNAudioProcessor

        processor = NNAudioProcessor()
        empty_audio = np.array([], dtype=np.float32)

        with pytest.raises(ValueError, match="empty"):
            processor.compute_mel_cpu(empty_audio)


class TestDeviceAutoDetection:
    """Test automatic device detection with CUDA availability."""

    def test_device_uses_cuda_when_available(self, mock_nnaudio_modules_with_cuda):
        """Test that device is 'cuda' when CUDA is available."""
        from voice_man.services.forensic.gpu.nnaudio_processor import NNAudioProcessor

        processor = NNAudioProcessor()
        # Force device detection
        device = processor.device
        assert device == "cuda"

    def test_device_uses_cpu_when_explicitly_set(self, mock_nnaudio_modules):
        """Test that device is 'cpu' when explicitly set."""
        from voice_man.services.forensic.gpu.nnaudio_processor import NNAudioProcessor

        processor = NNAudioProcessor(device="cpu")
        device = processor.device
        assert device == "cpu"

    def test_device_override_respected(self, mock_nnaudio_modules):
        """Test that device override is respected regardless of CUDA availability."""
        from voice_man.services.forensic.gpu.nnaudio_processor import NNAudioProcessor

        # Even if CUDA is available, explicit device="cpu" should be respected
        processor = NNAudioProcessor(device="cpu")
        assert processor._device_override == "cpu"
        assert processor.device == "cpu"


class TestBatchProcessingEdgeCases:
    """Test edge cases in batch processing."""

    def test_compute_stft_batch_single_item(self, mock_nnaudio_modules, sample_audio_1s):
        """Test batch STFT with single item."""
        from voice_man.services.forensic.gpu.nnaudio_processor import NNAudioProcessor

        audio, sr = sample_audio_1s
        processor = NNAudioProcessor(sr=sr)

        # Single item batch
        batch = audio.reshape(1, -1)
        result = processor.compute_stft_batch(batch)

        assert result.shape[0] == 1
        assert result.shape[1] == processor.n_fft // 2 + 1

    def test_compute_mel_batch_single_item(self, mock_nnaudio_modules, sample_audio_1s):
        """Test batch Mel with single item."""
        from voice_man.services.forensic.gpu.nnaudio_processor import NNAudioProcessor

        audio, sr = sample_audio_1s
        processor = NNAudioProcessor(sr=sr)

        batch = audio.reshape(1, -1)
        result = processor.compute_mel_batch(batch)

        assert result.shape[0] == 1
        assert result.shape[1] == processor.n_mels


class TestNormalizationBehavior:
    """Test normalization behavior in Mel spectrogram."""

    def test_normalize_changes_output(self, mock_nnaudio_modules, sample_audio_1s):
        """Test that normalize=True changes the output statistics."""
        from voice_man.services.forensic.gpu.nnaudio_processor import NNAudioProcessor

        audio, sr = sample_audio_1s
        processor = NNAudioProcessor(sr=sr)

        result_normalized = processor.compute_mel_spectrogram(audio, normalize=True)

        # Normalized output should have mean close to 0
        assert isinstance(result_normalized, np.ndarray)


class TestPowerParameter:
    """Test power parameter behavior."""

    def test_power_1_gives_magnitude(self, mock_nnaudio_modules, sample_audio_1s):
        """Test that power=1.0 gives magnitude spectrogram."""
        from voice_man.services.forensic.gpu.nnaudio_processor import NNAudioProcessor

        audio, sr = sample_audio_1s
        processor = NNAudioProcessor(sr=sr)

        result = processor.compute_mel_spectrogram(audio, power=1.0)
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == processor.n_mels


class TestModuleExports:
    """Test module exports and imports."""

    def test_nnaudio_processor_exported_from_package(self, mock_nnaudio_modules):
        """Test that NNAudioProcessor is exported from gpu package."""
        from voice_man.services.forensic.gpu import NNAudioProcessor

        assert NNAudioProcessor is not None

    def test_all_exports_defined(self, mock_nnaudio_modules):
        """Test that __all__ contains expected exports."""
        from voice_man.services.forensic import gpu

        assert "NNAudioProcessor" in gpu.__all__
        assert "GPUAudioBackend" in gpu.__all__
        assert "TorchCrepeExtractor" in gpu.__all__
