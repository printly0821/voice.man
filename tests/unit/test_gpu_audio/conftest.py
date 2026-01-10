"""
Shared fixtures for GPU audio tests.
SPEC-GPUAUDIO-001: Provides mock fixtures for testing without GPU dependencies.
"""

import sys
import pytest
import numpy as np
from unittest.mock import MagicMock, patch


@pytest.fixture(autouse=True, scope="session")
def mock_gpu_modules():
    """
    Mock GPU-related modules at session level before any imports.

    This fixture mocks:
    - torchcrepe: GPU F0 extraction library
    - torch.cuda: CUDA availability checks

    The mocking is done at the session level to prevent import conflicts.
    """
    # Create mock torch module with cuda support
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.float32 = "float32"
    mock_torch.device.return_value = "cpu"

    # Mock tensor behavior
    mock_tensor = MagicMock()
    mock_tensor.unsqueeze.return_value = mock_tensor
    mock_tensor.to.return_value = mock_tensor
    mock_torch.tensor.return_value = mock_tensor

    # Create mock torchcrepe module
    mock_torchcrepe = MagicMock()

    # Default predict behavior
    mock_f0 = MagicMock()
    mock_f0.squeeze.return_value.cpu.return_value.numpy.return_value = np.array(
        [200.0, 200.0, 200.0, 200.0]
    )

    mock_periodicity = MagicMock()
    mock_periodicity.squeeze.return_value.cpu.return_value.numpy.return_value = np.array(
        [0.9, 0.9, 0.9, 0.9]
    )

    mock_torchcrepe.predict.return_value = (mock_f0, mock_periodicity)

    # Store original torchcrepe module if it exists
    original_torchcrepe = sys.modules.get("torchcrepe")

    # Inject mocks
    sys.modules["torchcrepe"] = mock_torchcrepe

    yield {
        "torch": mock_torch,
        "torchcrepe": mock_torchcrepe,
    }

    # Restore original modules if they existed
    if original_torchcrepe is not None:
        sys.modules["torchcrepe"] = original_torchcrepe
    elif "torchcrepe" in sys.modules:
        del sys.modules["torchcrepe"]


@pytest.fixture
def mock_cuda_available():
    """Fixture to mock CUDA as available."""
    with patch("torch.cuda.is_available", return_value=True):
        yield


@pytest.fixture
def mock_cuda_unavailable():
    """Fixture to mock CUDA as unavailable."""
    with patch("torch.cuda.is_available", return_value=False):
        yield


@pytest.fixture
def sample_audio_1s():
    """Generate a 1-second 200Hz sine wave test signal."""
    sr = 16000
    duration = 1.0
    frequency = 200.0
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)
    return audio, sr


@pytest.fixture
def sample_windows_5():
    """Generate 5 audio windows for batch testing."""
    sr = 16000
    window_duration = 1.0
    num_windows = 5
    frequency = 200.0

    windows = []
    for _ in range(num_windows):
        t = np.linspace(0, window_duration, int(sr * window_duration), dtype=np.float32)
        window = np.sin(2 * np.pi * frequency * t).astype(np.float32)
        windows.append(window)

    return np.array(windows), sr


@pytest.fixture
def configure_torchcrepe_mock(mock_gpu_modules):
    """
    Factory fixture to configure torchcrepe mock with custom return values.

    Usage:
        def test_something(configure_torchcrepe_mock):
            configure_torchcrepe_mock(
                f0_values=[100.0, 200.0],
                confidence_values=[0.9, 0.8]
            )
    """

    def _configure(f0_values=None, confidence_values=None):
        if f0_values is None:
            f0_values = [200.0, 200.0, 200.0, 200.0]
        if confidence_values is None:
            confidence_values = [0.9, 0.9, 0.9, 0.9]

        mock_f0 = MagicMock()
        mock_f0.squeeze.return_value.cpu.return_value.numpy.return_value = np.array(f0_values)

        mock_periodicity = MagicMock()
        mock_periodicity.squeeze.return_value.cpu.return_value.numpy.return_value = np.array(
            confidence_values
        )

        mock_gpu_modules["torchcrepe"].predict.return_value = (mock_f0, mock_periodicity)

    return _configure


# ============================================================================
# NNAudio Mock Fixtures for Phase 2
# ============================================================================


class MockSTFTLayer:
    """Mock STFT layer that properly handles .to() and __call__."""

    def __init__(self, n_fft=2048, hop_length=512, **kwargs):
        self.n_fft = n_fft
        self.hop_length = hop_length

    def to(self, device):
        """Return self for chaining."""
        return self

    def __call__(self, x):
        """Mock STFT forward pass - returns magnitude spectrogram."""
        # Determine audio length from input
        audio_len = 16000  # Default

        # Create mock output tensor with correct shape
        freq_bins = self.n_fft // 2 + 1
        time_frames = (audio_len - self.n_fft) // self.hop_length + 1
        time_frames = max(1, time_frames)

        # Create actual numpy output
        output_data = np.abs(np.random.randn(1, freq_bins, time_frames).astype(np.float32))

        # Create mock tensor with proper chain
        mock_output = MagicMock()
        mock_output.squeeze.return_value.cpu.return_value.numpy.return_value = output_data[0]
        return mock_output


class MockMelLayer:
    """Mock MelSpectrogram layer that properly handles .to() and __call__."""

    def __init__(self, n_fft=2048, hop_length=512, n_mels=128, **kwargs):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

    def to(self, device):
        """Return self for chaining."""
        return self

    def __call__(self, x):
        """Mock Mel spectrogram forward pass."""
        audio_len = 16000  # Default
        time_frames = (audio_len - self.n_fft) // self.hop_length + 1
        time_frames = max(1, time_frames)

        output_data = np.abs(np.random.randn(1, self.n_mels, time_frames).astype(np.float32))

        mock_output = MagicMock()
        mock_output.squeeze.return_value.cpu.return_value.numpy.return_value = output_data[0]
        return mock_output


@pytest.fixture
def mock_nnaudio_modules(mock_gpu_modules):
    """
    Mock nnAudio-related modules for NNAudioProcessor testing.

    This fixture mocks:
    - nnAudio.features.stft.STFT: GPU STFT layer
    - nnAudio.features.mel.MelSpectrogram: GPU Mel spectrogram layer
    - torch: PyTorch for tensor operations

    Uses the existing mock_gpu_modules as a dependency.
    """
    # Create mock nnAudio module structure
    mock_nnaudio = MagicMock()
    mock_nnaudio_features = MagicMock()
    mock_nnaudio_stft = MagicMock()
    mock_nnaudio_mel = MagicMock()

    # Use class-based mocks that handle .to() properly
    mock_nnaudio_stft.STFT = MockSTFTLayer
    mock_nnaudio_mel.MelSpectrogram = MockMelLayer

    # Set up module structure
    mock_nnaudio.features = mock_nnaudio_features
    mock_nnaudio_features.stft = mock_nnaudio_stft
    mock_nnaudio_features.mel = mock_nnaudio_mel

    # Store original modules
    original_nnaudio = sys.modules.get("nnAudio")
    original_nnaudio_features = sys.modules.get("nnAudio.features")
    original_nnaudio_stft = sys.modules.get("nnAudio.features.stft")
    original_nnaudio_mel = sys.modules.get("nnAudio.features.mel")

    # Inject mocks
    sys.modules["nnAudio"] = mock_nnaudio
    sys.modules["nnAudio.features"] = mock_nnaudio_features
    sys.modules["nnAudio.features.stft"] = mock_nnaudio_stft
    sys.modules["nnAudio.features.mel"] = mock_nnaudio_mel

    yield {
        "torch": mock_gpu_modules["torch"],
        "torchcrepe": mock_gpu_modules["torchcrepe"],
        "nnAudio": mock_nnaudio,
    }

    # Restore original modules
    if original_nnaudio is not None:
        sys.modules["nnAudio"] = original_nnaudio
    elif "nnAudio" in sys.modules:
        del sys.modules["nnAudio"]

    if original_nnaudio_features is not None:
        sys.modules["nnAudio.features"] = original_nnaudio_features
    elif "nnAudio.features" in sys.modules:
        del sys.modules["nnAudio.features"]

    if original_nnaudio_stft is not None:
        sys.modules["nnAudio.features.stft"] = original_nnaudio_stft
    elif "nnAudio.features.stft" in sys.modules:
        del sys.modules["nnAudio.features.stft"]

    if original_nnaudio_mel is not None:
        sys.modules["nnAudio.features.mel"] = original_nnaudio_mel
    elif "nnAudio.features.mel" in sys.modules:
        del sys.modules["nnAudio.features.mel"]


@pytest.fixture
def mock_nnaudio_modules_with_cuda(mock_nnaudio_modules):
    """
    Mock nnAudio modules with CUDA available.

    Extends mock_nnaudio_modules to simulate CUDA being available.
    """
    with patch("torch.cuda.is_available", return_value=True):
        yield mock_nnaudio_modules
