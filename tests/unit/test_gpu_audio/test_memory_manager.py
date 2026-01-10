"""
GPU Memory Manager Tests
SPEC-GPUAUDIO-001: Unit tests for GPUMemoryManager

Tests for dynamic batch size adjustment and GPU memory monitoring.
"""

import pytest
from unittest.mock import patch, MagicMock


class TestMemoryManagerConfiguration:
    """Test memory manager configuration."""

    def test_default_configuration(self, mock_gpu_modules):
        """Test default configuration values."""
        from voice_man.services.forensic.gpu.memory_manager import GPUMemoryManager

        manager = GPUMemoryManager()

        assert manager.min_batch_size == GPUMemoryManager.DEFAULT_MIN_BATCH_SIZE
        assert manager.max_batch_size == GPUMemoryManager.DEFAULT_MAX_BATCH_SIZE
        assert manager.memory_threshold_gb == GPUMemoryManager.DEFAULT_MEMORY_THRESHOLD_GB

    def test_custom_configuration(self, mock_gpu_modules):
        """Test custom configuration values."""
        from voice_man.services.forensic.gpu.memory_manager import GPUMemoryManager

        manager = GPUMemoryManager(
            min_batch_size=64,
            max_batch_size=2048,
            memory_threshold_gb=4.0,
        )

        assert manager.min_batch_size == 64
        assert manager.max_batch_size == 2048
        assert manager.memory_threshold_gb == 4.0


class TestBatchSizeManagement:
    """Test batch size management."""

    def test_current_batch_size_initial(self, mock_gpu_modules):
        """Test initial batch size is max_batch_size."""
        from voice_man.services.forensic.gpu.memory_manager import GPUMemoryManager

        manager = GPUMemoryManager(max_batch_size=2048)

        assert manager.current_batch_size == 2048

    def test_reset_batch_size(self, mock_gpu_modules):
        """Test reset_batch_size restores to maximum."""
        from voice_man.services.forensic.gpu.memory_manager import GPUMemoryManager

        manager = GPUMemoryManager(max_batch_size=2048)
        manager._current_batch_size = 512

        manager.reset_batch_size()

        assert manager.current_batch_size == 2048


class TestGPUMemoryMonitoring:
    """Test GPU memory monitoring."""

    def test_get_available_memory_no_gpu(self, mock_gpu_modules):
        """Test get_available_memory_gb returns 0 when no GPU."""
        from voice_man.services.forensic.gpu.memory_manager import GPUMemoryManager

        # Mock torch.cuda.is_available to return False
        with patch("torch.cuda.is_available", return_value=False):
            manager = GPUMemoryManager()
            memory = manager.get_available_memory_gb()

            assert memory == 0.0

    def test_get_available_memory_with_gpu(self, mock_gpu_modules):
        """Test get_available_memory_gb with GPU available."""
        from voice_man.services.forensic.gpu.memory_manager import GPUMemoryManager

        # Mock torch.cuda with memory info
        mock_props = MagicMock()
        mock_props.total_memory = 8 * (1024**3)  # 8GB total

        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.get_device_properties", return_value=mock_props):
                with patch("torch.cuda.memory_allocated", return_value=2 * (1024**3)):  # 2GB used
                    manager = GPUMemoryManager()
                    memory = manager.get_available_memory_gb()

                    # Should return approximately 6GB available
                    assert memory == pytest.approx(6.0, rel=0.1)

    def test_get_available_memory_handles_exception(self, mock_gpu_modules):
        """Test get_available_memory_gb handles exceptions gracefully."""
        from voice_man.services.forensic.gpu.memory_manager import GPUMemoryManager

        with patch("torch.cuda.is_available", side_effect=Exception("CUDA error")):
            manager = GPUMemoryManager()
            memory = manager.get_available_memory_gb()

            assert memory == 0.0


class TestBatchSizeAdjustment:
    """Test dynamic batch size adjustment."""

    def test_adjust_batch_size_low_memory(self, mock_gpu_modules):
        """Test batch size reduction when memory is low."""
        from voice_man.services.forensic.gpu.memory_manager import GPUMemoryManager

        manager = GPUMemoryManager(
            min_batch_size=128,
            max_batch_size=2048,
            memory_threshold_gb=2.0,
        )

        # Mock low GPU memory
        with patch.object(manager, "get_available_memory_gb", return_value=1.0):
            new_batch_size = manager.adjust_batch_size()

            # Should be reduced by 50%
            assert new_batch_size == 1024

    def test_adjust_batch_size_respects_minimum(self, mock_gpu_modules):
        """Test batch size does not go below minimum."""
        from voice_man.services.forensic.gpu.memory_manager import GPUMemoryManager

        manager = GPUMemoryManager(
            min_batch_size=128,
            max_batch_size=256,
            memory_threshold_gb=2.0,
        )

        # Set current batch size to minimum
        manager._current_batch_size = 128

        # Mock low GPU memory
        with patch.object(manager, "get_available_memory_gb", return_value=0.5):
            new_batch_size = manager.adjust_batch_size()

            # Should stay at minimum
            assert new_batch_size == 128

    def test_adjust_batch_size_sufficient_memory(self, mock_gpu_modules):
        """Test batch size not reduced when memory sufficient."""
        from voice_man.services.forensic.gpu.memory_manager import GPUMemoryManager

        manager = GPUMemoryManager(
            min_batch_size=128,
            max_batch_size=2048,
            memory_threshold_gb=2.0,
        )

        # Mock sufficient GPU memory
        with patch.object(manager, "get_available_memory_gb", return_value=4.0):
            new_batch_size = manager.adjust_batch_size()

            # Should stay at max
            assert new_batch_size == 2048
