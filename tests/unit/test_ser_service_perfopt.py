"""
Unit tests for SER Service Performance Optimization (SPEC-PERFOPT-001).

Tests for:
- TASK-002: SER model class-level caching
- TASK-003: GPU-first device detection
- TASK-004: preload_models() method

TDD RED Phase: These tests should FAIL before implementation.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import numpy as np


class TestSERModelCaching:
    """Test SER model class-level caching (TASK-002)."""

    def test_model_cache_class_attribute_exists(self):
        """Test that _model_cache class attribute exists."""
        from voice_man.services.forensic.ser_service import SERService

        assert hasattr(SERService, "_model_cache"), (
            "SERService should have _model_cache class attribute"
        )

    def test_model_cache_is_dict(self):
        """Test that _model_cache is a dictionary."""
        from voice_man.services.forensic.ser_service import SERService

        assert isinstance(SERService._model_cache, dict), "_model_cache should be a dictionary"

    def test_get_or_load_model_method_exists(self):
        """Test that _get_or_load_model method exists."""
        from voice_man.services.forensic.ser_service import SERService

        service = SERService()
        assert hasattr(service, "_get_or_load_model"), (
            "SERService should have _get_or_load_model method"
        )

    def test_model_cache_persists_across_instances(self):
        """Test that model cache persists across SERService instances."""
        from voice_man.services.forensic.ser_service import SERService

        # Clear cache before test
        SERService._model_cache.clear()

        # Add a test entry to cache
        SERService._model_cache["test_model"] = "cached_value"

        # Create new instance - cache should persist as class attribute
        _ = SERService()

        # Cache should still contain the value
        assert "test_model" in SERService._model_cache, (
            "Model cache should persist across instances"
        )
        assert SERService._model_cache["test_model"] == "cached_value"

        # Cleanup
        SERService._model_cache.clear()

    def test_get_or_load_model_returns_cached_model(self):
        """Test that _get_or_load_model returns cached model if available."""
        from voice_man.services.forensic.ser_service import SERService

        # Clear cache before test
        SERService._model_cache.clear()

        service = SERService()

        # Add mock model to cache
        mock_model = MagicMock()
        SERService._model_cache["primary"] = mock_model

        # Should return cached model without loading
        result = service._get_or_load_model("primary")
        assert result is mock_model, "_get_or_load_model should return cached model"

        # Cleanup
        SERService._model_cache.clear()

    def test_clear_model_cache_method_exists(self):
        """Test that clear_model_cache class method exists."""
        from voice_man.services.forensic.ser_service import SERService

        assert hasattr(SERService, "clear_model_cache"), (
            "SERService should have clear_model_cache method"
        )

    def test_clear_model_cache_clears_all_models(self):
        """Test that clear_model_cache clears all cached models."""
        from voice_man.services.forensic.ser_service import SERService

        # Add test entries
        SERService._model_cache["model1"] = "value1"
        SERService._model_cache["model2"] = "value2"

        # Clear cache
        SERService.clear_model_cache()

        assert len(SERService._model_cache) == 0, "clear_model_cache should clear all cached models"


class TestGPUFirstDeviceDetection:
    """Test GPU-first device detection (TASK-003)."""

    def test_detect_optimal_device_method_exists(self):
        """Test that _detect_optimal_device method exists."""
        from voice_man.services.forensic.ser_service import SERService

        service = SERService()
        assert hasattr(service, "_detect_optimal_device"), (
            "SERService should have _detect_optimal_device method"
        )

    @patch("torch.cuda.is_available", return_value=True)
    def test_default_device_is_cuda_when_available(self, mock_cuda):
        """Test that default device is 'cuda' when GPU is available."""
        from voice_man.services.forensic.ser_service import SERService

        service = SERService()
        device = service._detect_optimal_device()

        assert device == "cuda", (
            f"Default device should be 'cuda' when GPU available, got '{device}'"
        )

    @patch("torch.cuda.is_available", return_value=False)
    def test_fallback_to_cpu_when_gpu_unavailable(self, mock_cuda):
        """Test that device falls back to 'cpu' when GPU is unavailable."""
        from voice_man.services.forensic.ser_service import SERService

        service = SERService()
        device = service._detect_optimal_device()

        assert device == "cpu", (
            f"Device should fall back to 'cpu' when GPU unavailable, got '{device}'"
        )

    def test_explicit_device_overrides_auto_detection(self):
        """Test that explicit device setting overrides auto-detection."""
        from voice_man.services.forensic.ser_service import SERService

        service = SERService(device="cpu")
        device = service._detect_optimal_device()

        assert device == "cpu", "Explicit 'cpu' device should override auto-detection"

    @patch("torch.cuda.is_available", return_value=True)
    def test_auto_device_detects_cuda(self, mock_cuda):
        """Test that 'auto' device setting correctly detects CUDA."""
        from voice_man.services.forensic.ser_service import SERService

        service = SERService(device="auto")
        device = service._detect_optimal_device()

        assert device == "cuda", "Auto device should detect CUDA when available"


class TestPreloadModels:
    """Test preload_models async method (TASK-004)."""

    def test_preload_models_method_exists(self):
        """Test that preload_models async method exists."""
        from voice_man.services.forensic.ser_service import SERService
        import asyncio

        service = SERService()
        assert hasattr(service, "preload_models"), "SERService should have preload_models method"
        # Check it's a coroutine function
        assert asyncio.iscoroutinefunction(service.preload_models), (
            "preload_models should be an async method"
        )

    @pytest.mark.asyncio
    async def test_preload_models_returns_stats(self):
        """Test that preload_models returns model load statistics."""
        from voice_man.services.forensic.ser_service import SERService

        service = SERService()

        # Mock the model loading to avoid actual model downloads
        with patch.object(service, "_load_primary_model"):
            with patch.object(service, "_load_secondary_model"):
                stats = await service.preload_models()

        assert isinstance(stats, dict), "preload_models should return a dict"
        assert "load_time_seconds" in stats, "Stats should include 'load_time_seconds'"
        assert "models_loaded" in stats, "Stats should include 'models_loaded'"

    @pytest.mark.asyncio
    async def test_preload_models_loads_primary_model(self):
        """Test that preload_models loads the primary model."""
        from voice_man.services.forensic.ser_service import SERService

        service = SERService()

        with patch.object(service, "_load_primary_model") as mock_primary:
            with patch.object(service, "_load_secondary_model"):
                await service.preload_models()

        mock_primary.assert_called_once()

    @pytest.mark.asyncio
    async def test_preload_models_loads_secondary_model_when_ensemble(self):
        """Test that preload_models loads secondary model when use_ensemble=True."""
        from voice_man.services.forensic.ser_service import SERService

        service = SERService(use_ensemble=True)

        with patch.object(service, "_load_primary_model"):
            with patch.object(service, "_load_secondary_model") as mock_secondary:
                await service.preload_models()

        mock_secondary.assert_called_once()

    @pytest.mark.asyncio
    async def test_preload_models_skips_secondary_when_no_ensemble(self):
        """Test that preload_models skips secondary model when use_ensemble=False."""
        from voice_man.services.forensic.ser_service import SERService

        service = SERService(use_ensemble=False)

        with patch.object(service, "_load_primary_model"):
            with patch.object(service, "_load_secondary_model") as mock_secondary:
                await service.preload_models()

        mock_secondary.assert_not_called()

    @pytest.mark.asyncio
    async def test_preload_models_reports_memory_usage(self):
        """Test that preload_models reports memory usage in stats."""
        from voice_man.services.forensic.ser_service import SERService

        service = SERService()

        with patch.object(service, "_load_primary_model"):
            with patch.object(service, "_load_secondary_model"):
                stats = await service.preload_models()

        assert "memory_used_mb" in stats, "Stats should include 'memory_used_mb'"


class TestModelCacheIntegration:
    """Integration tests for model caching behavior."""

    def test_load_primary_model_uses_cache(self):
        """Test that _load_primary_model checks cache first."""
        from voice_man.services.forensic.ser_service import SERService

        # Clear cache
        SERService._model_cache.clear()

        service = SERService()

        # Mock the actual model loading
        mock_model = MagicMock()
        mock_processor = MagicMock()

        with patch("transformers.Wav2Vec2Processor.from_pretrained", return_value=mock_processor):
            with patch("transformers.Wav2Vec2Model.from_pretrained", return_value=mock_model):
                # First load should populate cache
                service._load_primary_model()

                # Check cache was populated
                assert (
                    "primary_model" in SERService._model_cache or service._primary_model is not None
                )

        # Cleanup
        SERService._model_cache.clear()
