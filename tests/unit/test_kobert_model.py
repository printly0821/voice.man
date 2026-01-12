"""
KoBERT Model Tests - TAG-001
RED Phase: Failing tests for KoBERT model wrapper with GPU/CPU auto-detection

Tests fail because implementation doesn't exist yet
"""

import pytest
from typing import Optional
from unittest.mock import Mock, patch

# These imports will fail initially - this is the RED phase
from voice_man.services.nlp.kobert_model import KoBERTModel, DeviceType


class TestKoBERTModelInitialization:
    """Test KoBERT model initialization and device detection"""

    def test_model_initialization_with_auto_device(self):
        """Test automatic device detection (GPU/CPU)"""
        model = KoBERTModel(device="auto")

        assert model is not None
        assert model.device in [DeviceType.CUDA, DeviceType.CPU]
        assert model.model_name == "skt/kobert-base-v1"

    def test_model_initialization_with_cpu_device(self):
        """Test forced CPU device"""
        model = KoBERTModel(device="cpu")

        assert model.device == DeviceType.CPU
        assert model.is_loaded()

    @pytest.mark.skipif(not pytest.importorskip("torch"), reason="PyTorch not installed")
    def test_model_initialization_with_cuda_device(self):
        """Test forced CUDA device when available"""
        try:
            import torch

            if torch.cuda.is_available():
                model = KoBERTModel(device="cuda")
                assert model.device == DeviceType.CUDA
                assert model.is_loaded()
            else:
                pytest.skip("CUDA not available")
        except ImportError:
            pytest.skip("PyTorch not installed")

    def test_model_singleton_pattern(self):
        """Test that model follows singleton pattern"""
        model1 = KoBERTModel(device="cpu")
        model2 = KoBERTModel(device="cpu")

        # Same instance should be returned
        assert model1 is model2

    def test_model_loading_status(self):
        """Test model loading status tracking"""
        model = KoBERTModel(device="cpu")

        # Initially should be loaded
        assert model.is_loaded()
        assert not model.is_loading()

    def test_gpu_memory_check(self):
        """Test GPU memory availability check"""
        model = KoBERTModel(device="cpu")

        if model.device == DeviceType.CUDA:
            memory_info = model.get_gpu_memory_info()
            assert memory_info is not None
            assert "total" in memory_info
            assert "free" in memory_info
        else:
            # Should return None for CPU
            assert model.get_gpu_memory_info() is None


class TestKoBERTModelInference:
    """Test KoBERT model inference functionality"""

    @pytest.fixture
    def sample_text(self) -> str:
        """Sample Korean text for testing"""
        return "안녕하세요, 오늘 날씨가 좋네요."

    def test_encode_single_text(self, sample_text: str):
        """Test encoding single text"""
        model = KoBERTModel(device="cpu")

        result = model.encode(sample_text)

        assert result is not None
        assert hasattr(result, "last_hidden_state") or hasattr(result, "pooler_output")
        assert result.last_hidden_state.shape[0] == 1  # Batch size 1

    def test_encode_batch_texts(self, sample_text: str):
        """Test encoding multiple texts"""
        model = KoBERTModel(device="cpu")

        texts = [sample_text, "감사합니다.", "잘 지내나요?"]
        results = model.encode_batch(texts)

        assert results is not None
        assert len(results) == 3

    def test_get_embeddings(self, sample_text: str):
        """Test getting text embeddings"""
        model = KoBERTModel(device="cpu")

        embeddings = model.get_embeddings(sample_text)

        assert embeddings is not None
        assert embeddings.shape[0] == 1  # Batch size
        assert embeddings.shape[-1] == 768  # BERT hidden size

    def test_tokenization(self, sample_text: str):
        """Test text tokenization"""
        model = KoBERTModel(device="cpu")

        tokens = model.tokenize(sample_text)

        assert tokens is not None
        assert "input_ids" in tokens
        assert "attention_mask" in tokens


class TestKoBERTModelErrorHandling:
    """Test KoBERT model error handling"""

    def test_invalid_device_fallback(self):
        """Test fallback to CPU when invalid device specified"""
        with pytest.warns(UserWarning, match="Invalid device.*fallback to CPU"):
            model = KoBERTModel(device="invalid_device")
            assert model.device == DeviceType.CPU

    def test_empty_text_handling(self):
        """Test handling of empty text"""
        model = KoBERTModel(device="cpu")

        with pytest.raises(ValueError, match="Text cannot be empty"):
            model.encode("")

    def test_model_loading_failure(self):
        """Test handling of model loading failure"""
        with patch("voice_man.services.nlp.kobert_model.AutoModel.from_pretrained") as mock_load:
            mock_load.side_effect = Exception("Model loading failed")

            with pytest.raises(RuntimeError, match="Failed to load KoBERT model"):
                KoBERTModel(device="cpu")

    def test_cuda_unavailable_fallback(self):
        """Test fallback to CPU when CUDA is not available"""
        with patch("torch.cuda.is_available", return_value=False):
            with pytest.warns(UserWarning, match="CUDA not available.*using CPU"):
                model = KoBERTModel(device="auto")
                assert model.device == DeviceType.CPU


class TestKoBERTModelPerformance:
    """Test KoBERT model performance requirements"""

    def test_single_inference_performance(self):
        """Test single inference completes within 100ms"""
        import time

        model = KoBERTModel(device="cpu")
        text = "안녕하세요, 오늘 날씨가 좋네요."

        start_time = time.time()
        model.encode(text)
        inference_time = (time.time() - start_time) * 1000  # Convert to ms

        # Should complete within 100ms on CPU (allow some margin)
        assert inference_time < 500, f"Single inference took {inference_time:.2f}ms"

    def test_batch_inference_performance(self):
        """Test batch inference completes within 500ms"""
        import time

        model = KoBERTModel(device="cpu")
        texts = ["안녕하세요", "감사합니다", "잘 지내나요", "만나서 반가워", "좋은 하루"]

        start_time = time.time()
        model.encode_batch(texts)
        inference_time = (time.time() - start_time) * 1000  # Convert to ms

        # Should complete within 500ms for batch of 5
        assert inference_time < 1000, f"Batch inference took {inference_time:.2f}ms"


class TestKoBERTModelConfiguration:
    """Test KoBERT model configuration"""

    def test_custom_model_name(self):
        """Test loading custom fine-tuned model"""
        model = KoBERTModel(device="cpu", model_name="custom/kobert-finetuned")

        assert model.model_name == "custom/kobert-finetuned"

    def test_max_length_configuration(self):
        """Test custom max_length configuration"""
        model = KoBERTModel(device="cpu", max_length=256)

        assert model.max_length == 256

        # Test that tokenization respects max_length
        long_text = "안녕하세요 " * 100
        tokens = model.tokenize(long_text)

        # Check that sequence is truncated/padded to max_length
        assert tokens["input_ids"].shape[-1] <= 256


class TestKoBERTModelUtilities:
    """Test KoBERT model utility functions"""

    def test_device_info_string(self):
        """Test device info string representation"""
        model = KoBERTModel(device="cpu")

        info = model.get_device_info()
        assert isinstance(info, str)
        assert "CPU" in info or "CUDA" in info

    def test_model_info(self):
        """Test model information"""
        model = KoBERTModel(device="cpu")

        info = model.get_model_info()
        assert isinstance(info, dict)
        assert "model_name" in info
        assert "device" in info
        assert "hidden_size" in info
        assert "max_length" in info

    def test_clear_cache(self):
        """Test clearing model cache"""
        model = KoBERTModel(device="cpu")

        # Should not raise any errors
        model.clear_cache()

    def test_warmup(self):
        """Test model warmup for initial inference"""
        model = KoBERTModel(device="cpu")

        # Should not raise any errors
        model.warmup()

        # After warmup, model should still be loaded
        assert model.is_loaded()
