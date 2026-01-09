"""
Tests for Speech Emotion Recognition (SER) Service
SPEC-FORENSIC-001 Phase 2-B: SER service tests

TDD RED Phase: These tests define the expected behavior of the SER service.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock


class TestSERServiceInit:
    """Tests for SERService initialization."""

    def test_init_default_parameters(self):
        """Test SERService initializes with default parameters."""
        from voice_man.services.forensic.ser_service import SERService

        service = SERService()

        assert service.device in ["auto", "cpu", "cuda"]
        assert service.use_ensemble is True

    def test_init_custom_device(self):
        """Test SERService initializes with custom device."""
        from voice_man.services.forensic.ser_service import SERService

        service = SERService(device="cpu")
        assert service.device == "cpu"

    def test_init_disable_ensemble(self):
        """Test SERService with ensemble disabled."""
        from voice_man.services.forensic.ser_service import SERService

        service = SERService(use_ensemble=False)
        assert service.use_ensemble is False

    def test_lazy_model_loading(self):
        """Test that models are not loaded until first use."""
        from voice_man.services.forensic.ser_service import SERService

        service = SERService()

        # Models should not be loaded yet
        assert service._primary_model is None
        assert service._secondary_model is None


class TestDeviceDetection:
    """Tests for device auto-detection."""

    def test_device_auto_detection_cpu_fallback(self):
        """Test device auto-detection falls back to CPU."""
        from voice_man.services.forensic.ser_service import SERService

        with patch("torch.cuda.is_available", return_value=False):
            service = SERService(device="auto")
            detected = service._detect_device()
            assert detected == "cpu"

    def test_device_auto_detection_cuda_available(self):
        """Test device auto-detection uses CUDA when available."""
        from voice_man.services.forensic.ser_service import SERService

        with patch("torch.cuda.is_available", return_value=True):
            service = SERService(device="auto")
            detected = service._detect_device()
            assert detected == "cuda"

    def test_device_explicit_cpu(self):
        """Test explicit CPU device selection."""
        from voice_man.services.forensic.ser_service import SERService

        service = SERService(device="cpu")
        detected = service._detect_device()
        assert detected == "cpu"


class TestAnalyzeEmotionDimensions:
    """Tests for analyze_emotion_dimensions() method."""

    def test_analyze_emotion_dimensions_returns_valid_result(self):
        """Test dimension analysis returns valid EmotionDimensions."""
        from voice_man.services.forensic.ser_service import SERService
        from voice_man.models.forensic.emotion_recognition import EmotionDimensions

        service = SERService(device="cpu")

        # Create mock audio
        sr = 16000
        duration = 2.0
        audio = np.random.randn(int(sr * duration)).astype(np.float32) * 0.3

        # Mock the model loading and inference
        with patch.object(service, "_load_primary_model"):
            with patch.object(service, "_run_primary_inference") as mock_inference:
                mock_inference.return_value = {
                    "arousal": 0.7,
                    "dominance": 0.5,
                    "valence": 0.3,
                }

                result = service.analyze_emotion_dimensions(audio, sr)

        assert isinstance(result, EmotionDimensions)
        assert 0.0 <= result.arousal <= 1.0
        assert 0.0 <= result.dominance <= 1.0
        assert 0.0 <= result.valence <= 1.0

    def test_analyze_emotion_dimensions_range_validation(self):
        """Test dimension values are clamped to valid range."""
        from voice_man.services.forensic.ser_service import SERService

        service = SERService(device="cpu")

        sr = 16000
        audio = np.random.randn(sr * 2).astype(np.float32) * 0.3

        with patch.object(service, "_load_primary_model"):
            with patch.object(service, "_run_primary_inference") as mock_inference:
                # Return out-of-range values
                mock_inference.return_value = {
                    "arousal": 1.5,  # Should be clamped to 1.0
                    "dominance": -0.2,  # Should be clamped to 0.0
                    "valence": 0.5,
                }

                result = service.analyze_emotion_dimensions(audio, sr)

        assert result.arousal == 1.0
        assert result.dominance == 0.0
        assert result.valence == 0.5


class TestAnalyzeCategoricalEmotion:
    """Tests for analyze_categorical_emotion() method."""

    def test_analyze_categorical_emotion_returns_valid_result(self):
        """Test categorical analysis returns valid CategoricalEmotion."""
        from voice_man.services.forensic.ser_service import SERService
        from voice_man.models.forensic.emotion_recognition import CategoricalEmotion

        service = SERService(device="cpu")

        sr = 16000
        audio = np.random.randn(sr * 2).astype(np.float32) * 0.3

        with patch.object(service, "_load_secondary_model"):
            with patch.object(service, "_run_secondary_inference") as mock_inference:
                mock_inference.return_value = {
                    "emotion": "angry",
                    "confidence": 0.85,
                    "probabilities": {
                        "angry": 0.85,
                        "happy": 0.05,
                        "sad": 0.05,
                        "neutral": 0.05,
                    },
                }

                result = service.analyze_categorical_emotion(audio, sr)

        assert isinstance(result, CategoricalEmotion)
        assert result.emotion_type in [
            "angry",
            "happy",
            "sad",
            "neutral",
            "fear",
            "disgust",
            "surprise",
        ]
        assert 0.0 <= result.confidence <= 1.0

    def test_analyze_categorical_emotion_all_types(self):
        """Test categorical analysis for all emotion types."""
        from voice_man.services.forensic.ser_service import SERService

        service = SERService(device="cpu")

        sr = 16000
        audio = np.random.randn(sr * 2).astype(np.float32) * 0.3

        emotion_types = ["angry", "happy", "sad", "neutral"]

        for emotion_type in emotion_types:
            with patch.object(service, "_load_secondary_model"):
                with patch.object(service, "_run_secondary_inference") as mock_inference:
                    mock_inference.return_value = {
                        "emotion": emotion_type,
                        "confidence": 0.8,
                        "probabilities": {},
                    }

                    result = service.analyze_categorical_emotion(audio, sr)

            assert result.emotion_type == emotion_type


class TestAnalyzeEnsemble:
    """Tests for analyze_ensemble() method."""

    def test_analyze_ensemble_returns_multi_model_result(self):
        """Test ensemble analysis returns MultiModelEmotionResult."""
        from voice_man.services.forensic.ser_service import SERService
        from voice_man.models.forensic.emotion_recognition import MultiModelEmotionResult

        service = SERService(device="cpu", use_ensemble=True)

        sr = 16000
        audio = np.random.randn(sr * 2).astype(np.float32) * 0.3

        with patch.object(service, "_load_primary_model"):
            with patch.object(service, "_load_secondary_model"):
                with patch.object(service, "_run_primary_inference") as mock_primary:
                    with patch.object(service, "_run_secondary_inference") as mock_secondary:
                        mock_primary.return_value = {
                            "arousal": 0.7,
                            "dominance": 0.5,
                            "valence": 0.3,
                        }
                        mock_secondary.return_value = {
                            "emotion": "angry",
                            "confidence": 0.8,
                            "probabilities": {},
                        }

                        result = service.analyze_ensemble(audio, sr)

        assert isinstance(result, MultiModelEmotionResult)
        assert result.primary_result is not None
        assert result.secondary_result is not None
        assert 0.0 <= result.ensemble_confidence <= 1.0

    def test_analyze_ensemble_single_model_fallback(self):
        """Test ensemble falls back to single model if one fails."""
        from voice_man.services.forensic.ser_service import SERService

        service = SERService(device="cpu", use_ensemble=True)

        sr = 16000
        audio = np.random.randn(sr * 2).astype(np.float32) * 0.3

        with patch.object(service, "_load_primary_model"):
            with patch.object(service, "_load_secondary_model"):
                with patch.object(service, "_run_primary_inference") as mock_primary:
                    with patch.object(service, "_run_secondary_inference") as mock_secondary:
                        mock_primary.return_value = {
                            "arousal": 0.6,
                            "dominance": 0.5,
                            "valence": 0.4,
                        }
                        # Secondary model fails
                        mock_secondary.side_effect = Exception("Model failed")

                        result = service.analyze_ensemble(audio, sr)

        assert result.primary_result is not None
        assert result.secondary_result is None

    def test_analyze_ensemble_confidence_weighting(self):
        """Test ensemble uses confidence weighting."""
        from voice_man.services.forensic.ser_service import SERService

        service = SERService(device="cpu", use_ensemble=True)

        sr = 16000
        audio = np.random.randn(sr * 2).astype(np.float32) * 0.3

        with patch.object(service, "_load_primary_model"):
            with patch.object(service, "_load_secondary_model"):
                with patch.object(service, "_run_primary_inference") as mock_primary:
                    with patch.object(service, "_run_secondary_inference") as mock_secondary:
                        mock_primary.return_value = {
                            "arousal": 0.8,
                            "dominance": 0.6,
                            "valence": 0.2,
                        }
                        mock_secondary.return_value = {
                            "emotion": "angry",
                            "confidence": 0.9,
                            "probabilities": {},
                        }

                        result = service.analyze_ensemble(audio, sr)

        assert result.confidence_weighted is True


class TestGetForensicEmotionIndicators:
    """Tests for get_forensic_emotion_indicators() method."""

    def test_forensic_indicators_high_arousal_low_valence(self):
        """Test detection of high arousal + low valence pattern."""
        from voice_man.services.forensic.ser_service import SERService
        from voice_man.models.forensic.emotion_recognition import (
            MultiModelEmotionResult,
            EmotionAnalysisResult,
            EmotionDimensions,
            CategoricalEmotion,
        )

        service = SERService(device="cpu")

        # Create result with high arousal, low valence
        result = MultiModelEmotionResult(
            primary_result=EmotionAnalysisResult(
                dimensions=EmotionDimensions(arousal=0.85, dominance=0.6, valence=0.15),
                model_used="primary-model",
                processing_time_ms=100.0,
            ),
            secondary_result=EmotionAnalysisResult(
                categorical=CategoricalEmotion(emotion_type="angry", confidence=0.8),
                model_used="secondary-model",
                processing_time_ms=80.0,
            ),
            ensemble_confidence=0.82,
            audio_duration_seconds=3.0,
        )

        indicators = service.get_forensic_emotion_indicators(result)

        assert indicators.high_arousal_low_valence is True
        assert indicators.stress_indicator is True

    def test_forensic_indicators_emotion_inconsistency(self):
        """Test detection of emotion inconsistency (deception indicator)."""
        from voice_man.services.forensic.ser_service import SERService
        from voice_man.models.forensic.emotion_recognition import (
            MultiModelEmotionResult,
            EmotionAnalysisResult,
            EmotionDimensions,
            CategoricalEmotion,
        )

        service = SERService(device="cpu")

        # Create result with inconsistent emotions
        # High valence (positive) but "angry" categorical emotion
        result = MultiModelEmotionResult(
            primary_result=EmotionAnalysisResult(
                dimensions=EmotionDimensions(arousal=0.3, dominance=0.5, valence=0.9),
                model_used="primary-model",
                processing_time_ms=100.0,
            ),
            secondary_result=EmotionAnalysisResult(
                categorical=CategoricalEmotion(emotion_type="angry", confidence=0.85),
                model_used="secondary-model",
                processing_time_ms=80.0,
            ),
            ensemble_confidence=0.6,
            audio_duration_seconds=3.0,
        )

        indicators = service.get_forensic_emotion_indicators(result)

        # High inconsistency between positive valence and angry emotion
        assert indicators.emotion_inconsistency_score > 0.5
        assert indicators.deception_indicator is True

    def test_forensic_indicators_arousal_level_classification(self):
        """Test arousal level classification."""
        from voice_man.services.forensic.ser_service import SERService
        from voice_man.models.forensic.emotion_recognition import (
            MultiModelEmotionResult,
            EmotionAnalysisResult,
            EmotionDimensions,
        )

        service = SERService(device="cpu")

        # Test low arousal
        result_low = MultiModelEmotionResult(
            primary_result=EmotionAnalysisResult(
                dimensions=EmotionDimensions(arousal=0.2, dominance=0.5, valence=0.5),
                model_used="model",
                processing_time_ms=100.0,
            ),
            ensemble_confidence=0.8,
            audio_duration_seconds=3.0,
        )
        indicators_low = service.get_forensic_emotion_indicators(result_low)
        assert indicators_low.arousal_level == "low"

        # Test high arousal
        result_high = MultiModelEmotionResult(
            primary_result=EmotionAnalysisResult(
                dimensions=EmotionDimensions(arousal=0.85, dominance=0.5, valence=0.5),
                model_used="model",
                processing_time_ms=100.0,
            ),
            ensemble_confidence=0.8,
            audio_duration_seconds=3.0,
        )
        indicators_high = service.get_forensic_emotion_indicators(result_high)
        assert indicators_high.arousal_level == "high"

    def test_forensic_indicators_dominant_emotion(self):
        """Test dominant emotion extraction."""
        from voice_man.services.forensic.ser_service import SERService
        from voice_man.models.forensic.emotion_recognition import (
            MultiModelEmotionResult,
            EmotionAnalysisResult,
            CategoricalEmotion,
        )

        service = SERService(device="cpu")

        result = MultiModelEmotionResult(
            secondary_result=EmotionAnalysisResult(
                categorical=CategoricalEmotion(emotion_type="sad", confidence=0.9),
                model_used="model",
                processing_time_ms=100.0,
            ),
            ensemble_confidence=0.9,
            audio_duration_seconds=3.0,
        )

        indicators = service.get_forensic_emotion_indicators(result)
        assert indicators.dominant_emotion == "sad"


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_audio_handling(self):
        """Test handling of empty audio array."""
        from voice_man.services.forensic.ser_service import SERService

        service = SERService(device="cpu")

        sr = 16000
        audio = np.array([], dtype=np.float32)

        with pytest.raises(ValueError, match="Audio array is empty"):
            service.analyze_ensemble(audio, sr)

    def test_short_audio_handling(self):
        """Test handling of very short audio (< 0.5s)."""
        from voice_man.services.forensic.ser_service import SERService

        service = SERService(device="cpu")

        sr = 16000
        # 0.1 second audio
        audio = np.random.randn(int(sr * 0.1)).astype(np.float32) * 0.3

        with pytest.raises(ValueError, match="Audio too short"):
            service.analyze_ensemble(audio, sr)

    def test_invalid_sample_rate(self):
        """Test handling of invalid sample rate."""
        from voice_man.services.forensic.ser_service import SERService

        service = SERService(device="cpu")

        sr = 0  # Invalid
        audio = np.random.randn(16000).astype(np.float32) * 0.3

        with pytest.raises(ValueError, match="Invalid sample rate"):
            service.analyze_ensemble(audio, sr)

    def test_audio_path_loading(self):
        """Test loading audio from file path."""
        from voice_man.services.forensic.ser_service import SERService

        service = SERService(device="cpu")

        # Test with non-existent file
        with pytest.raises(FileNotFoundError):
            service.analyze_ensemble_from_file("/nonexistent/path/audio.wav")

    def test_graceful_degradation_no_transformers(self):
        """Test graceful degradation when transformers not installed."""
        from voice_man.services.forensic.ser_service import SERService

        with patch.dict("sys.modules", {"transformers": None}):
            service = SERService(device="cpu")
            assert service._transformers_available is False


class TestModelLoading:
    """Tests for model loading behavior."""

    def test_primary_model_name(self):
        """Test primary model name is correct."""
        from voice_man.services.forensic.ser_service import SERService

        service = SERService(device="cpu")
        assert service.PRIMARY_MODEL_NAME == "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"

    def test_secondary_model_name(self):
        """Test secondary model name is correct."""
        from voice_man.services.forensic.ser_service import SERService

        service = SERService(device="cpu")
        assert service.SECONDARY_MODEL_NAME == "speechbrain/emotion-recognition-wav2vec2-IEMOCAP"

    def test_model_loading_caches_models(self):
        """Test that models are cached after first load."""
        from voice_man.services.forensic.ser_service import SERService

        service = SERService(device="cpu")

        # Mock the actual model loading
        mock_model = MagicMock()
        with patch.object(service, "_load_model_from_hub", return_value=mock_model):
            service._load_primary_model()
            first_model = service._primary_model

            service._load_primary_model()
            second_model = service._primary_model

        # Should be the same cached model
        assert first_model is second_model


class TestAudioPreprocessing:
    """Tests for audio preprocessing."""

    def test_resample_to_16khz(self):
        """Test audio is resampled to 16kHz if needed."""
        from voice_man.services.forensic.ser_service import SERService

        service = SERService(device="cpu")

        # 44.1kHz audio
        sr_original = 44100
        duration = 2.0
        audio = np.random.randn(int(sr_original * duration)).astype(np.float32) * 0.3

        resampled, sr_new = service._preprocess_audio(audio, sr_original)

        assert sr_new == 16000
        # Duration should be approximately the same
        expected_samples = int(16000 * duration)
        assert abs(len(resampled) - expected_samples) < 100

    def test_normalize_audio(self):
        """Test audio is normalized."""
        from voice_man.services.forensic.ser_service import SERService

        service = SERService(device="cpu")

        sr = 16000
        # Audio with large amplitude
        audio = np.random.randn(sr * 2).astype(np.float32) * 5.0

        normalized, _ = service._preprocess_audio(audio, sr)

        # Should be normalized to [-1, 1] range
        assert np.max(np.abs(normalized)) <= 1.0

    def test_mono_conversion(self):
        """Test stereo audio is converted to mono."""
        from voice_man.services.forensic.ser_service import SERService

        service = SERService(device="cpu")

        sr = 16000
        # Stereo audio (2 channels)
        audio_stereo = np.random.randn(2, sr * 2).astype(np.float32) * 0.3

        mono, _ = service._preprocess_audio(audio_stereo, sr)

        assert mono.ndim == 1


class TestIntegration:
    """Integration tests with mocked models."""

    def test_full_analysis_pipeline(self):
        """Test full analysis pipeline end-to-end."""
        from voice_man.services.forensic.ser_service import SERService
        from voice_man.models.forensic.emotion_recognition import (
            MultiModelEmotionResult,
            ForensicEmotionIndicators,
        )

        service = SERService(device="cpu", use_ensemble=True)

        sr = 16000
        duration = 3.0
        audio = np.random.randn(int(sr * duration)).astype(np.float32) * 0.3

        with patch.object(service, "_load_primary_model"):
            with patch.object(service, "_load_secondary_model"):
                with patch.object(service, "_run_primary_inference") as mock_primary:
                    with patch.object(service, "_run_secondary_inference") as mock_secondary:
                        mock_primary.return_value = {
                            "arousal": 0.75,
                            "dominance": 0.55,
                            "valence": 0.25,
                        }
                        mock_secondary.return_value = {
                            "emotion": "angry",
                            "confidence": 0.82,
                            "probabilities": {
                                "angry": 0.82,
                                "happy": 0.05,
                                "sad": 0.08,
                                "neutral": 0.05,
                            },
                        }

                        result = service.analyze_ensemble(audio, sr)
                        indicators = service.get_forensic_emotion_indicators(result)

        assert isinstance(result, MultiModelEmotionResult)
        assert isinstance(indicators, ForensicEmotionIndicators)
        assert result.audio_duration_seconds == pytest.approx(duration, rel=0.1)
        assert indicators.high_arousal_low_valence is True
        assert indicators.dominant_emotion == "angry"
