"""
WhisperX Configuration Tests

RED-GREEN-REFACTOR TDD cycle for SPEC-WHISPERX-001
Tests for U1, E1, S1, S2, N1, N2 requirements.
"""

import os
import pytest
from unittest.mock import patch, MagicMock


class TestWhisperXConfig:
    """Tests for WhisperX configuration dataclass."""

    def test_default_config_values(self):
        """Test default configuration values are set correctly."""
        from voice_man.config.whisperx_config import WhisperXConfig

        config = WhisperXConfig()

        # Default values from SPEC
        assert config.model_size == "large-v3"
        assert config.device == "cuda"
        assert config.compute_type == "float16"
        assert config.language == "ko"

    def test_config_from_environment_variables(self):
        """Test configuration can be loaded from environment variables."""
        from voice_man.config.whisperx_config import WhisperXConfig

        with patch.dict(
            os.environ,
            {
                "WHISPERX_MODEL_SIZE": "medium",
                "WHISPERX_LANGUAGE": "en",
                "WHISPERX_DEVICE": "cpu",
                "WHISPERX_COMPUTE_TYPE": "int8",
            },
        ):
            config = WhisperXConfig.from_env()

            assert config.model_size == "medium"
            assert config.language == "en"
            assert config.device == "cpu"
            assert config.compute_type == "int8"

    def test_hf_token_not_hardcoded(self):
        """N2: Ensure HF_TOKEN is never hardcoded in config."""
        from voice_man.config.whisperx_config import WhisperXConfig

        config = WhisperXConfig()

        # Token should not be stored in config directly
        assert not hasattr(config, "hf_token")

    def test_get_hf_token_from_environment(self):
        """E1: HF token should be retrieved from environment only."""
        from voice_man.config.whisperx_config import get_hf_token

        with patch.dict(os.environ, {"HF_TOKEN": "test_token_12345"}):
            token = get_hf_token()
            assert token == "test_token_12345"

    def test_get_hf_token_raises_when_missing(self):
        """E1: Should raise error when HF_TOKEN is not set."""
        from voice_man.config.whisperx_config import get_hf_token, HFTokenNotFoundError

        with patch.dict(os.environ, {}, clear=True):
            # Remove HF_TOKEN if it exists
            os.environ.pop("HF_TOKEN", None)

            with pytest.raises(HFTokenNotFoundError):
                get_hf_token()

    def test_korean_alignment_model_config(self):
        """S1: Korean language should use specific alignment model."""
        from voice_man.config.whisperx_config import WhisperXConfig

        config = WhisperXConfig(language="ko")

        assert config.alignment_model == "jonatasgrosman/wav2vec2-large-xlsr-53-korean"

    def test_english_alignment_model_config(self):
        """S1: English language should use appropriate alignment model."""
        from voice_man.config.whisperx_config import WhisperXConfig

        config = WhisperXConfig(language="en")

        assert config.alignment_model == "WAV2VEC2_ASR_BASE_960H"

    def test_gpu_memory_thresholds(self):
        """S2: GPU memory thresholds for sequential loading."""
        from voice_man.config.whisperx_config import WhisperXConfig

        config = WhisperXConfig()

        # S2: 70% threshold for sequential model loading
        assert config.gpu_memory_threshold == 70.0

    def test_chunk_settings_for_long_audio(self):
        """S3: Chunk settings for audio > 30 minutes."""
        from voice_man.config.whisperx_config import WhisperXConfig

        config = WhisperXConfig()

        # S3: 10 minute chunks with 30 second overlap
        assert config.max_audio_duration == 1800  # 30 minutes in seconds
        assert config.chunk_duration == 600  # 10 minutes in seconds
        assert config.chunk_overlap == 30  # 30 seconds overlap

    def test_supported_audio_formats(self):
        """E2: Supported input audio formats."""
        from voice_man.config.whisperx_config import WhisperXConfig

        config = WhisperXConfig()

        supported = config.supported_formats
        assert "m4a" in supported
        assert "mp3" in supported
        assert "wav" in supported
        assert "flac" in supported
        assert "ogg" in supported

    def test_speaker_count_limits(self):
        """E4: Speaker count limits (1-10)."""
        from voice_man.config.whisperx_config import WhisperXConfig

        config = WhisperXConfig()

        assert config.min_speakers == 1
        assert config.max_speakers == 10

    def test_timestamp_accuracy_target(self):
        """U2: Word-level timestamp accuracy target."""
        from voice_man.config.whisperx_config import WhisperXConfig

        config = WhisperXConfig()

        # U2: 100ms accuracy target
        assert config.timestamp_accuracy_ms == 100

    def test_model_memory_requirements(self):
        """N1: Model memory requirements for OOM prevention."""
        from voice_man.config.whisperx_config import WhisperXConfig

        config = WhisperXConfig()

        # Memory requirements from SPEC
        assert config.whisper_memory_gb == 3.0
        assert config.wav2vec_memory_gb == 1.2
        assert config.pyannote_memory_gb == 1.0
        assert config.total_memory_gb == pytest.approx(5.2, rel=0.1)

    def test_progress_stages(self):
        """E3: Pipeline progress stages configuration."""
        from voice_man.config.whisperx_config import WhisperXConfig

        config = WhisperXConfig()

        # E3: Progress percentages per stage
        assert config.progress_stages["transcription"] == (0, 40)
        assert config.progress_stages["alignment"] == (40, 70)
        assert config.progress_stages["diarization"] == (70, 100)

    def test_diarization_model_name(self):
        """F3: Pyannote diarization model name."""
        from voice_man.config.whisperx_config import WhisperXConfig

        config = WhisperXConfig()

        assert config.diarization_model == "pyannote/speaker-diarization-3.1"

    def test_config_to_dict(self):
        """Test config can be serialized to dictionary."""
        from voice_man.config.whisperx_config import WhisperXConfig

        config = WhisperXConfig()
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert "model_size" in config_dict
        assert "device" in config_dict
        assert "language" in config_dict
