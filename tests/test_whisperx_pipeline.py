"""
WhisperX Pipeline Tests

RED-GREEN-REFACTOR TDD cycle for SPEC-WHISPERX-001
Tests for F1, F2, F3, F4, U1, U2, U3 requirements.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from pathlib import Path
import tempfile
import os


class TestWhisperXPipeline:
    """Tests for WhisperXPipeline class (F1)."""

    @pytest.fixture
    def mock_audio_file(self, tmp_path):
        """Create a mock audio file for testing."""
        audio_file = tmp_path / "test_audio.wav"
        # Create a minimal file with some content
        audio_file.write_bytes(b"RIFF" + b"\x00" * 1000)
        return str(audio_file)

    def test_pipeline_initialization(self):
        """F1: Test pipeline initialization with default parameters."""
        from voice_man.models.whisperx_pipeline import WhisperXPipeline

        with patch.dict(os.environ, {"HF_TOKEN": "test_token"}):
            with patch("voice_man.models.whisperx_pipeline.whisperx") as mock_wx:
                mock_wx.load_model.return_value = MagicMock()

                pipeline = WhisperXPipeline(
                    model_size="large-v3",
                    device="cuda",
                    language="ko",
                )

                assert pipeline.model_size == "large-v3"
                assert pipeline.device == "cuda"
                assert pipeline.language == "ko"

    def test_pipeline_initialization_with_cpu_fallback(self):
        """E1: Test pipeline falls back to CPU when GPU unavailable."""
        from voice_man.models.whisperx_pipeline import WhisperXPipeline

        with patch.dict(os.environ, {"HF_TOKEN": "test_token"}):
            with patch("voice_man.models.whisperx_pipeline.whisperx") as mock_wx:
                with patch("voice_man.models.whisperx_pipeline._import_torch") as mock_torch:
                    mock_torch_module = MagicMock()
                    mock_torch_module.cuda.is_available.return_value = False
                    mock_torch.return_value = mock_torch_module
                    mock_wx.load_model.return_value = MagicMock()

                    pipeline = WhisperXPipeline(
                        model_size="large-v3",
                        device="auto",
                        language="ko",
                    )

                    assert pipeline.device == "cpu"

    def test_pipeline_validates_hf_token_on_init(self):
        """E1: Pipeline should validate HF token at initialization."""
        from voice_man.models.whisperx_pipeline import WhisperXPipeline
        from voice_man.config.whisperx_config import HFTokenNotFoundError

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("HF_TOKEN", None)

            with pytest.raises(HFTokenNotFoundError):
                WhisperXPipeline(
                    model_size="large-v3",
                    device="cuda",
                    language="ko",
                )

    @pytest.mark.asyncio
    async def test_process_returns_complete_result(self, mock_audio_file):
        """F1: Test process() returns complete pipeline result."""
        from voice_man.models.whisperx_pipeline import WhisperXPipeline, PipelineResult

        with patch.dict(os.environ, {"HF_TOKEN": "test_token"}):
            with patch("voice_man.models.whisperx_pipeline.whisperx") as mock_wx:
                # Setup mocks
                mock_wx.load_model.return_value = MagicMock()
                mock_wx.load_audio.return_value = MagicMock()
                mock_wx.load_align_model.return_value = (MagicMock(), MagicMock())

                pipeline = WhisperXPipeline(
                    model_size="large-v3",
                    device="cuda",
                    language="ko",
                )

                # Mock internal methods
                pipeline._transcribe = AsyncMock(return_value={"segments": []})
                pipeline._align = AsyncMock(return_value={"segments": []})
                pipeline._diarize = AsyncMock(return_value={"segments": []})

                result = await pipeline.process(mock_audio_file, num_speakers=2)

                assert isinstance(result, PipelineResult)
                assert hasattr(result, "text")
                assert hasattr(result, "segments")
                assert hasattr(result, "speakers")
                assert hasattr(result, "speaker_stats")

    @pytest.mark.asyncio
    async def test_transcribe_returns_segments(self, mock_audio_file):
        """F1: Test transcribe() returns segments with timing."""
        from voice_man.models.whisperx_pipeline import WhisperXPipeline

        with patch.dict(os.environ, {"HF_TOKEN": "test_token"}):
            with patch("voice_man.models.whisperx_pipeline.whisperx") as mock_wx:
                mock_model = MagicMock()
                mock_model.transcribe.return_value = {
                    "segments": [
                        {"start": 0.0, "end": 2.0, "text": "Hello"},
                        {"start": 2.0, "end": 4.0, "text": "World"},
                    ],
                    "language": "ko",
                }
                mock_wx.load_model.return_value = mock_model
                mock_wx.load_audio.return_value = MagicMock()

                pipeline = WhisperXPipeline(
                    model_size="large-v3",
                    device="cuda",
                    language="ko",
                )

                result = await pipeline.transcribe(mock_audio_file)

                assert "segments" in result
                assert len(result["segments"]) == 2

    @pytest.mark.asyncio
    async def test_align_returns_word_timestamps(self, mock_audio_file):
        """F2: Test align() returns word-level timestamps."""
        from voice_man.models.whisperx_pipeline import WhisperXPipeline

        with patch.dict(os.environ, {"HF_TOKEN": "test_token"}):
            with patch("voice_man.models.whisperx_pipeline.whisperx") as mock_wx:
                mock_wx.load_model.return_value = MagicMock()
                mock_wx.load_audio.return_value = MagicMock()
                mock_wx.load_align_model.return_value = (MagicMock(), MagicMock())
                mock_wx.align.return_value = {
                    "segments": [
                        {
                            "start": 0.0,
                            "end": 2.0,
                            "text": "Hello World",
                            "words": [
                                {"word": "Hello", "start": 0.0, "end": 0.8, "score": 0.95},
                                {"word": "World", "start": 1.0, "end": 1.8, "score": 0.92},
                            ],
                        }
                    ]
                }

                pipeline = WhisperXPipeline(
                    model_size="large-v3",
                    device="cuda",
                    language="ko",
                )

                segments = {"segments": [{"start": 0.0, "end": 2.0, "text": "Hello World"}]}
                audio = MagicMock()

                result = await pipeline.align(segments, audio)

                assert "segments" in result
                assert "words" in result["segments"][0]
                # U2: Check word-level data
                words = result["segments"][0]["words"]
                assert len(words) == 2
                assert "start" in words[0]
                assert "end" in words[0]
                assert "score" in words[0]

    @pytest.mark.asyncio
    async def test_diarize_returns_speaker_segments(self, mock_audio_file):
        """F3: Test diarize() returns speaker segments."""
        from voice_man.models.whisperx_pipeline import WhisperXPipeline

        with patch.dict(os.environ, {"HF_TOKEN": "test_token"}):
            with patch("voice_man.models.whisperx_pipeline.whisperx") as mock_wx:
                mock_wx.load_model.return_value = MagicMock()
                mock_wx.load_audio.return_value = MagicMock()

                # Mock diarization result
                mock_wx.DiarizationPipeline.return_value = MagicMock()
                mock_wx.assign_word_speakers.return_value = {
                    "segments": [
                        {"speaker": "SPEAKER_00", "start": 0.0, "end": 2.5},
                        {"speaker": "SPEAKER_01", "start": 2.5, "end": 5.0},
                    ]
                }

                pipeline = WhisperXPipeline(
                    model_size="large-v3",
                    device="cuda",
                    language="ko",
                )

                audio = MagicMock()
                segments = {"segments": []}

                result = await pipeline.diarize(audio, segments, num_speakers=2)

                assert "segments" in result
                # U3: Check speaker ID assignment
                for seg in result["segments"]:
                    assert "speaker" in seg
                    assert seg["speaker"].startswith("SPEAKER_")

    def test_gpu_context_consistency(self):
        """U1: All models should run on same GPU context."""
        from voice_man.models.whisperx_pipeline import WhisperXPipeline

        with patch.dict(os.environ, {"HF_TOKEN": "test_token"}):
            with patch("voice_man.models.whisperx_pipeline.whisperx") as mock_wx:
                mock_wx.load_model.return_value = MagicMock()

                pipeline = WhisperXPipeline(
                    model_size="large-v3",
                    device="cuda",
                    language="ko",
                )

                # All models should use same device
                assert pipeline._whisper_device == pipeline._align_device
                assert pipeline._align_device == pipeline._diarize_device

    @pytest.mark.asyncio
    async def test_speaker_stats_generation(self, mock_audio_file):
        """F4: Test speaker statistics generation."""
        from voice_man.models.whisperx_pipeline import WhisperXPipeline

        with patch.dict(os.environ, {"HF_TOKEN": "test_token"}):
            with patch("voice_man.models.whisperx_pipeline.whisperx") as mock_wx:
                mock_wx.load_model.return_value = MagicMock()

                pipeline = WhisperXPipeline(
                    model_size="large-v3",
                    device="cuda",
                    language="ko",
                )

                # Mock segments with speaker info
                segments = [
                    {"speaker": "SPEAKER_00", "start": 0.0, "end": 10.0, "text": "Hello"},
                    {"speaker": "SPEAKER_01", "start": 10.0, "end": 20.0, "text": "Hi"},
                    {"speaker": "SPEAKER_00", "start": 20.0, "end": 30.0, "text": "Bye"},
                ]

                stats = pipeline.generate_speaker_stats(segments)

                # F4: Verify statistics
                assert "SPEAKER_00" in stats
                assert "SPEAKER_01" in stats
                assert stats["SPEAKER_00"]["total_duration"] == pytest.approx(20.0)
                assert stats["SPEAKER_01"]["total_duration"] == pytest.approx(10.0)
                assert stats["SPEAKER_00"]["turn_count"] == 2
                assert stats["SPEAKER_01"]["turn_count"] == 1
                assert "speech_ratio" in stats["SPEAKER_00"]

    @pytest.mark.asyncio
    async def test_progress_callback_called(self, mock_audio_file):
        """E3: Test progress callback is called for each stage."""
        from voice_man.models.whisperx_pipeline import WhisperXPipeline

        progress_updates = []

        def progress_callback(stage: str, progress: float, message: str):
            progress_updates.append((stage, progress, message))

        with patch.dict(os.environ, {"HF_TOKEN": "test_token"}):
            with patch("voice_man.models.whisperx_pipeline.whisperx") as mock_wx:
                mock_wx.load_model.return_value = MagicMock()
                mock_wx.load_audio.return_value = MagicMock()

                pipeline = WhisperXPipeline(
                    model_size="large-v3",
                    device="cuda",
                    language="ko",
                )

                # Mock internal methods
                pipeline._transcribe = AsyncMock(return_value={"segments": []})
                pipeline._align = AsyncMock(return_value={"segments": []})
                pipeline._diarize = AsyncMock(return_value={"segments": []})

                await pipeline.process(
                    mock_audio_file,
                    num_speakers=2,
                    progress_callback=progress_callback,
                )

                # E3: Verify progress updates for each stage
                stages = [p[0] for p in progress_updates]
                assert "transcription" in stages
                assert "alignment" in stages
                assert "diarization" in stages

    @pytest.mark.asyncio
    async def test_sequential_model_loading_on_high_memory(self, mock_audio_file):
        """S2: Sequential model loading when GPU memory > 70%."""
        from voice_man.models.whisperx_pipeline import WhisperXPipeline

        with patch.dict(os.environ, {"HF_TOKEN": "test_token"}):
            with patch("voice_man.models.whisperx_pipeline.whisperx") as mock_wx:
                with patch("voice_man.services.gpu_monitor_service.GPUMonitorService") as mock_gpu:
                    # Simulate high GPU memory usage
                    mock_gpu_instance = MagicMock()
                    mock_gpu_instance.get_gpu_memory_stats.return_value = {
                        "usage_percentage": 75.0,
                        "available": True,
                    }
                    mock_gpu.return_value = mock_gpu_instance

                    mock_wx.load_model.return_value = MagicMock()

                    pipeline = WhisperXPipeline(
                        model_size="large-v3",
                        device="cuda",
                        language="ko",
                    )

                    # Should use sequential loading
                    assert pipeline._sequential_loading is True

    def test_model_unload_clears_gpu_cache(self):
        """N1: Model unload should clear GPU cache."""
        from voice_man.models.whisperx_pipeline import WhisperXPipeline

        with patch.dict(os.environ, {"HF_TOKEN": "test_token"}):
            with patch("voice_man.models.whisperx_pipeline.whisperx") as mock_wx:
                with patch("torch.cuda.empty_cache") as mock_cache_clear:
                    mock_wx.load_model.return_value = MagicMock()

                    pipeline = WhisperXPipeline(
                        model_size="large-v3",
                        device="cuda",
                        language="ko",
                    )

                    pipeline.unload()

                    mock_cache_clear.assert_called()


class TestPipelineResult:
    """Tests for PipelineResult dataclass."""

    def test_pipeline_result_fields(self):
        """Test PipelineResult has all required fields."""
        from voice_man.models.whisperx_pipeline import PipelineResult

        result = PipelineResult(
            text="Hello World",
            segments=[{"start": 0.0, "end": 2.0, "text": "Hello World"}],
            speakers=["SPEAKER_00", "SPEAKER_01"],
            speaker_stats={"SPEAKER_00": {"total_duration": 10.0}},
            language="ko",
            word_segments=[{"word": "Hello", "start": 0.0, "end": 0.5}],
        )

        assert result.text == "Hello World"
        assert len(result.segments) == 1
        assert len(result.speakers) == 2
        assert "SPEAKER_00" in result.speaker_stats

    def test_pipeline_result_to_dict(self):
        """Test PipelineResult can be converted to dict."""
        from voice_man.models.whisperx_pipeline import PipelineResult

        result = PipelineResult(
            text="Test",
            segments=[],
            speakers=[],
            speaker_stats={},
            language="ko",
            word_segments=[],
        )

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert "text" in result_dict
        assert "segments" in result_dict
        assert "speakers" in result_dict


class TestChunkProcessing:
    """Tests for long audio chunk processing (S3)."""

    @pytest.fixture
    def mock_long_audio(self, tmp_path):
        """Create a mock long audio file."""
        audio_file = tmp_path / "long_audio.wav"
        audio_file.write_bytes(b"RIFF" + b"\x00" * 10000)
        return str(audio_file)

    @pytest.mark.asyncio
    async def test_long_audio_is_chunked(self, mock_long_audio):
        """S3: Audio > 30 min should be split into 10 min chunks."""
        from voice_man.models.whisperx_pipeline import WhisperXPipeline

        with patch.dict(os.environ, {"HF_TOKEN": "test_token"}):
            with patch("voice_man.models.whisperx_pipeline.whisperx") as mock_wx:
                mock_wx.load_model.return_value = MagicMock()
                mock_wx.load_audio.return_value = MagicMock()

                # Mock audio duration > 30 minutes
                with patch.object(
                    WhisperXPipeline, "_get_audio_duration", return_value=2400.0
                ):  # 40 minutes
                    pipeline = WhisperXPipeline(
                        model_size="large-v3",
                        device="cuda",
                        language="ko",
                    )

                    chunks = pipeline._split_audio_to_chunks(mock_long_audio)

                    # S3: Should create chunks
                    assert len(chunks) >= 4  # 40 min / 10 min = 4 chunks

    @pytest.mark.asyncio
    async def test_chunk_overlap(self, mock_long_audio):
        """S3: Chunks should have 30 second overlap."""
        from voice_man.models.whisperx_pipeline import WhisperXPipeline

        with patch.dict(os.environ, {"HF_TOKEN": "test_token"}):
            with patch("voice_man.models.whisperx_pipeline.whisperx") as mock_wx:
                mock_wx.load_model.return_value = MagicMock()

                pipeline = WhisperXPipeline(
                    model_size="large-v3",
                    device="cuda",
                    language="ko",
                )

                # Verify overlap configuration
                assert pipeline.config.chunk_overlap == 30  # 30 seconds
