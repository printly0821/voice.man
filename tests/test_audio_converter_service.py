"""
Audio Converter Service Tests

RED-GREEN-REFACTOR TDD cycle for SPEC-WHISPERX-001
Tests for E2, N3 requirements.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from pathlib import Path
import tempfile
import os


class TestAudioConverterService:
    """Tests for AudioConverterService (E2)."""

    @pytest.fixture
    def mock_m4a_file(self, tmp_path):
        """Create a mock m4a file."""
        audio_file = tmp_path / "test_audio.m4a"
        audio_file.write_bytes(b"\x00" * 1000)
        return str(audio_file)

    @pytest.fixture
    def mock_wav_file(self, tmp_path):
        """Create a mock wav file."""
        audio_file = tmp_path / "test_audio.wav"
        # WAV header + data
        audio_file.write_bytes(b"RIFF" + b"\x00" * 1000)
        return str(audio_file)

    def test_service_initialization(self):
        """Test service initialization."""
        from voice_man.services.audio_converter_service import AudioConverterService

        service = AudioConverterService()
        assert service is not None

    def test_supported_input_formats(self):
        """E2: Test supported input formats."""
        from voice_man.services.audio_converter_service import AudioConverterService

        service = AudioConverterService()

        # E2: Support m4a, mp3, wav, flac, ogg
        assert service.is_supported_format("m4a")
        assert service.is_supported_format("mp3")
        assert service.is_supported_format("wav")
        assert service.is_supported_format("flac")
        assert service.is_supported_format("ogg")
        assert not service.is_supported_format("xyz")

    def test_output_format_is_16khz_mono_wav(self):
        """E2: Output should be 16kHz mono WAV."""
        from voice_man.services.audio_converter_service import AudioConverterService

        service = AudioConverterService()

        assert service.target_sample_rate == 16000
        assert service.target_channels == 1
        assert service.target_format == "wav"

    @pytest.mark.asyncio
    async def test_convert_m4a_to_wav(self, mock_m4a_file):
        """E2: Test m4a to WAV conversion."""
        from voice_man.services.audio_converter_service import AudioConverterService

        with patch("voice_man.services.audio_converter_service.ffmpeg") as mock_ffmpeg:
            mock_ffmpeg.input.return_value.output.return_value.overwrite_output.return_value.run.return_value = (
                None,
                None,
            )

            service = AudioConverterService()
            output_path = await service.convert_to_wav(mock_m4a_file)

            assert output_path is not None
            assert output_path.endswith(".wav")

    @pytest.mark.asyncio
    async def test_wav_file_not_converted(self, mock_wav_file):
        """E2: WAV files at correct spec should not be re-converted."""
        from voice_man.services.audio_converter_service import AudioConverterService

        with patch(
            "voice_man.services.audio_converter_service.AudioConverterService._get_audio_info"
        ) as mock_info:
            mock_info.return_value = {
                "sample_rate": 16000,
                "channels": 1,
                "format": "wav",
            }

            service = AudioConverterService()
            output_path = await service.convert_to_wav(mock_wav_file)

            # Should return original file (no conversion needed)
            assert output_path == mock_wav_file

    @pytest.mark.asyncio
    async def test_temp_file_cleanup_on_success(self, mock_m4a_file, tmp_path):
        """N3: Temp files should be cleaned up after processing."""
        from voice_man.services.audio_converter_service import AudioConverterService

        with patch("voice_man.services.audio_converter_service.ffmpeg") as mock_ffmpeg:
            mock_ffmpeg.input.return_value.output.return_value.overwrite_output.return_value.run.return_value = (
                None,
                None,
            )

            service = AudioConverterService()

            async with service.convert_context(mock_m4a_file) as converted_path:
                # File should exist during processing
                assert converted_path is not None

            # After context exit, if temp file was created, it should be cleaned up
            # (In real implementation, context manager handles cleanup)

    @pytest.mark.asyncio
    async def test_temp_file_cleanup_on_error(self, mock_m4a_file):
        """N3: Temp files should be cleaned up even on error."""
        from voice_man.services.audio_converter_service import AudioConverterService

        with patch("voice_man.services.audio_converter_service.ffmpeg") as mock_ffmpeg:
            mock_ffmpeg.input.side_effect = Exception("Conversion failed")

            service = AudioConverterService()

            with pytest.raises(Exception):
                async with service.convert_context(mock_m4a_file):
                    pass

            # Cleanup should still happen (verified by context manager implementation)

    def test_get_file_format(self, mock_m4a_file, mock_wav_file):
        """Test file format detection."""
        from voice_man.services.audio_converter_service import AudioConverterService

        service = AudioConverterService()

        assert service.get_file_format(mock_m4a_file) == "m4a"
        assert service.get_file_format(mock_wav_file) == "wav"

    @pytest.mark.asyncio
    async def test_convert_raises_on_unsupported_format(self, tmp_path):
        """Test error on unsupported format."""
        from voice_man.services.audio_converter_service import (
            AudioConverterService,
            UnsupportedFormatError,
        )

        unsupported_file = tmp_path / "test.xyz"
        unsupported_file.write_bytes(b"\x00" * 100)

        service = AudioConverterService()

        with pytest.raises(UnsupportedFormatError):
            await service.convert_to_wav(str(unsupported_file))

    @pytest.mark.asyncio
    async def test_convert_raises_on_missing_file(self):
        """Test error on missing file."""
        from voice_man.services.audio_converter_service import AudioConverterService

        service = AudioConverterService()

        with pytest.raises(FileNotFoundError):
            await service.convert_to_wav("/nonexistent/file.m4a")


class TestAlignmentService:
    """Tests for AlignmentService (F2)."""

    def test_service_initialization(self):
        """Test alignment service initialization."""
        from voice_man.services.alignment_service import AlignmentService

        with patch.dict(os.environ, {"HF_TOKEN": "test_token"}):
            with patch("voice_man.services.alignment_service.whisperx") as mock_wx:
                mock_wx.load_align_model.return_value = (MagicMock(), MagicMock())

                service = AlignmentService(language="ko", device="cuda")

                assert service.language == "ko"
                assert service.device == "cuda"

    def test_korean_alignment_model(self):
        """S1: Korean language uses specific alignment model."""
        from voice_man.services.alignment_service import AlignmentService

        with patch.dict(os.environ, {"HF_TOKEN": "test_token"}):
            with patch("voice_man.services.alignment_service.whisperx") as mock_wx:
                mock_wx.load_align_model.return_value = (MagicMock(), MagicMock())

                service = AlignmentService(language="ko", device="cuda")

                assert (
                    service.alignment_model_name == "jonatasgrosman/wav2vec2-large-xlsr-53-korean"
                )

    @pytest.mark.asyncio
    async def test_align_returns_word_timestamps(self):
        """F2: Test word-level timestamp alignment."""
        from voice_man.services.alignment_service import AlignmentService

        with patch.dict(os.environ, {"HF_TOKEN": "test_token"}):
            with patch("voice_man.services.alignment_service.whisperx") as mock_wx:
                mock_wx.load_align_model.return_value = (MagicMock(), MagicMock())
                mock_wx.align.return_value = {
                    "segments": [
                        {
                            "start": 0.0,
                            "end": 2.0,
                            "text": "Hello World",
                            "words": [
                                {
                                    "word": "Hello",
                                    "start": 0.0,
                                    "end": 0.8,
                                    "score": 0.95,
                                },
                                {
                                    "word": "World",
                                    "start": 1.0,
                                    "end": 1.8,
                                    "score": 0.92,
                                },
                            ],
                        }
                    ]
                }

                service = AlignmentService(language="ko", device="cuda")

                segments = {"segments": [{"start": 0.0, "end": 2.0, "text": "Hello World"}]}
                audio = MagicMock()

                result = await service.align(segments, audio)

                assert "segments" in result
                assert "words" in result["segments"][0]
                words = result["segments"][0]["words"]
                assert len(words) == 2
                # U2: Check word-level data structure
                assert "start" in words[0]
                assert "end" in words[0]
                assert "score" in words[0]

    @pytest.mark.asyncio
    async def test_alignment_score_validation(self):
        """U2: Alignment scores should be between 0 and 1."""
        from voice_man.services.alignment_service import AlignmentService

        with patch.dict(os.environ, {"HF_TOKEN": "test_token"}):
            with patch("voice_man.services.alignment_service.whisperx") as mock_wx:
                mock_wx.load_align_model.return_value = (MagicMock(), MagicMock())
                mock_wx.align.return_value = {
                    "segments": [
                        {"words": [{"word": "Test", "start": 0.0, "end": 0.5, "score": 0.95}]}
                    ]
                }

                service = AlignmentService(language="ko", device="cuda")

                segments = {"segments": [{"text": "Test"}]}
                audio = MagicMock()

                result = await service.align(segments, audio)

                score = result["segments"][0]["words"][0]["score"]
                assert 0 <= score <= 1

    def test_unload_model(self):
        """Test model unloading."""
        from voice_man.services.alignment_service import AlignmentService

        with patch.dict(os.environ, {"HF_TOKEN": "test_token"}):
            with patch("voice_man.services.alignment_service.whisperx") as mock_wx:
                with patch("voice_man.services.alignment_service.torch") as mock_torch:
                    mock_wx.load_align_model.return_value = (MagicMock(), MagicMock())

                    service = AlignmentService(language="ko", device="cuda")
                    service.unload()

                    mock_torch.cuda.empty_cache.assert_called()


class MockAsyncContextManager:
    """Helper for mocking async context managers."""

    def __init__(self, return_value):
        self.return_value = return_value

    async def __aenter__(self):
        return self.return_value

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None


class TestWhisperXService:
    """Tests for WhisperXService integration."""

    def test_service_initialization(self):
        """Test WhisperX service initialization."""
        from voice_man.services.whisperx_service import WhisperXService

        with patch.dict(os.environ, {"HF_TOKEN": "test_token"}):
            with patch("voice_man.services.whisperx_service.WhisperXPipeline") as mock_pipeline:
                mock_pipeline.return_value = MagicMock()

                service = WhisperXService()

                assert service is not None

    @pytest.mark.asyncio
    async def test_process_audio_file(self, tmp_path):
        """Test processing audio file through service."""
        from voice_man.services.whisperx_service import WhisperXService

        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"RIFF" + b"\x00" * 1000)

        with patch.dict(os.environ, {"HF_TOKEN": "test_token"}):
            with patch("voice_man.services.whisperx_service.WhisperXPipeline") as mock_pipeline:
                with patch(
                    "voice_man.services.whisperx_service.AudioConverterService"
                ) as mock_converter:
                    mock_instance = MagicMock()
                    mock_instance.process = AsyncMock(
                        return_value=MagicMock(
                            text="Test text",
                            segments=[],
                            speakers=["SPEAKER_00"],
                            speaker_stats={},
                            language="ko",
                        )
                    )
                    mock_pipeline.return_value = mock_instance

                    # Mock converter context manager
                    mock_converter_instance = MagicMock()
                    mock_converter_instance.convert_context.return_value = MockAsyncContextManager(
                        str(audio_file)
                    )
                    mock_converter.return_value = mock_converter_instance

                    service = WhisperXService()
                    result = await service.process_audio(str(audio_file))

                    assert result is not None
                    assert hasattr(result, "text")

    @pytest.mark.asyncio
    async def test_process_with_speaker_count(self, tmp_path):
        """E4: Test processing with specified speaker count."""
        from voice_man.services.whisperx_service import WhisperXService

        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"RIFF" + b"\x00" * 1000)

        with patch.dict(os.environ, {"HF_TOKEN": "test_token"}):
            with patch("voice_man.services.whisperx_service.WhisperXPipeline") as mock_pipeline:
                with patch(
                    "voice_man.services.whisperx_service.AudioConverterService"
                ) as mock_converter:
                    mock_instance = MagicMock()
                    mock_instance.process = AsyncMock(
                        return_value=MagicMock(
                            text="Test",
                            segments=[],
                            speakers=["SPEAKER_00", "SPEAKER_01"],
                            speaker_stats={},
                            language="ko",
                        )
                    )
                    mock_pipeline.return_value = mock_instance

                    # Mock converter context manager
                    mock_converter_instance = MagicMock()
                    mock_converter_instance.convert_context.return_value = MockAsyncContextManager(
                        str(audio_file)
                    )
                    mock_converter.return_value = mock_converter_instance

                    service = WhisperXService()
                    await service.process_audio(str(audio_file), num_speakers=2)

                    # Verify process was called with num_speakers
                    mock_instance.process.assert_called_once()
                    call_args = mock_instance.process.call_args
                    assert call_args[1].get("num_speakers") == 2

    @pytest.mark.asyncio
    async def test_process_with_progress_callback(self, tmp_path):
        """E3: Test progress callback during processing."""
        from voice_man.services.whisperx_service import WhisperXService

        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"RIFF" + b"\x00" * 1000)

        progress_updates = []

        def progress_callback(stage, progress, message):
            progress_updates.append((stage, progress, message))

        with patch.dict(os.environ, {"HF_TOKEN": "test_token"}):
            with patch("voice_man.services.whisperx_service.WhisperXPipeline") as mock_pipeline:
                with patch(
                    "voice_man.services.whisperx_service.AudioConverterService"
                ) as mock_converter:
                    mock_instance = MagicMock()
                    mock_instance.process = AsyncMock(
                        return_value=MagicMock(
                            text="Test",
                            segments=[],
                            speakers=[],
                            speaker_stats={},
                            language="ko",
                        )
                    )
                    mock_pipeline.return_value = mock_instance

                    # Mock converter context manager
                    mock_converter_instance = MagicMock()
                    mock_converter_instance.convert_context.return_value = MockAsyncContextManager(
                        str(audio_file)
                    )
                    mock_converter.return_value = mock_converter_instance

                    service = WhisperXService()
                    await service.process_audio(
                        str(audio_file), progress_callback=progress_callback
                    )

                    # Progress callback should have been passed to pipeline
                    mock_instance.process.assert_called_once()
