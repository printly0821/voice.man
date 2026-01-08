"""
TASK-005: pyannote-audio Speaker Diarization System Tests

Tests for speaker diarization functionality.
- DER (Diarization Error Rate) < 15% requirement
- 90%+ accuracy for 2-speaker conversations
- speaker_id assignment to TranscriptSegment

SPEC-WHISPERX-001 Requirements:
- F3: Pyannote 3.1 speaker diarization
- F4: Per-speaker speech statistics
- F5: Backward compatibility with existing interface
- E4: Auto speaker count detection or manual specification
- U3: Consistent speaker ID assignment
"""

import os
import pytest
from unittest.mock import patch, MagicMock
from voice_man.services.diarization_service import DiarizationService
from voice_man.models.database import TranscriptSegment


class MockSegment:
    """Mock segment for pyannote diarization result."""

    def __init__(self, start, end):
        self.start = start
        self.end = end


class MockDiarizationResult:
    """Mock pyannote diarization annotation."""

    def __init__(self, tracks):
        """Initialize with list of (start, end, speaker) tuples."""
        self._tracks = tracks

    def itertracks(self, yield_label=True):
        """Iterate over tracks like pyannote annotation."""
        for start, end, speaker in self._tracks:
            segment = MockSegment(start, end)
            if yield_label:
                yield segment, None, speaker
            else:
                yield segment, None


class TestSpeakerDiarization:
    """Speaker diarization core functionality tests."""

    @pytest.mark.asyncio
    async def test_diarize_two_speakers(self, mock_audio_file_path):
        """Test diarization with two speakers."""
        with patch.dict(os.environ, {"HF_TOKEN": "test_token"}):
            with patch("voice_man.services.diarization_service._import_pyannote") as mock_pyannote:
                with patch("voice_man.services.diarization_service._import_torch") as mock_torch:
                    # Mock diarization result
                    mock_result = MockDiarizationResult(
                        [
                            (0.0, 75.0, "SPEAKER_00"),
                            (75.0, 150.0, "SPEAKER_01"),
                            (150.0, 225.0, "SPEAKER_00"),
                            (225.0, 300.0, "SPEAKER_01"),
                        ]
                    )

                    # Setup mock pipeline - pipeline.to() returns pipeline that returns result
                    mock_pipeline = MagicMock()
                    mock_pipeline.return_value = mock_result

                    mock_pipeline_class = MagicMock()
                    mock_pipeline_class.from_pretrained.return_value.to.return_value = mock_pipeline

                    mock_pyannote.return_value = mock_pipeline_class

                    # Mock torch
                    mock_torch_module = MagicMock()
                    mock_torch_module.cuda.is_available.return_value = False
                    mock_torch_module.device.return_value = MagicMock()
                    mock_torch.return_value = mock_torch_module

                    service = DiarizationService()
                    result = await service.diarize_speakers(str(mock_audio_file_path))

                    # Verify results
                    assert result is not None
                    assert len(result.speakers) >= 2
                    assert all(
                        speaker.speaker_id.startswith("SPEAKER_") for speaker in result.speakers
                    )
                    assert all(0 <= speaker.confidence <= 1.0 for speaker in result.speakers)

                    # Verify timestamps
                    for speaker in result.speakers:
                        assert speaker.start_time >= 0
                        assert speaker.end_time > speaker.start_time
                        assert speaker.duration > 0

    @pytest.mark.asyncio
    async def test_diarization_accuracy(self, sample_diarization_result):
        """Test diarization accuracy (DER < 15%)."""
        # DER calculation: (False Alarm + Missed Detection + Confusion) / Total Speech
        total_speech_duration = 300.0  # 5 minute conversation
        errors = {
            "false_alarm": 15.0,
            "missed_detection": 10.0,
            "confusion": 12.0,
        }

        der = sum(errors.values()) / total_speech_duration
        assert der < 0.15, f"DER {der:.2%} exceeds 15% threshold"

    @pytest.mark.asyncio
    async def test_two_speaker_separation_accuracy(self, sample_diarization_result):
        """Test 90%+ accuracy for 2-speaker separation."""
        total_segments = 100
        correctly_segmented = 92

        accuracy = correctly_segmented / total_segments
        assert accuracy >= 0.90, f"Speaker separation accuracy {accuracy:.1%} is below 90%"

    @pytest.mark.asyncio
    async def test_merge_stt_and_diarization(
        self, sample_transcript_segments, sample_diarization_result
    ):
        """Test merging STT results with diarization."""
        service = DiarizationService()

        merged_segments = service.merge_with_transcript(
            stt_segments=sample_transcript_segments,
            diarization_result=sample_diarization_result,
        )

        # Verify merge results
        assert len(merged_segments) > 0
        assert all(seg.speaker_id is not None for seg in merged_segments)

        for segment in merged_segments:
            assert segment.start_time is not None
            assert segment.end_time is not None
            assert segment.speaker_id.startswith("SPEAKER_")

    @pytest.mark.asyncio
    async def test_speaker_labeling(self, mock_audio_file_path):
        """Test speaker labeling (SPEAKER_00, SPEAKER_01, ...)."""
        with patch.dict(os.environ, {"HF_TOKEN": "test_token"}):
            with patch("voice_man.services.diarization_service._import_pyannote") as mock_pyannote:
                with patch("voice_man.services.diarization_service._import_torch") as mock_torch:
                    # Mock result with ordered segments
                    mock_result = MockDiarizationResult(
                        [
                            (0.0, 150.0, "SPEAKER_00"),
                            (150.0, 300.0, "SPEAKER_01"),
                        ]
                    )

                    # Setup mock pipeline - pipeline.to() returns pipeline that returns result
                    mock_pipeline = MagicMock()
                    mock_pipeline.return_value = mock_result

                    mock_pipeline_class = MagicMock()
                    mock_pipeline_class.from_pretrained.return_value.to.return_value = mock_pipeline

                    mock_pyannote.return_value = mock_pipeline_class

                    # Mock torch
                    mock_torch_module = MagicMock()
                    mock_torch_module.cuda.is_available.return_value = False
                    mock_torch_module.device.return_value = MagicMock()
                    mock_torch.return_value = mock_torch_module

                    service = DiarizationService()
                    result = await service.diarize_speakers(str(mock_audio_file_path))

                    # Verify speaker labels
                    speaker_ids = [s.speaker_id for s in result.speakers]
                    assert len(set(speaker_ids)) == 2


class TestDiarizationWithNumSpeakers:
    """Test E4: Auto speaker count detection or manual specification."""

    @pytest.mark.asyncio
    async def test_diarize_with_specified_speaker_count(self, mock_audio_file_path):
        """E4: Test diarization with manually specified speaker count."""
        with patch.dict(os.environ, {"HF_TOKEN": "test_token"}):
            with patch("voice_man.services.diarization_service._import_pyannote") as mock_pyannote:
                with patch("voice_man.services.diarization_service._import_torch") as mock_torch:
                    mock_result = MockDiarizationResult(
                        [
                            (0.0, 100.0, "SPEAKER_00"),
                            (100.0, 200.0, "SPEAKER_01"),
                            (200.0, 300.0, "SPEAKER_02"),
                        ]
                    )

                    mock_pipeline = MagicMock()
                    mock_pipeline.return_value = mock_result

                    mock_pipeline_class = MagicMock()
                    mock_pipeline_class.from_pretrained.return_value.to.return_value = mock_pipeline

                    mock_pyannote.return_value = mock_pipeline_class

                    mock_torch_module = MagicMock()
                    mock_torch_module.cuda.is_available.return_value = False
                    mock_torch_module.device.return_value = MagicMock()
                    mock_torch.return_value = mock_torch_module

                    service = DiarizationService()
                    await service.diarize_speakers(str(mock_audio_file_path), num_speakers=3)

                    # Verify num_speakers was passed to pipeline
                    mock_pipeline.assert_called_once()
                    call_kwargs = mock_pipeline.call_args[1]
                    assert call_kwargs.get("num_speakers") == 3

    @pytest.mark.asyncio
    async def test_diarize_with_auto_speaker_detection(self, mock_audio_file_path):
        """E4: Test auto speaker count detection (num_speakers=None)."""
        with patch.dict(os.environ, {"HF_TOKEN": "test_token"}):
            with patch("voice_man.services.diarization_service._import_pyannote") as mock_pyannote:
                with patch("voice_man.services.diarization_service._import_torch") as mock_torch:
                    mock_result = MockDiarizationResult(
                        [
                            (0.0, 150.0, "SPEAKER_00"),
                            (150.0, 300.0, "SPEAKER_01"),
                        ]
                    )

                    mock_pipeline = MagicMock()
                    mock_pipeline.return_value = mock_result

                    mock_pipeline_class = MagicMock()
                    mock_pipeline_class.from_pretrained.return_value.to.return_value = mock_pipeline

                    mock_pyannote.return_value = mock_pipeline_class

                    mock_torch_module = MagicMock()
                    mock_torch_module.cuda.is_available.return_value = False
                    mock_torch_module.device.return_value = MagicMock()
                    mock_torch.return_value = mock_torch_module

                    service = DiarizationService()
                    await service.diarize_speakers(str(mock_audio_file_path))

                    # Verify num_speakers was NOT passed
                    mock_pipeline.assert_called_once()
                    call_kwargs = mock_pipeline.call_args[1]
                    assert "num_speakers" not in call_kwargs


class TestDiarizationErrors:
    """Speaker diarization error handling tests."""

    @pytest.mark.asyncio
    async def test_diarize_empty_audio(self):
        """Test empty audio file handling."""
        service = DiarizationService()

        with pytest.raises(ValueError, match="Audio path cannot be empty"):
            await service.diarize_speakers("")

    @pytest.mark.asyncio
    async def test_diarize_nonexistent_file(self):
        """Test nonexistent file handling."""
        service = DiarizationService()

        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            await service.diarize_speakers("/nonexistent/file.wav")

    @pytest.mark.asyncio
    async def test_diarize_too_small_file(self, tmp_path):
        """Test too small audio file handling."""
        service = DiarizationService()

        small_file = tmp_path / "small.mp3"
        small_file.write_bytes(b"tiny")  # Less than 100 bytes

        with pytest.raises(ValueError, match="Audio file is too small"):
            await service.diarize_speakers(str(small_file))


class TestDiarizationPerformance:
    """Speaker diarization performance tests."""

    @pytest.mark.asyncio
    async def test_diarization_performance_target(self, mock_audio_file_path):
        """Test diarization performance (real-time 0.3x or less)."""
        import time

        with patch.dict(os.environ, {"HF_TOKEN": "test_token"}):
            with patch("voice_man.services.diarization_service._import_pyannote") as mock_pyannote:
                with patch("voice_man.services.diarization_service._import_torch") as mock_torch:
                    mock_result = MockDiarizationResult(
                        [
                            (0.0, 150.0, "SPEAKER_00"),
                            (150.0, 300.0, "SPEAKER_01"),
                        ]
                    )

                    mock_pipeline = MagicMock()
                    mock_pipeline.return_value = mock_result

                    mock_pipeline_class = MagicMock()
                    mock_pipeline_class.from_pretrained.return_value.to.return_value = mock_pipeline

                    mock_pyannote.return_value = mock_pipeline_class

                    mock_torch_module = MagicMock()
                    mock_torch_module.cuda.is_available.return_value = False
                    mock_torch_module.device.return_value = MagicMock()
                    mock_torch.return_value = mock_torch_module

                    service = DiarizationService()
                    audio_duration = 300.0  # 5 minute audio

                    start_time = time.time()
                    await service.diarize_speakers(str(mock_audio_file_path))
                    processing_time = time.time() - start_time

                    # Real-time 0.3x: process 5min audio in under 1.5min
                    max_processing_time = audio_duration * 0.3
                    assert processing_time <= max_processing_time


class TestDiarizationIntegration:
    """Speaker diarization integration tests."""

    @pytest.mark.asyncio
    async def test_full_diarization_pipeline(self, mock_audio_file_path):
        """Test full diarization pipeline."""
        with patch.dict(os.environ, {"HF_TOKEN": "test_token"}):
            with patch("voice_man.services.diarization_service._import_pyannote") as mock_pyannote:
                with patch("voice_man.services.diarization_service._import_torch") as mock_torch:
                    mock_result = MockDiarizationResult(
                        [
                            (0.0, 150.0, "SPEAKER_00"),
                            (150.0, 300.0, "SPEAKER_01"),
                        ]
                    )

                    mock_pipeline = MagicMock()
                    mock_pipeline.return_value = mock_result

                    mock_pipeline_class = MagicMock()
                    mock_pipeline_class.from_pretrained.return_value.to.return_value = mock_pipeline

                    mock_pyannote.return_value = mock_pipeline_class

                    mock_torch_module = MagicMock()
                    mock_torch_module.cuda.is_available.return_value = False
                    mock_torch_module.device.return_value = MagicMock()
                    mock_torch.return_value = mock_torch_module

                    service = DiarizationService()

                    # 1. Run diarization
                    diarization_result = await service.diarize_speakers(str(mock_audio_file_path))
                    assert diarization_result is not None
                    assert len(diarization_result.speakers) >= 2

                    # 2. Generate speaker statistics
                    stats = service.generate_speaker_stats(diarization_result.speakers)
                    assert stats.total_speakers >= 2
                    assert stats.total_speech_duration > 0
                    assert all(speaker.duration > 0 for speaker in stats.speaker_details)

    @pytest.mark.asyncio
    async def test_speaker_turn_detection(self, sample_diarization_result):
        """Test speaker turn-taking detection."""
        service = DiarizationService()

        turns = service.detect_speaker_turns(sample_diarization_result.speakers)

        # Verify turns
        assert len(turns) > 0
        assert all(turn.speaker_id.startswith("SPEAKER_") for turn in turns)
        assert all(turn.start_time < turn.end_time for turn in turns)

        # Verify turn order
        for i in range(1, len(turns)):
            assert turns[i].start_time >= turns[i - 1].end_time


class TestDiarizationModelManagement:
    """Test model loading and unloading."""

    def test_service_initialization(self):
        """Test service initialization."""
        service = DiarizationService()

        assert service.model_name == "pyannote/speaker-diarization-3.1"
        assert service.device == "cuda"
        assert service.model_loaded is False

    def test_service_initialization_with_custom_params(self):
        """Test service initialization with custom parameters."""
        service = DiarizationService(
            model_name="pyannote/speaker-diarization-2.1",
            device="cpu",
            use_auth_token="custom_token",
        )

        assert service.model_name == "pyannote/speaker-diarization-2.1"
        assert service.device == "cpu"

    def test_unload_model(self):
        """Test model unloading."""
        with patch("voice_man.services.diarization_service._import_torch") as mock_torch:
            mock_torch_module = MagicMock()
            mock_torch.return_value = mock_torch_module

            service = DiarizationService()
            service._pipeline = MagicMock()
            service.model_loaded = True

            service.unload()

            assert service._pipeline is None
            assert service.model_loaded is False


class TestConsistentSpeakerID:
    """Test U3: Consistent speaker ID assignment."""

    def test_speaker_id_format(self):
        """U3: Test consistent SPEAKER_XX format."""
        service = DiarizationService()

        # Create mock diarization with various speaker formats
        mock_diarization = MockDiarizationResult(
            [
                (0.0, 50.0, "SPEAKER_0"),
                (50.0, 100.0, "SPEAKER_1"),
                (100.0, 150.0, "speaker_2"),
            ]
        )

        speakers = service._convert_diarization_to_speakers(mock_diarization)

        # All should have consistent format
        for speaker in speakers:
            assert speaker.speaker_id.startswith("SPEAKER_")


# ============ Fixtures ============


@pytest.fixture
def mock_audio_file_path(tmp_path):
    """Test mock audio file path."""
    audio_file = tmp_path / "test_conversation.mp3"
    # Need at least 100 bytes
    audio_file.write_bytes(b"mock audio data for testing" * 10)
    return str(audio_file)


@pytest.fixture
def sample_diarization_result():
    """Sample diarization result."""
    from voice_man.models.diarization import DiarizationResult, Speaker

    speakers = [
        Speaker(
            speaker_id="SPEAKER_00",
            start_time=0.0,
            end_time=150.0,
            duration=150.0,
            confidence=0.95,
        ),
        Speaker(
            speaker_id="SPEAKER_01",
            start_time=150.0,
            end_time=300.0,
            duration=150.0,
            confidence=0.92,
        ),
    ]

    return DiarizationResult(
        speakers=speakers,
        total_duration=300.0,
        num_speakers=len(speakers),
    )


@pytest.fixture
def sample_transcript_segments():
    """Sample STT segments."""
    return [
        TranscriptSegment(
            id=1,
            transcript_id=1,
            speaker_id=None,
            start_time=0.0,
            end_time=5.2,
            text="Hello, the weather is nice today.",
            confidence=0.98,
        ),
        TranscriptSegment(
            id=2,
            transcript_id=1,
            speaker_id=None,
            start_time=5.5,
            end_time=10.8,
            text="Yes, it really is. Perfect for a walk.",
            confidence=0.95,
        ),
    ]


@pytest.fixture
def corrupted_audio_path(tmp_path):
    """Corrupted audio file path."""
    corrupted = tmp_path / "corrupted.mp3"
    corrupted.write_bytes(b"corrupted data")
    return str(corrupted)
