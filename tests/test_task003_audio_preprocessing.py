"""
TASK-003: FFmpeg 오디오 전처리 파이프라인 테스트

테스트 목적:
- 오디오 메타데이터 추출 기능 검증
- 손상된 파일 감지 기능 검증
- 다양한 오디오 형식 처리 검증
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.voice_man import services


class TestAudioMetadataExtraction:
    """오디오 메타데이터 추출 테스트"""

    @pytest.mark.asyncio
    async def test_extract_metadata_from_mp3(self):
        """
        Given: 유효한 MP3 파일
        When: 메타데이터 추출 요청
        Then: 정확한 메타데이터 반환 (duration, sample_rate, channels)
        """
        # Given
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_file = Path(tmp.name)

        # When
        with patch("ffmpeg.probe") as mock_probe:
            mock_probe.return_value = {
                "format": {"duration": "5.0", "size": "800000"},
                "streams": [{"sample_rate": "16000", "channels": 1, "codec_name": "mp3"}],
            }

            metadata = await services.extract_audio_metadata(tmp_file)

        # Then
        assert metadata.duration_seconds > 0
        assert metadata.sample_rate > 0
        assert metadata.channels in [1, 2]
        assert metadata.format == "mp3"

    @pytest.mark.asyncio
    async def test_extract_metadata_from_wav(self):
        """
        Given: 유효한 WAV 파일
        When: 메타데이터 추출 요청
        Then: 정확한 메타데이터 반환
        """
        # Given
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_file = Path(tmp.name)

        # When
        with patch("ffmpeg.probe") as mock_probe:
            mock_probe.return_value = {
                "format": {"duration": "3.5", "size": "560000"},
                "streams": [{"sample_rate": "44100", "channels": 2, "codec_name": "pcm_s16le"}],
            }

            metadata = await services.extract_audio_metadata(tmp_file)

        # Then
        assert metadata.duration_seconds > 0
        assert metadata.sample_rate > 0
        assert metadata.format == "wav"

    @pytest.mark.asyncio
    async def test_extract_metadata_duration_accuracy(self):
        """
        Given: 3초 길이의 오디오 파일
        When: 메타데이터 추출
        Then: duration ±0.1초 오차 이내
        """
        # Given
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_file = Path(tmp.name)

        # When
        with patch("ffmpeg.probe") as mock_probe:
            mock_probe.return_value = {
                "format": {"duration": "3.0", "size": "480000"},
                "streams": [{"sample_rate": "16000", "channels": 1, "codec_name": "mp3"}],
            }

            metadata = await services.extract_audio_metadata(tmp_file)

        # Then
        assert 2.9 <= metadata.duration_seconds <= 3.1

    @pytest.mark.asyncio
    async def test_extract_metadata_includes_sample_rate(self):
        """
        Given: 16000Hz 샘플레이트 오디오
        When: 메타데이터 추출
        Then: sample_rate = 16000
        """
        # Given
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_file = Path(tmp.name)

        # When
        with patch("ffmpeg.probe") as mock_probe:
            mock_probe.return_value = {
                "format": {"duration": "5.0", "size": "800000"},
                "streams": [{"sample_rate": "16000", "channels": 2, "codec_name": "pcm_s16le"}],
            }

            metadata = await services.extract_audio_metadata(tmp_file)

        # Then
        assert metadata.sample_rate == 16000

    @pytest.mark.asyncio
    async def test_extract_metadata_includes_channels(self):
        """
        Given: 스테레오 오디오 (2채널)
        When: 메타데이터 추출
        Then: channels = 2
        """
        # Given
        with tempfile.NamedTemporaryFile(suffix=".flac", delete=False) as tmp:
            tmp_file = Path(tmp.name)

        # When
        with patch("ffmpeg.probe") as mock_probe:
            mock_probe.return_value = {
                "format": {"duration": "4.0", "size": "640000"},
                "streams": [{"sample_rate": "44100", "channels": 2, "codec_name": "flac"}],
            }

            metadata = await services.extract_audio_metadata(tmp_file)

        # Then
        assert metadata.channels == 2


class TestCorruptedFileDetection:
    """손상된 파일 감지 테스트"""

    @pytest.mark.asyncio
    async def test_detect_valid_mp3_file(self):
        """
        Given: 유효한 MP3 파일
        When: 손상 감지 요청
        Then: False 반환 (손상되지 않음)
        """
        # Given
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_file = Path(tmp.name)
            tmp.write(b"ID3" + b"\x00" * 1000)

        # When
        with patch("ffmpeg.probe") as mock_probe:
            mock_probe.return_value = {
                "format": {"duration": "5.0", "size": "800000"},
                "streams": [{"sample_rate": "16000", "channels": 1}],
            }

            is_corrupted = await services.detect_corrupted_file(tmp_file)

        # Then
        assert is_corrupted is False

    @pytest.mark.asyncio
    async def test_detect_corrupted_missing_header(self):
        """
        Given: 헤더가 손상된 MP3 파일
        When: 손상 감지 요청
        Then: True 반환 (손상됨)
        """
        # Given
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_file = Path(tmp.name)
            tmp.write(b"CORRUPTED" + b"\x00" * 1000)

        # When
        with patch("ffmpeg.probe") as mock_probe:
            mock_probe.side_effect = Exception("Invalid data")

            is_corrupted = await services.detect_corrupted_file(tmp_file)

        # Then
        assert is_corrupted is True

    @pytest.mark.asyncio
    async def test_detect_corrupted_zero_duration(self):
        """
        Given: 재생 시간이 0초인 파일
        When: 손상 감지 요청
        Then: True 반환 (손상됨)
        """
        # Given
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_file = Path(tmp.name)

        # When
        with patch("ffmpeg.probe") as mock_probe:
            mock_probe.return_value = {
                "format": {"duration": "0.0", "size": "0"},
                "streams": [{"sample_rate": "16000", "channels": 1}],
            }

            is_corrupted = await services.detect_corrupted_file(tmp_file)

        # Then
        assert is_corrupted is True

    @pytest.mark.asyncio
    async def test_detect_corrupted_invalid_codec(self):
        """
        Given: 지원하지 않는 코덱
        When: 손상 감지 요청
        Then: True 반환 (손상됨)
        """
        # Given
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_file = Path(tmp.name)

        # When
        with patch("ffmpeg.probe") as mock_probe:
            mock_probe.side_effect = Exception("Unsupported codec")

            is_corrupted = await services.detect_corrupted_file(tmp_file)

        # Then
        assert is_corrupted is True


class TestAudioFormatsSupport:
    """다양한 오디오 형식 지원 테스트"""

    @pytest.mark.asyncio
    async def test_process_m4a_format(self):
        """
        Given: M4A 파일
        When: 메타데이터 추출
        Then: 성공적으로 메타데이터 반환
        """
        # Given
        with tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as tmp:
            tmp_file = Path(tmp.name)

        # When
        with patch("ffmpeg.probe") as mock_probe:
            mock_probe.return_value = {
                "format": {"duration": "3.5", "size": "560000"},
                "streams": [{"sample_rate": "44100", "channels": 2, "codec_name": "aac"}],
            }

            metadata = await services.extract_audio_metadata(tmp_file)

        # Then
        assert metadata.format == "m4a"
        assert metadata.duration_seconds == 3.5

    @pytest.mark.asyncio
    async def test_process_flac_format(self):
        """
        Given: FLAC 파일
        When: 메타데이터 추출
        Then: 성공적으로 메타데이터 반환
        """
        # Given
        with tempfile.NamedTemporaryFile(suffix=".flac", delete=False) as tmp:
            tmp_file = Path(tmp.name)

        # When
        with patch("ffmpeg.probe") as mock_probe:
            mock_probe.return_value = {
                "format": {"duration": "2.8", "size": "448000"},
                "streams": [{"sample_rate": "48000", "channels": 2, "codec_name": "flac"}],
            }

            metadata = await services.extract_audio_metadata(tmp_file)

        # Then
        assert metadata.format == "flac"
        assert metadata.duration_seconds == 2.8

    @pytest.mark.asyncio
    async def test_process_ogg_format(self):
        """
        Given: OGG 파일
        When: 메타데이터 추출
        Then: 성공적으로 메타데이터 반환
        """
        # Given
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
            tmp_file = Path(tmp.name)

        # When
        with patch("ffmpeg.probe") as mock_probe:
            mock_probe.return_value = {
                "format": {"duration": "4.2", "size": "672000"},
                "streams": [{"sample_rate": "48000", "channels": 1, "codec_name": "vorbis"}],
            }

            metadata = await services.extract_audio_metadata(tmp_file)

        # Then
        assert metadata.format == "ogg"
        assert metadata.duration_seconds == 4.2
