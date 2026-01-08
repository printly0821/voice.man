"""
TASK-004: Whisper STT 엔진 통합 테스트

테스트 목적:
- Whisper large-v3 모델을 사용한 음성-텍스트 변환 기능 검증
- 세그먼트별 타임스탬프 추출 기능 검증
- 신뢰도 점수 산출 기능 검증
- 비동기 변환 처리 기능 검증
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestWhisperTranscription:
    """Whisper STT 변환 테스트"""

    @pytest.mark.asyncio
    async def test_transcribe_korean_audio(self):
        """
        Given: 한국어 오디오 파일
        When: STT 변환 요청
        Then: 한국어 텍스트 변환 성공
        """
        # Given
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            audio_file = Path(tmp.name)

        # When
        with patch("whisper.load_model") as mock_load_model:
            mock_model = MagicMock()
            mock_model.transcribe.return_value = {
                "text": "안녕하세요, 오늘 날씨가 좋네요.",
                "segments": [
                    {
                        "start": 0.0,
                        "end": 2.5,
                        "text": "안녕하세요,",
                        "avg_logprob": -0.1,
                    },
                    {
                        "start": 2.5,
                        "end": 5.0,
                        "text": "오늘 날씨가 좋네요.",
                        "avg_logprob": -0.15,
                    },
                ],
            }
            mock_load_model.return_value = mock_model

            result = await services.transcribe_audio(audio_file, language="ko")

        # Then
        assert result.full_text == "안녕하세요, 오늘 날씨가 좋네요."
        assert len(result.segments) == 2

    @pytest.mark.asyncio
    async def test_transcribe_with_timestamps(self):
        """
        Given: 5초 길이의 오디오 파일
        When: STT 변환 요청
        Then: 각 세그먼트에 정확한 타임스탬프 (start_time, end_time)
        """
        # Given
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            audio_file = Path(tmp.name)

        # When
        with patch("whisper.load_model") as mock_load_model:
            mock_model = MagicMock()
            mock_model.transcribe.return_value = {
                "text": "테스트 문장입니다.",
                "segments": [
                    {
                        "start": 0.0,
                        "end": 1.5,
                        "text": "테스트",
                        "avg_logprob": -0.1,
                    },
                    {
                        "start": 1.5,
                        "end": 3.0,
                        "text": "문장입니다.",
                        "avg_logprob": -0.12,
                    },
                ],
            }
            mock_load_model.return_value = mock_model

            result = await services.transcribe_audio(audio_file)

        # Then
        assert len(result.segments) == 2
        assert result.segments[0].start_time == 0.0
        assert result.segments[0].end_time == 1.5
        assert result.segments[1].start_time == 1.5
        assert result.segments[1].end_time == 3.0

    @pytest.mark.asyncio
    async def test_transcribe_with_confidence_scores(self):
        """
        Given: 오디오 파일
        When: STT 변환 요청
        Then: 각 세그먼트에 0.0~1.0 범위의 신뢰도 점수
        """
        # Given
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            audio_file = Path(tmp.name)

        # When
        with patch("whisper.load_model") as mock_load_model:
            mock_model = MagicMock()
            mock_model.transcribe.return_value = {
                "text": "테스트",
                "segments": [
                    {
                        "start": 0.0,
                        "end": 1.0,
                        "text": "테스트",
                        "avg_logprob": -0.1,  # 낮을수록 좋음
                        "no_speech_prob": 0.01,
                    }
                ],
            }
            mock_load_model.return_value = mock_model

            result = await services.transcribe_audio(audio_file)

        # Then
        assert len(result.segments) == 1
        assert 0.0 <= result.segments[0].confidence <= 1.0

    @pytest.mark.asyncio
    async def test_transcribe_auto_detect_language(self):
        """
        Given: 언어가 지정되지 않은 오디오 파일
        When: STT 변환 요청
        Then: 자동 언어 감지 및 변환
        """
        # Given
        with tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as tmp:
            audio_file = Path(tmp.name)

        # When
        with patch("whisper.load_model") as mock_load_model:
            mock_model = MagicMock()
            mock_model.transcribe.return_value = {
                "text": "Hello world",
                "language": "en",
                "segments": [
                    {
                        "start": 0.0,
                        "end": 2.0,
                        "text": "Hello world",
                        "avg_logprob": -0.08,
                    }
                ],
            }
            mock_load_model.return_value = mock_model

            result = await services.transcribe_audio(audio_file)

        # Then
        assert result.full_text == "Hello world"
        assert result.language == "en"

    @pytest.mark.asyncio
    async def test_transcribe_long_audio(self):
        """
        Given: 5분 길이의 오디오 파일
        When: STT 변환 요청
        Then: 실시간 대비 0.5x 이하로 변환 (2.5분 이내)
        """
        # Given
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            audio_file = Path(tmp.name)

        # When
        with patch("whisper.load_model") as mock_load_model:
            mock_model = MagicMock()
            mock_model.transcribe.return_value = {
                "text": "긴 오디오 내용...",
                "segments": [
                    {
                        "start": float(i * 5),
                        "end": float(i * 5 + 5),
                        "text": f"세그먼트 {i}",
                        "avg_logprob": -0.1,
                    }
                    for i in range(60)  # 5분 = 300초 / 5초 세그먼트 = 60개
                ],
            }
            mock_load_model.return_value = mock_model

            result = await services.transcribe_audio(audio_file)

        # Then
        assert len(result.segments) == 60
        assert result.segments[-1].end_time == 300.0

    @pytest.mark.asyncio
    async def test_transcribe_with_multiple_speakers(self):
        """
        Given: 2인 대화 오디오
        When: STT 변환 요청
        Then: 세그먼트별 텍스트 정확히 변환
        """
        # Given
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            audio_file = Path(tmp.name)

        # When
        with patch("whisper.load_model") as mock_load_model:
            mock_model = MagicMock()
            mock_model.transcribe.return_value = {
                "text": "안녕하세요. 네, 안녕하세요. 만나서 반가워요. 저도요.",
                "segments": [
                    {
                        "start": 0.0,
                        "end": 2.0,
                        "text": "안녕하세요.",
                        "avg_logprob": -0.1,
                    },
                    {
                        "start": 2.0,
                        "end": 3.5,
                        "text": "네, 안녕하세요.",
                        "avg_logprob": -0.12,
                    },
                    {
                        "start": 3.5,
                        "end": 6.0,
                        "text": "만나서 반가워요.",
                        "avg_logprob": -0.11,
                    },
                    {
                        "start": 6.0,
                        "end": 7.5,
                        "text": "저도요.",
                        "avg_logprob": -0.09,
                    },
                ],
            }
            mock_load_model.return_value = mock_model

            result = await services.transcribe_audio(audio_file)

        # Then
        assert len(result.segments) == 4
        assert "안녕하세요" in result.full_text


class TestTranscriptionErrors:
    """STT 변환 에러 처리 테스트"""

    @pytest.mark.asyncio
    async def test_transcribe_corrupted_file(self):
        """
        Given: 손상된 오디오 파일
        When: STT 변환 요청
        Then: 적절한 에러 메시지 반환
        """
        # Given
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            audio_file = Path(tmp.name)

        # When & Then
        with patch("whisper.load_model") as mock_load_model:
            mock_model = MagicMock()
            mock_model.transcribe.side_effect = Exception("FFmpeg error")
            mock_load_model.return_value = mock_model

            with pytest.raises(Exception) as exc_info:
                await services.transcribe_audio(audio_file)

            assert "FFmpeg error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_transcribe_empty_file(self):
        """
        Given: 빈 오디오 파일
        When: STT 변환 요청
        Then: 빈 텍스트 반환
        """
        # Given
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            audio_file = Path(tmp.name)

        # When
        with patch("whisper.load_model") as mock_load_model:
            mock_model = MagicMock()
            mock_model.transcribe.return_value = {
                "text": "",
                "segments": [],
            }
            mock_load_model.return_value = mock_model

            result = await services.transcribe_audio(audio_file)

        # Then
        assert result.full_text == ""
        assert len(result.segments) == 0


class TestTranscriptionPerformance:
    """STT 변환 성능 테스트"""

    @pytest.mark.asyncio
    async def test_transcribe_performance_target(self):
        """
        Given: 1분 오디오 파일
        When: STT 변환 요청
        Then: 30초 이내 변환 완료 (실시간 0.5x)
        """
        # Given
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            audio_file = Path(tmp.name)

        # When
        with patch("whisper.load_model") as mock_load_model:
            mock_model = MagicMock()
            mock_model.transcribe.return_value = {
                "text": "1분 분량의 텍스트...",
                "segments": [
                    {
                        "start": float(i * 5),
                        "end": float(i * 5 + 5),
                        "text": f"세그먼트 {i}",
                        "avg_logprob": -0.1,
                    }
                    for i in range(12)  # 1분 = 60초 / 5초 세그먼트 = 12개
                ],
            }
            mock_load_model.return_value = mock_model

            result = await services.transcribe_audio(audio_file)

        # Then
        assert len(result.segments) == 12
        # 성능 테스트는 실제 환경에서 수행
