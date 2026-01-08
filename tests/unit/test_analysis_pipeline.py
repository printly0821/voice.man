"""
단일 파일 분석 파이프라인 테스트

TASK-004: STT, 범죄 태깅, 가스라이팅 감지, 감정 분석 통합 파이프라인 테스트
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from voice_man.services.analysis_pipeline_service import (
    SingleFileAnalysisPipeline,
    AnalysisResult,
)


class TestAnalysisResult:
    """AnalysisResult 데이터 클래스 테스트"""

    def test_initialization_success(self):
        """성공 상태로 초기화"""
        result = AnalysisResult(
            file_path="/test/audio.m4a",
            status="success",
            transcription={"text": "테스트"},
        )

        assert result.file_path == "/test/audio.m4a"
        assert result.status == "success"
        assert result.transcription == {"text": "테스트"}
        assert result.crime_tags == []
        assert result.gaslighting_patterns == []
        assert result.emotions == []
        assert result.error is None

    def test_initialization_with_error(self):
        """에러 상태로 초기화"""
        result = AnalysisResult(
            file_path="/test/audio.m4a",
            status="failed",
            error="Transcription failed",
        )

        assert result.status == "failed"
        assert result.error == "Transcription failed"
        assert result.transcription is None

    def test_to_dict(self):
        """딕셔너리 변환 테스트"""
        result = AnalysisResult(
            file_path="/test/audio.m4a",
            status="success",
            transcription={"text": "테스트"},
            crime_tags=[],
            gaslighting_patterns=[],
            emotions=[],
        )

        result_dict = result.to_dict()

        assert result_dict["file_path"] == "/test/audio.m4a"
        assert result_dict["status"] == "success"
        assert result_dict["transcription"] == {"text": "테스트"}
        assert "timestamp" in result_dict


class TestSingleFileAnalysisPipeline:
    """단일 파일 분석 파이프라인 테스트"""

    @pytest.fixture
    def pipeline(self):
        """파이프라인 인스턴스 fixture"""
        return SingleFileAnalysisPipeline()

    @pytest.fixture
    def mock_audio_file(self, tmp_path):
        """테스트용 오디오 파일 fixture"""
        audio_file = tmp_path / "test_audio.m4a"
        audio_file.write_bytes(b"fake audio data")
        return audio_file

    def test_initialization(self, pipeline):
        """초기화 테스트"""
        assert pipeline.crime_service is not None
        assert pipeline.gaslighting_service is not None
        assert pipeline.emotion_service is not None
        assert pipeline.whisper_model is None
        assert pipeline.progress_tracker is not None

    def test_initialization_with_progress_tracker(self):
        """진행률 추적기와 함께 초기화"""
        from voice_man.services.progress_service import ProgressTracker, ProgressConfig

        tracker = ProgressTracker(ProgressConfig())
        pipeline = SingleFileAnalysisPipeline(progress_tracker=tracker)

        assert pipeline.progress_tracker == tracker

    def test_load_whisper_model(self, pipeline):
        """Whisper 모델 로딩 테스트"""
        with patch("whisper.load_model") as mock_load_model:
            mock_model = Mock()
            mock_load_model.return_value = mock_model

            pipeline._load_whisper_model()

            assert pipeline.whisper_model == mock_model
            mock_load_model.assert_called_once_with("base", device="cpu")

    async def test_transcribe_audio_success(self, pipeline, mock_audio_file):
        """STT 변환 성공 테스트"""
        # Mock 설정
        with patch("whisper.load_model") as mock_load_model:
            mock_model = Mock()
            mock_model.transcribe.return_value = {
                "text": "안녕하세요 테스트입니다",
                "segments": [{"start": 0.0, "end": 2.0, "text": "안녕하세요 테스트입니다"}],
                "language": "ko",
            }
            mock_load_model.return_value = mock_model

            # 테스트
            result = await pipeline.transcribe_audio(mock_audio_file)

            assert result["text"] == "안녕하세요 테스트입니다"
            assert result["language"] == "ko"
            assert len(result["segments"]) == 1

    async def test_transcribe_audio_failure(self, pipeline, mock_audio_file):
        """STT 변환 실패 테스트"""
        # Whisper 모델이 이미 로드된 상태로 설정
        pipeline.whisper_model = Mock()
        pipeline.whisper_model.transcribe.side_effect = Exception("Transcription failed")

        # transcribe_audio는 이미 모델이 로드되어 있으면 다시 로드하지 않음
        with pytest.raises(Exception, match="Transcription failed"):
            await pipeline.transcribe_audio(mock_audio_file)

    def test_analyze_crime_tags(self, pipeline):
        """범죄 태그 분석 테스트"""
        text = "죽여버린다 가만 안 둔다"

        tags = pipeline.analyze_crime_tags(text)

        assert len(tags) > 0
        # 협박 태그가 있어야 함
        if tags:
            tag = tags[0]
            # type이 Enum인지 문자열인지 확인
            if hasattr(tag, "type"):
                if hasattr(tag.type, "value"):
                    assert tag.type.value == "협박"
                else:
                    assert tag.type == "협박" or "협박" in str(tag.type)

    def test_analyze_gaslighting(self, pipeline):
        """가스라이팅 패턴 분석 테스트"""
        text = "그런 적 없어 네가 기억을 잘못한 거야"

        patterns = pipeline.analyze_gaslighting(text)

        assert len(patterns) > 0
        # 부정 패턴이 있어야 함
        if patterns and hasattr(patterns[0], "type"):
            pattern = patterns[0]
            if hasattr(pattern.type, "value"):
                assert pattern.type.value == "부정"
            else:
                assert pattern.type == "부정" or "부정" in str(pattern.type)

    def test_analyze_emotions(self, pipeline):
        """감정 분석 테스트"""
        text = "정말 화가 나고 속상해"

        emotion = pipeline.analyze_emotions(text, "Speaker_1")

        assert emotion is not None
        assert hasattr(emotion, "primary_emotion")

    @patch.object(SingleFileAnalysisPipeline, "transcribe_audio", new_callable=AsyncMock)
    async def test_analyze_single_file_success(self, mock_transcribe, pipeline, mock_audio_file):
        """단일 파일 분석 성공 테스트"""
        # Mock STT 결과
        mock_transcribe.return_value = {
            "text": "안녕하세요 정상적인 대화입니다",
            "segments": [],
            "language": "ko",
        }

        # 테스트
        result = await pipeline.analyze_single_file(mock_audio_file)

        assert result.status == "success"
        assert result.transcription is not None
        assert result.error is None
        assert result.crime_tags is not None
        assert result.gaslighting_patterns is not None
        assert result.emotions is not None

    @patch.object(SingleFileAnalysisPipeline, "transcribe_audio", new_callable=AsyncMock)
    async def test_analyze_single_file_empty_transcription(
        self, mock_transcribe, pipeline, mock_audio_file
    ):
        """빈 STT 결과 테스트"""
        # Mock 빈 결과
        mock_transcribe.return_value = {"text": "", "segments": [], "language": "ko"}

        # 테스트
        result = await pipeline.analyze_single_file(mock_audio_file)

        assert result.status == "partial"
        assert result.error == "Empty transcription"

    @patch.object(SingleFileAnalysisPipeline, "transcribe_audio", new_callable=AsyncMock)
    async def test_analyze_single_file_transcription_error(
        self, mock_transcribe, pipeline, mock_audio_file
    ):
        """STT 변환 에러 테스트"""
        # Mock 에러
        mock_transcribe.side_effect = Exception("Transcription failed")

        # 테스트
        result = await pipeline.analyze_single_file(mock_audio_file)

        assert result.status == "failed"
        assert result.error is not None
        assert "Transcription failed" in result.error or "failed" in result.error

    @patch.object(SingleFileAnalysisPipeline, "analyze_single_file", new_callable=AsyncMock)
    async def test_analyze_multiple_files(self, mock_analyze_single, pipeline, tmp_path):
        """다중 파일 분석 테스트"""
        # Mock 파일 생성
        files = [tmp_path / f"test_{i}.m4a" for i in range(3)]
        for f in files:
            f.write_bytes(b"fake audio")

        # Mock 분석 결과
        mock_analyze_single.return_value = AnalysisResult(
            file_path="test",
            status="success",
            transcription={"text": "test"},
        )

        # 테스트
        results = await pipeline.analyze_multiple_files(files, batch_size=2, max_workers=2)

        assert len(results) == 3
        assert mock_analyze_single.call_count == 3

    def test_get_progress_summary(self, pipeline):
        """진행률 요약 조회 테스트"""
        summary = pipeline.get_progress_summary()

        # ProgressTracker.get_progress_summary()가 반환하는 키 확인
        expected_keys = ["total_files", "completed_files", "current_batch", "eta_formatted"]
        for key in expected_keys:
            assert key in summary


class TestIntegration:
    """통합 테스트 (화자 분리 Mock 사용)"""

    @pytest.fixture
    def pipeline_with_mock(self):
        """Mock된 Whisper 모델로 파이프라인 초기화"""
        pipeline = SingleFileAnalysisPipeline()

        # Whisper 모델 mock
        mock_model = Mock()
        mock_model.transcribe.return_value = {
            "text": "죽여버린다 그런 적 없어 정말 화가 나",
            "segments": [
                {"start": 0.0, "end": 3.0, "text": "죽여버린다 그런 적 없어 정말 화가 나"}
            ],
            "language": "ko",
        }

        pipeline.whisper_model = mock_model
        return pipeline

    @pytest.mark.asyncio
    async def test_full_pipeline_with_crime_and_gaslighting(self, pipeline_with_mock, tmp_path):
        """전체 파이프라인 테스트 (범죄 태그 + 가스라이팅)"""
        # 테스트 파일 생성
        audio_file = tmp_path / "test_full.m4a"
        audio_file.write_bytes(b"fake audio")

        # 분석 실행
        result = await pipeline_with_mock.analyze_single_file(audio_file)

        # 검증
        assert result.status == "success"
        assert result.transcription is not None
        assert result.transcription["text"] == "죽여버린다 그런 적 없어 정말 화가 나"

        # 범죄 태그 확인 (협박)
        assert len(result.crime_tags) > 0

        # 가스라이팅 패턴 확인 (부정)
        assert len(result.gaslighting_patterns) > 0

        # 감정 분석 확인
        assert len(result.emotions) > 0

    @pytest.mark.asyncio
    async def test_full_pipeline_normal_conversation(self, pipeline_with_mock, tmp_path):
        """정상 대화 파이프라인 테스트"""
        # Whisper 모델 mock (정상 대화)
        pipeline_with_mock.whisper_model.transcribe.return_value = {
            "text": "안녕하세요 오늘 날씨가 좋네요 네 맞아요",
            "segments": [],
            "language": "ko",
        }

        # 테스트 파일
        audio_file = tmp_path / "test_normal.m4a"
        audio_file.write_bytes(b"fake audio")

        # 분석 실행
        result = await pipeline_with_mock.analyze_single_file(audio_file)

        # 검증
        assert result.status == "success"
        assert result.transcription is not None

        # 범죄 태그 없어야 함
        assert len(result.crime_tags) == 0

        # 가스라이팅 패턴 없어야 함
        assert len(result.gaslighting_patterns) == 0
