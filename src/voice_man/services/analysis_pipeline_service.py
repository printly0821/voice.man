"""
단일 파일 분석 파이프라인 서비스

STT 변환, 범죄 태깅, 가스라이팅 감지, 감정 분석을 통합하는 파이프라인 서비스
CPU 환경에 최적화 (화자 분리 제외, 키워드 기반 분석)
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime
import asyncio

from voice_man.services.batch_service import BatchProcessor
from voice_man.services.progress_service import ProgressTracker, ProgressConfig
from voice_man.services.crime_tagging_service import CrimeTaggingService
from voice_man.services.gaslighting_service import GaslightingService
from voice_man.services.emotion_service import EmotionAnalysisService


logger = logging.getLogger(__name__)


class AnalysisResult:
    """단일 파일 분석 결과"""

    def __init__(
        self,
        file_path: str,
        status: str,
        transcription: Optional[Dict] = None,
        crime_tags: Optional[List] = None,
        gaslighting_patterns: Optional[List] = None,
        emotions: Optional[List] = None,
        error: Optional[str] = None,
    ):
        self.file_path = file_path
        self.status = status  # success, failed, partial
        self.transcription = transcription  # STT 결과
        self.crime_tags = crime_tags or []  # 범죄 태그
        self.gaslighting_patterns = gaslighting_patterns or []  # 가스라이팅 패턴
        self.emotions = emotions or []  # 감정 분석
        self.error = error
        self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict:
        """딕셔너리로 변환"""
        return {
            "file_path": self.file_path,
            "status": self.status,
            "transcription": self.transcription,
            "crime_tags": [
                tag.__dict__ if hasattr(tag, "__dict__") else tag for tag in self.crime_tags
            ],
            "gaslighting_patterns": [
                p.__dict__ if hasattr(p, "__dict__") else p for p in self.gaslighting_patterns
            ],
            "emotions": [e.__dict__ if hasattr(e, "__dict__") else e for e in self.emotions],
            "error": self.error,
            "timestamp": self.timestamp,
        }


class SingleFileAnalysisPipeline:
    """
    단일 파일 분석 파이프라인

    처리 단계:
    1. STT 변환 (Whisper CPU 버전)
    2. 범죄 태깅 (키워드 매칭)
    3. 가스라이팅 감지 (키워드 매칭)
    4. 감정 분석 (규칙 기반)
    5. 진행률 업데이트
    """

    def __init__(
        self,
        progress_tracker: Optional[ProgressTracker] = None,
        use_whisper_cpu: bool = True,
    ):
        """
        초기화

        Args:
            progress_tracker: 진행률 추적기 (선택)
            use_whisper_cpu: Whisper CPU 버전 사용 여부
        """
        self.progress_tracker = progress_tracker or ProgressTracker(ProgressConfig())
        self.use_whisper_cpu = use_whisper_cpu

        # 분석 서비스 초기화
        self.crime_service = CrimeTaggingService()
        self.gaslighting_service = GaslightingService()
        self.emotion_service = EmotionAnalysisService()

        # Whisper 모델 (지연 로딩)
        self.whisper_model = None

    def _load_whisper_model(self):
        """Whisper 모델 로딩 (CPU 버전)"""
        if self.whisper_model is not None:
            return

        try:
            import whisper

            logger.info("Loading Whisper model (CPU version)...")
            model_size = "base"  # CPU 환경에서는 base 모델 사용
            self.whisper_model = whisper.load_model(model_size, device="cpu")
            logger.info(f"Whisper model loaded: {model_size}")
        except ImportError:
            logger.error("Whisper not installed. Install with: pip install openai-whisper")
            raise RuntimeError("Whisper not available")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise

    async def transcribe_audio(self, audio_path: Path) -> Dict:
        """
        음성을 텍스트로 변환 (STT)

        Args:
            audio_path: 오디오 파일 경로

        Returns:
            변환 결과 딕셔너리
        """
        if self.whisper_model is None:
            self._load_whisper_model()

        try:
            # Whisper는 동기 라이브러리이므로 스레드 풀에서 실행
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self._transcribe_sync, audio_path)

            return {
                "text": result["text"],
                "segments": [
                    {"start": seg["start"], "end": seg["end"], "text": seg["text"]}
                    for seg in result["segments"]
                ],
                "language": result.get("language", "unknown"),
            }
        except Exception as e:
            logger.error(f"Transcription failed for {audio_path}: {e}")
            raise

    def _transcribe_sync(self, audio_path: Path) -> Dict:
        """동기 STT 처리"""
        result = self.whisper_model.transcribe(
            str(audio_path),
            language="ko",  # 한국어
            task="transcribe",
            word_timestamps=True,
        )
        return result

    def analyze_crime_tags(self, text: str) -> List:
        """
        범죄 태그 분석

        Args:
            text: 분석할 텍스트

        Returns:
            범죄 태그 리스트
        """
        return self.crime_service.detect_crime(text)

    def analyze_gaslighting(self, text: str) -> List:
        """
        가스라이팅 패턴 분석

        Args:
            text: 분석할 텍스트

        Returns:
            가스라이팅 패턴 리스트
        """
        return self.gaslighting_service.detect_patterns(text)

    def analyze_emotions(self, text: str, speaker_id: str = "Unknown") -> Any:
        """
        감정 분석

        Args:
            text: 분석할 텍스트
            speaker_id: 화자 ID

        Returns:
            감정 분석 결과
        """
        return self.emotion_service.analyze_emotion(text, speaker_id)

    async def analyze_single_file(self, file_path: Path) -> AnalysisResult:
        """
        단일 파일 전체 분석

        Args:
            file_path: 오디오 파일 경로

        Returns:
            AnalysisResult 객체
        """
        file_name = file_path.name

        try:
            # 진행률 추적 시작
            if self.progress_tracker:
                self.progress_tracker.start_file(file_path)

            logger.info(f"Starting analysis for {file_name}")

            # STEP 1: STT 변환
            logger.info(f"[STT] Transcribing {file_name}...")
            transcription = await self.transcribe_audio(file_path)
            full_text = transcription["text"]

            if not full_text or not full_text.strip():
                logger.warning(f"Empty transcription for {file_name}")
                return AnalysisResult(
                    file_path=str(file_path),
                    status="partial",
                    error="Empty transcription",
                )

            logger.info(f"[STT] Transcription complete: {len(full_text)} characters")

            # 진행률 업데이트 (30%)
            if self.progress_tracker:
                self.progress_tracker.update_file_progress(file_path, 0.3)

            # STEP 2: 범죄 태깅
            logger.info("[CRIME] Detecting crime tags...")
            crime_tags = self.analyze_crime_tags(full_text)
            logger.info(f"[CRIME] Found {len(crime_tags)} crime tags")

            # 진행률 업데이트 (50%)
            if self.progress_tracker:
                self.progress_tracker.update_file_progress(file_path, 0.5)

            # STEP 3: 가스라이팅 감지
            logger.info("[GASLIGHTING] Detecting patterns...")
            gaslighting_patterns = self.analyze_gaslighting(full_text)
            logger.info(f"[GASLIGHTING] Found {len(gaslighting_patterns)} patterns")

            # 진행률 업데이트 (70%)
            if self.progress_tracker:
                self.progress_tracker.update_file_progress(file_path, 0.7)

            # STEP 4: 감정 분석
            # 화자 분리가 없으므로 전체 텍스트를 하나의 화자로 처리
            logger.info("[EMOTION] Analyzing emotions...")
            emotions = [self.analyze_emotions(full_text, "Speaker_Unknown")]
            logger.info("[EMOTION] Analysis complete")

            # 진행률 업데이트 (100%)
            if self.progress_tracker:
                self.progress_tracker.complete_file(file_path)

            # 분석 결과 생성
            result = AnalysisResult(
                file_path=str(file_path),
                status="success",
                transcription=transcription,
                crime_tags=crime_tags,
                gaslighting_patterns=gaslighting_patterns,
                emotions=emotions,
            )

            logger.info(f"Analysis complete for {file_name}")

            return result

        except Exception as e:
            error_msg = f"Analysis failed for {file_name}: {str(e)}"
            logger.error(error_msg)

            # 진행률 추적 실패 처리
            if self.progress_tracker:
                self.progress_tracker.fail_file(file_path, str(e))

            return AnalysisResult(
                file_path=str(file_path),
                status="failed",
                error=error_msg,
            )

    async def analyze_multiple_files(
        self,
        file_paths: List[Path],
        batch_size: int = 5,
        max_workers: int = 4,
    ) -> List[AnalysisResult]:
        """
        여러 파일 분석 (배치 처리)

        Args:
            file_paths: 파일 경로 리스트
            batch_size: 배치 크기
            max_workers: 최대 worker 수

        Returns:
            AnalysisResult 리스트
        """
        from voice_man.services.batch_service import BatchConfig

        # 배치 프로세서 생성
        batch_config = BatchConfig(
            batch_size=batch_size,
            max_workers=max_workers,
            continue_on_error=True,
        )
        batch_processor = BatchProcessor(batch_config)

        # 분석 함수 정의
        async def process_func(file_path: Path) -> Dict:
            result = await self.analyze_single_file(file_path)
            return result.to_dict()

        # 배치 처리 실행
        batch_results = await batch_processor.process_all(file_paths, process_func)

        # BatchResult를 AnalysisResult로 변환
        analysis_results = []
        for batch_result in batch_results:
            if batch_result.status == "success":
                # 딕셔너리를 다시 AnalysisResult로 변환
                data = batch_result.data
                analysis_results.append(
                    AnalysisResult(
                        file_path=data["file_path"],
                        status=data["status"],
                        transcription=data.get("transcription"),
                        crime_tags=data.get("crime_tags", []),
                        gaslighting_patterns=data.get("gaslighting_patterns", []),
                        emotions=data.get("emotions", []),
                        error=data.get("error"),
                    )
                )
            else:
                analysis_results.append(
                    AnalysisResult(
                        file_path=batch_result.file_path,
                        status="failed",
                        error=batch_result.error,
                    )
                )

        return analysis_results

    def get_progress_summary(self) -> Dict:
        """
        진행률 요약 조회

        Returns:
            진행률 요약 딕셔너리
        """
        if self.progress_tracker:
            return self.progress_tracker.get_progress_summary()
        return {"overall_progress": 0.0, "total_files": 0, "completed_files": 0}
