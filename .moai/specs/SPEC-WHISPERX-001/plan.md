# SPEC-WHISPERX-001 구현 계획

## 개요

**목표**: WhisperX 통합 파이프라인으로 STT + 타임스탬프 정렬 + 화자분리 end-to-end GPU 처리 구현

**성능 목표**: 183개 m4a 파일 처리 시간 3분 -> 1.2분 (2.5배 추가 향상)

**전략**: 3단계 점진적 통합 접근
- Step 1: WhisperX 환경 구축 및 기본 파이프라인
- Step 2: 화자 분리 통합 및 기존 서비스 연동
- Step 3: 성능 최적화 및 품질 검증

---

## Step 1: WhisperX 환경 구축 및 기본 파이프라인

### 목표
- WhisperX 라이브러리 설치 및 환경 구성
- 기본 파이프라인 클래스 구현
- GPU에서 전사 + 정렬 통합 실행

### 구현 작업

#### 1.1 의존성 추가
**파일**: `pyproject.toml`

**변경 내용**:
```toml
[project]
dependencies = [
    # 기존 의존성 유지
    ...
    # WhisperX 통합 파이프라인 (신규)
    "whisperx>=3.1.5",
    "pyannote-audio>=3.1.1",
    "transformers>=4.36.0",
    "huggingface-hub>=0.20.0",
]
```

**WHY**: WhisperX 및 관련 라이브러리 설치를 위한 의존성 정의.

---

#### 1.2 WhisperX 설정 모듈
**파일**: `src/voice_man/config/whisperx_config.py` (신규)

**코드**:
```python
"""WhisperX 파이프라인 설정"""
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class WhisperXConfig:
    """WhisperX 파이프라인 설정 클래스"""

    model_size: str = "large-v3"
    language: str = "ko"
    device: str = "cuda"
    compute_type: str = "float16"
    batch_size: int = 16

    # Hugging Face 설정
    hf_token: Optional[str] = None

    # Alignment 설정
    align_model: str = "jonatasgrosman/wav2vec2-large-xlsr-53-korean"

    # Diarization 설정
    diarization_model: str = "pyannote/speaker-diarization-3.1"
    min_speakers: int = 1
    max_speakers: int = 10

    # 청크 설정 (긴 오디오용)
    chunk_length_seconds: int = 600  # 10분
    chunk_overlap_seconds: int = 30

    def __post_init__(self):
        """환경 변수에서 설정 로드"""
        self.hf_token = self.hf_token or os.getenv("HF_TOKEN")
        self.model_size = os.getenv("WHISPERX_MODEL_SIZE", self.model_size)
        self.language = os.getenv("WHISPERX_LANGUAGE", self.language)
        self.device = os.getenv("WHISPERX_DEVICE", self.device)

    def validate(self) -> bool:
        """설정 유효성 검증"""
        if not self.hf_token:
            raise ValueError("HF_TOKEN 환경 변수가 필요합니다")
        return True
```

**WHY**: 중앙화된 설정 관리로 환경별 구성 유연성 확보.

---

#### 1.3 WhisperX 파이프라인 클래스
**파일**: `src/voice_man/models/whisperx_pipeline.py` (신규)

**코드**:
```python
"""
WhisperX 통합 파이프라인

STT + Alignment + Diarization을 GPU에서 end-to-end 처리합니다.

EARS Requirements:
- F1: WhisperX 통합 파이프라인 클래스
- U1: 동일 GPU 컨텍스트에서 모든 단계 실행
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import torch
import whisperx

from voice_man.config.whisperx_config import WhisperXConfig

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """파이프라인 처리 결과"""
    text: str
    segments: List[Dict[str, Any]]
    word_segments: List[Dict[str, Any]]
    speakers: List[Dict[str, Any]]
    language: str
    duration: float
    speaker_stats: Dict[str, Any]


class WhisperXPipeline:
    """
    WhisperX 통합 파이프라인

    Whisper 전사 + WAV2VEC2 정렬 + Pyannote 화자분리를
    단일 GPU 컨텍스트에서 실행합니다.
    """

    def __init__(self, config: Optional[WhisperXConfig] = None):
        """
        파이프라인 초기화

        Args:
            config: WhisperX 설정 (None이면 기본값 사용)
        """
        self.config = config or WhisperXConfig()
        self.config.validate()

        self.device = self._resolve_device()
        self.compute_type = self.config.compute_type if self.device == "cuda" else "int8"

        # 모델 레퍼런스
        self._whisper_model = None
        self._align_model = None
        self._align_metadata = None
        self._diarize_model = None

        self._load_models()

    def _resolve_device(self) -> str:
        """GPU 가용성 확인 및 디바이스 결정"""
        if self.config.device == "cuda" and torch.cuda.is_available():
            logger.info(f"GPU 사용: {torch.cuda.get_device_name(0)}")
            return "cuda"
        logger.warning("GPU 미사용, CPU 폴백")
        return "cpu"

    def _load_models(self) -> None:
        """모든 모델 로드 (E1: HF 토큰 검증 포함)"""
        logger.info("WhisperX 모델 로딩 시작...")

        # 1. Whisper 모델
        logger.info(f"Whisper 모델 로딩: {self.config.model_size}")
        self._whisper_model = whisperx.load_model(
            self.config.model_size,
            self.device,
            compute_type=self.compute_type,
            language=self.config.language,
        )

        # 2. Alignment 모델 (S1: 한국어 최적화)
        logger.info(f"Alignment 모델 로딩: {self.config.align_model}")
        self._align_model, self._align_metadata = whisperx.load_align_model(
            language_code=self.config.language,
            device=self.device,
            model_name=self.config.align_model,
        )

        # 3. Diarization 파이프라인 (E1: HF 토큰 사용)
        logger.info(f"Diarization 모델 로딩: {self.config.diarization_model}")
        self._diarize_model = whisperx.DiarizationPipeline(
            model_name=self.config.diarization_model,
            use_auth_token=self.config.hf_token,
            device=self.device,
        )

        logger.info("모든 모델 로딩 완료")

    def process(
        self,
        audio_path: str,
        num_speakers: Optional[int] = None,
    ) -> PipelineResult:
        """
        전체 파이프라인 실행: 전사 -> 정렬 -> 화자분리

        Args:
            audio_path: 오디오 파일 경로
            num_speakers: 화자 수 (None이면 자동 감지)

        Returns:
            PipelineResult: 통합 처리 결과
        """
        logger.info(f"파이프라인 처리 시작: {audio_path}")

        # 오디오 로드
        audio = whisperx.load_audio(audio_path)

        # 1. Whisper 전사
        logger.info("Step 1/3: 전사 수행...")
        result = self._whisper_model.transcribe(
            audio,
            batch_size=self.config.batch_size,
            language=self.config.language,
        )

        # 2. WAV2VEC2 정렬 (U2: word-level 타임스탬프)
        logger.info("Step 2/3: 타임스탬프 정렬...")
        result = whisperx.align(
            result["segments"],
            self._align_model,
            self._align_metadata,
            audio,
            self.device,
            return_char_alignments=False,
        )

        # 3. Pyannote 화자분리 (F3: GPU 병렬 화자분리)
        logger.info("Step 3/3: 화자 분리...")
        diarize_segments = self._diarize_model(
            audio,
            min_speakers=self.config.min_speakers,
            max_speakers=num_speakers or self.config.max_speakers,
        )

        # 화자 할당
        result = whisperx.assign_word_speakers(diarize_segments, result)

        # 결과 구성
        return self._build_result(result, audio_path)

    def _build_result(self, result: Dict, audio_path: str) -> PipelineResult:
        """결과 객체 구성"""
        segments = result.get("segments", [])
        word_segments = result.get("word_segments", [])

        # 전체 텍스트
        text = " ".join(seg.get("text", "") for seg in segments)

        # 화자 추출
        speakers = self._extract_speakers(segments)

        # 화자 통계 (F4)
        speaker_stats = self._compute_speaker_stats(segments, speakers)

        return PipelineResult(
            text=text,
            segments=segments,
            word_segments=word_segments,
            speakers=speakers,
            language=self.config.language,
            duration=segments[-1]["end"] if segments else 0.0,
            speaker_stats=speaker_stats,
        )

    def _extract_speakers(self, segments: List[Dict]) -> List[Dict]:
        """세그먼트에서 화자 정보 추출"""
        speakers = {}
        for seg in segments:
            speaker = seg.get("speaker", "UNKNOWN")
            if speaker not in speakers:
                speakers[speaker] = {
                    "speaker_id": speaker,
                    "segments": [],
                    "total_duration": 0.0,
                }
            speakers[speaker]["segments"].append({
                "start": seg["start"],
                "end": seg["end"],
            })
            speakers[speaker]["total_duration"] += seg["end"] - seg["start"]

        return list(speakers.values())

    def _compute_speaker_stats(
        self,
        segments: List[Dict],
        speakers: List[Dict]
    ) -> Dict[str, Any]:
        """화자별 발화 통계 계산 (F4)"""
        total_duration = sum(s["total_duration"] for s in speakers)

        stats = {
            "total_speakers": len(speakers),
            "total_duration": total_duration,
            "speaker_details": [],
        }

        for speaker in speakers:
            stats["speaker_details"].append({
                "speaker_id": speaker["speaker_id"],
                "duration": speaker["total_duration"],
                "percentage": (speaker["total_duration"] / total_duration * 100)
                              if total_duration > 0 else 0,
                "turn_count": len(speaker["segments"]),
            })

        return stats

    def unload(self) -> None:
        """모델 언로드 및 메모리 해제"""
        del self._whisper_model
        del self._align_model
        del self._diarize_model

        if self.device == "cuda":
            torch.cuda.empty_cache()

        logger.info("모델 언로드 완료")
```

**WHY**: 단일 클래스로 전체 파이프라인을 캡슐화하여 사용성 및 유지보수성 향상.

---

### Step 1 검증 기준
- WhisperX 라이브러리 설치 성공
- 모든 모델 (Whisper, WAV2VEC2, Pyannote) GPU 로딩 성공
- 단일 파일 전사 + 정렬 + 화자분리 성공
- Word-level 타임스탬프 출력 확인

### Step 1 우선순위
- **P0 (최우선)**: 의존성 설치, 설정 모듈
- **P1 (우선)**: 파이프라인 클래스 기본 구현
- **P2 (일반)**: 오류 처리, 로깅 강화

---

## Step 2: 화자 분리 통합 및 기존 서비스 연동

### 목표
- 기존 `diarization_service.py` 인터페이스 호환
- 오디오 포맷 변환 서비스 구현
- 서비스 레이어 통합

### 구현 작업

#### 2.1 오디오 변환 서비스
**파일**: `src/voice_man/services/audio_converter_service.py` (신규)

**코드**:
```python
"""
오디오 포맷 변환 서비스

WhisperX 최적 포맷 (16kHz mono WAV)으로 변환합니다.

EARS Requirements:
- E2: m4a/mp3/wav 파일 자동 변환
"""

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional
import os

logger = logging.getLogger(__name__)


class AudioConverterService:
    """오디오 포맷 변환 서비스"""

    SUPPORTED_FORMATS = {".m4a", ".mp3", ".wav", ".flac", ".ogg", ".aac"}
    TARGET_SAMPLE_RATE = 16000
    TARGET_CHANNELS = 1  # mono

    def __init__(self, temp_dir: Optional[str] = None):
        """
        서비스 초기화

        Args:
            temp_dir: 임시 파일 저장 경로 (None이면 시스템 기본)
        """
        self.temp_dir = temp_dir
        self._temp_files = []

    def convert_to_wav(self, audio_path: str) -> str:
        """
        오디오를 WhisperX 최적 포맷으로 변환

        Args:
            audio_path: 원본 오디오 경로

        Returns:
            변환된 WAV 파일 경로
        """
        path = Path(audio_path)

        if not path.exists():
            raise FileNotFoundError(f"파일 없음: {audio_path}")

        if path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(f"지원하지 않는 포맷: {path.suffix}")

        # 이미 WAV이고 16kHz mono인 경우 변환 생략
        if path.suffix.lower() == ".wav" and self._is_optimal_format(audio_path):
            return audio_path

        # 임시 파일 생성
        temp_file = tempfile.NamedTemporaryFile(
            suffix=".wav",
            dir=self.temp_dir,
            delete=False,
        )
        temp_path = temp_file.name
        temp_file.close()

        self._temp_files.append(temp_path)

        # ffmpeg 변환
        try:
            cmd = [
                "ffmpeg", "-y", "-i", audio_path,
                "-ar", str(self.TARGET_SAMPLE_RATE),
                "-ac", str(self.TARGET_CHANNELS),
                "-acodec", "pcm_s16le",
                temp_path
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"변환 완료: {audio_path} -> {temp_path}")
            return temp_path
        except subprocess.CalledProcessError as e:
            logger.error(f"변환 실패: {e.stderr.decode()}")
            raise

    def _is_optimal_format(self, audio_path: str) -> bool:
        """파일이 이미 최적 포맷인지 확인"""
        try:
            import soundfile as sf
            info = sf.info(audio_path)
            return (
                info.samplerate == self.TARGET_SAMPLE_RATE and
                info.channels == self.TARGET_CHANNELS
            )
        except Exception:
            return False

    def cleanup(self) -> None:
        """임시 파일 정리 (N3: 임시 파일 삭제)"""
        for temp_path in self._temp_files:
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    logger.debug(f"임시 파일 삭제: {temp_path}")
            except Exception as e:
                logger.warning(f"임시 파일 삭제 실패: {e}")
        self._temp_files.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
```

**WHY**: 다양한 오디오 포맷을 WhisperX 최적 포맷으로 변환하여 처리 품질 보장.

---

#### 2.2 기존 DiarizationService 통합
**파일**: `src/voice_man/services/diarization_service.py` (수정)

**수정 내용**:
```python
"""
화자 분리 (Speaker Diarization) 서비스

WhisperX/pyannote-audio를 사용하여 오디오 파일에서 화자를 분리합니다.

EARS Requirements:
- F3: Pyannote GPU 병렬 화자 분리
- F5: 기존 서비스 인터페이스 호환
"""

from pathlib import Path
from typing import List, Optional

from voice_man.models.diarization import (
    DiarizationResult,
    Speaker,
    SpeakerTurn,
    SpeakerStats,
)
from voice_man.models.database import TranscriptSegment
from voice_man.models.whisperx_pipeline import WhisperXPipeline
from voice_man.config.whisperx_config import WhisperXConfig


class DiarizationService:
    """
    화자 분리 서비스

    WhisperX/pyannote-audio 모델을 사용하여 오디오 파일의 화자를 분리합니다.

    기존 인터페이스와 완전히 호환됩니다 (F5).
    """

    def __init__(self, config: Optional[WhisperXConfig] = None):
        """화자 분리 서비스 초기화"""
        self._config = config
        self._pipeline: Optional[WhisperXPipeline] = None
        self._model_loaded = False

    def _ensure_model_loaded(self) -> None:
        """모델 지연 로딩"""
        if not self._model_loaded:
            self._pipeline = WhisperXPipeline(self._config)
            self._model_loaded = True

    async def diarize_speakers(
        self, audio_path: str, num_speakers: Optional[int] = None
    ) -> DiarizationResult:
        """
        오디오 파일의 화자 분리 수행

        기존 인터페이스 유지 (F5)

        Args:
            audio_path: 오디오 파일 경로
            num_speakers: 화자 수 (None이면 자동 감지)

        Returns:
            DiarizationResult: 화자 분리 결과
        """
        # 입력 검증 (기존 로직 유지)
        if not audio_path or not Path(audio_path).exists():
            raise ValueError("오디오 파일이 비어있습니다")

        audio_file = Path(audio_path)
        if not audio_file.exists() or audio_file.stat().st_size < 100:
            raise ValueError("오디오 파일을 처리할 수 없습니다")

        # WhisperX 파이프라인 사용
        self._ensure_model_loaded()
        result = self._pipeline.process(audio_path, num_speakers=num_speakers)

        # 기존 DiarizationResult 형식으로 변환
        speakers = [
            Speaker(
                speaker_id=s["speaker_id"],
                start_time=s["segments"][0]["start"] if s["segments"] else 0.0,
                end_time=s["segments"][-1]["end"] if s["segments"] else 0.0,
                duration=s["total_duration"],
                confidence=0.95,  # WhisperX는 신뢰도 미제공, 기본값 사용
            )
            for s in result.speakers
        ]

        return DiarizationResult(
            speakers=speakers,
            total_duration=result.duration,
            num_speakers=len(speakers),
        )

    # ... (기존 메서드들 유지: merge_with_transcript, generate_speaker_stats, detect_speaker_turns)
```

**WHY**: 기존 코드베이스와의 호환성을 유지하면서 내부 구현을 WhisperX로 교체.

---

#### 2.3 WhisperX 서비스 레이어
**파일**: `src/voice_man/services/whisperx_service.py` (신규)

**코드**:
```python
"""
WhisperX 서비스 레이어

파이프라인과 애플리케이션 코드 사이의 추상화 계층을 제공합니다.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

from voice_man.models.whisperx_pipeline import WhisperXPipeline, PipelineResult
from voice_man.services.audio_converter_service import AudioConverterService
from voice_man.config.whisperx_config import WhisperXConfig

logger = logging.getLogger(__name__)


class WhisperXService:
    """
    WhisperX 서비스

    오디오 변환, 파이프라인 실행, 결과 후처리를 통합 관리합니다.
    """

    def __init__(self, config: Optional[WhisperXConfig] = None):
        """서비스 초기화"""
        self.config = config or WhisperXConfig()
        self._pipeline: Optional[WhisperXPipeline] = None
        self._converter = AudioConverterService()

    def initialize(self) -> None:
        """파이프라인 초기화 (지연 로딩)"""
        if self._pipeline is None:
            self.config.validate()
            self._pipeline = WhisperXPipeline(self.config)

    def process_audio(
        self,
        audio_path: str,
        num_speakers: Optional[int] = None,
        include_word_timestamps: bool = True,
    ) -> Dict[str, Any]:
        """
        오디오 파일 처리

        Args:
            audio_path: 오디오 파일 경로
            num_speakers: 화자 수 (자동 감지는 None)
            include_word_timestamps: word-level 타임스탬프 포함 여부

        Returns:
            처리 결과 딕셔너리
        """
        self.initialize()

        # 오디오 변환 (필요시)
        with self._converter as converter:
            converted_path = converter.convert_to_wav(audio_path)

            # 파이프라인 실행
            result = self._pipeline.process(converted_path, num_speakers)

            # 결과 포맷팅
            return self._format_result(result, include_word_timestamps)

    def process_batch(
        self,
        audio_paths: List[str],
        num_speakers: Optional[int] = None,
        progress_callback: Optional[callable] = None,
    ) -> List[Dict[str, Any]]:
        """
        다중 오디오 파일 배치 처리

        Args:
            audio_paths: 오디오 파일 경로 목록
            num_speakers: 화자 수
            progress_callback: 진행률 콜백 함수

        Returns:
            처리 결과 목록
        """
        results = []
        total = len(audio_paths)

        for i, path in enumerate(audio_paths):
            try:
                result = self.process_audio(path, num_speakers)
                result["status"] = "success"
                result["file"] = path
            except Exception as e:
                logger.error(f"처리 실패: {path} - {e}")
                result = {
                    "status": "failed",
                    "file": path,
                    "error": str(e),
                }

            results.append(result)

            if progress_callback:
                progress_callback((i + 1) / total * 100, path)

        return results

    def _format_result(
        self,
        result: PipelineResult,
        include_word_timestamps: bool,
    ) -> Dict[str, Any]:
        """결과 포맷팅"""
        output = {
            "text": result.text,
            "segments": result.segments,
            "speakers": result.speakers,
            "language": result.language,
            "duration": result.duration,
            "speaker_stats": result.speaker_stats,
        }

        if include_word_timestamps:
            output["word_segments"] = result.word_segments

        return output

    def cleanup(self) -> None:
        """리소스 정리"""
        if self._pipeline:
            self._pipeline.unload()
            self._pipeline = None
```

**WHY**: 서비스 레이어를 통해 파이프라인 복잡성을 숨기고 애플리케이션 코드와의 결합도 감소.

---

### Step 2 검증 기준
- 기존 `diarization_service.py` 테스트 통과
- 오디오 포맷 변환 (m4a -> wav) 성공
- 화자 분리 결과 기존 형식과 호환
- 임시 파일 자동 삭제 확인

### Step 2 우선순위
- **P0 (최우선)**: DiarizationService 통합
- **P1 (우선)**: 오디오 변환 서비스
- **P2 (일반)**: WhisperX 서비스 레이어

---

## Step 3: 성능 최적화 및 품질 검증

### 목표
- GPU 활용률 90% 달성
- 183개 파일 1.5분 이내 처리
- 테스트 커버리지 85% 이상

### 구현 작업

#### 3.1 GPU 메모리 최적화
**파일**: `src/voice_man/models/whisperx_pipeline.py` (수정)

**추가 내용**:
```python
def _check_memory_and_adjust(self) -> None:
    """
    GPU 메모리 확인 및 모델 로딩 전략 조정 (S2)
    """
    if self.device != "cuda":
        return

    import pynvml
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)

    usage_ratio = info.used / info.total

    if usage_ratio > 0.7:
        logger.warning(f"GPU 메모리 사용률 높음: {usage_ratio:.1%}")
        self._sequential_loading = True
    else:
        self._sequential_loading = False
```

**WHY**: GPU 메모리 상황에 따라 동적으로 로딩 전략 조정.

---

#### 3.2 긴 오디오 청크 분할 (S3)
**파일**: `src/voice_man/services/audio_chunker_service.py` (신규)

**코드**:
```python
"""
긴 오디오 파일 청크 분할 서비스

30분 초과 오디오를 10분 청크로 분할하여 처리합니다.

EARS Requirements:
- S3: 긴 오디오 파일 청크 분할 처리
"""

import logging
from pathlib import Path
from typing import List, Tuple
import tempfile

import soundfile as sf
import numpy as np

logger = logging.getLogger(__name__)


class AudioChunkerService:
    """긴 오디오 파일 청크 분할 서비스"""

    def __init__(
        self,
        max_duration: int = 1800,  # 30분
        chunk_duration: int = 600,  # 10분
        overlap: int = 30,  # 30초
    ):
        self.max_duration = max_duration
        self.chunk_duration = chunk_duration
        self.overlap = overlap

    def should_chunk(self, audio_path: str) -> bool:
        """청크 분할이 필요한지 확인"""
        info = sf.info(audio_path)
        return info.duration > self.max_duration

    def split_audio(self, audio_path: str) -> List[Tuple[str, float, float]]:
        """
        오디오를 청크로 분할

        Returns:
            List of (chunk_path, start_time, end_time)
        """
        audio, sr = sf.read(audio_path)
        duration = len(audio) / sr

        chunks = []
        start = 0

        while start < duration:
            end = min(start + self.chunk_duration, duration)

            # 샘플 추출
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            chunk_audio = audio[start_sample:end_sample]

            # 임시 파일 저장
            temp_file = tempfile.NamedTemporaryFile(
                suffix=".wav",
                delete=False,
            )
            sf.write(temp_file.name, chunk_audio, sr)

            chunks.append((temp_file.name, start, end))

            # 다음 청크 시작 (오버랩 적용)
            start = end - self.overlap

        return chunks

    def merge_results(
        self,
        chunk_results: List[dict],
        chunk_info: List[Tuple[str, float, float]],
    ) -> dict:
        """청크 결과 병합"""
        merged_segments = []
        merged_word_segments = []

        for result, (_, start_offset, _) in zip(chunk_results, chunk_info):
            # 시간 오프셋 적용
            for seg in result.get("segments", []):
                adjusted_seg = seg.copy()
                adjusted_seg["start"] += start_offset
                adjusted_seg["end"] += start_offset
                merged_segments.append(adjusted_seg)

            for word_seg in result.get("word_segments", []):
                adjusted_word = word_seg.copy()
                adjusted_word["start"] += start_offset
                adjusted_word["end"] += start_offset
                merged_word_segments.append(adjusted_word)

        # 중복 제거 (오버랩 구간)
        merged_segments = self._remove_duplicates(merged_segments)
        merged_word_segments = self._remove_duplicates(merged_word_segments)

        return {
            "segments": merged_segments,
            "word_segments": merged_word_segments,
            "text": " ".join(seg["text"] for seg in merged_segments),
        }

    def _remove_duplicates(self, segments: List[dict]) -> List[dict]:
        """오버랩 구간의 중복 세그먼트 제거"""
        if not segments:
            return []

        # 시작 시간 기준 정렬
        sorted_segs = sorted(segments, key=lambda x: x["start"])

        result = [sorted_segs[0]]
        for seg in sorted_segs[1:]:
            # 이전 세그먼트와 50% 이상 겹치면 스킵
            prev = result[-1]
            overlap = min(prev["end"], seg["end"]) - max(prev["start"], seg["start"])
            seg_duration = seg["end"] - seg["start"]

            if overlap < seg_duration * 0.5:
                result.append(seg)

        return result
```

**WHY**: 장시간 오디오 처리 시 메모리 오류 방지 및 안정성 확보.

---

#### 3.3 성능 리포트 확장
**파일**: `src/voice_man/services/performance_report_service.py` (수정)

**추가 메트릭**:
```python
def generate_whisperx_report(self, results: List[dict]) -> dict:
    """WhisperX 파이프라인 성능 리포트"""
    return {
        "total_files": len(results),
        "successful": sum(1 for r in results if r["status"] == "success"),
        "failed": sum(1 for r in results if r["status"] == "failed"),
        "total_duration_seconds": sum(r.get("duration", 0) for r in results),
        "avg_speakers_per_file": np.mean([
            len(r.get("speakers", []))
            for r in results if r["status"] == "success"
        ]),
        "word_timestamp_coverage": self._calculate_word_coverage(results),
        "gpu_utilization": self._get_gpu_stats(),
    }
```

**WHY**: 파이프라인 성능 및 품질 검증을 위한 상세 메트릭 제공.

---

### Step 3 검증 기준
- 183개 파일 1.5분 이내 처리 완료
- GPU 활용률 85% 이상
- 테스트 커버리지 85% 이상
- TRUST 5 품질 게이트 통과

### Step 3 우선순위
- **P0 (최우선)**: GPU 메모리 최적화
- **P1 (우선)**: 청크 분할 처리
- **P2 (일반)**: 성능 리포트 확장

---

## 테스트 전략

### 단위 테스트
- `test_whisperx_pipeline.py`: 파이프라인 기본 기능
- `test_audio_converter_service.py`: 오디오 변환
- `test_audio_chunker_service.py`: 청크 분할
- `test_diarization_integration.py`: 화자분리 통합

### 통합 테스트
- **소규모**: 5개 파일 (1분 이내)
- **중규모**: 50개 파일 (15분 이내)
- **전체**: 183개 파일 (1.5분 이내)

### 성능 테스트
- GPU 활용률 모니터링
- 메모리 사용량 프로파일링
- 타임스탬프 정확도 검증

---

## 의존성 및 순서

### Step 간 의존성
- **Step 1 -> Step 2**: 기본 파이프라인 후 서비스 통합
- **Step 2 -> Step 3**: 서비스 통합 후 최적화

### 병렬 작업 가능성
- Step 1 구현 중 Step 2 테스트 케이스 작성 가능
- Step 2 구현 중 Step 3 성능 측정 환경 구축 가능

---

## 롤백 계획

### Step 3 실패 시
- Step 2 (기본 통합) 유지
- 성능 목표 미달 시 Phase 2 (3분) 성능으로 유지

### Step 2 실패 시
- Step 1 (기본 파이프라인) 유지
- 기존 diarization_service 모의 구현으로 폴백

### Step 1 실패 시
- SPEC-PARALLEL-001 Phase 2 (faster-whisper) 유지
- 화자분리는 별도 후처리로 진행

---

## 마일스톤

### Milestone 1: 기본 파이프라인 완료 (Step 1)
- **목표**: WhisperX 전사 + 정렬 성공
- **검증**: 단일 파일 처리 테스트 통과
- **다음 단계**: Step 2 시작

### Milestone 2: 서비스 통합 완료 (Step 2)
- **목표**: 기존 인터페이스 호환성 확보
- **검증**: 기존 테스트 케이스 통과
- **다음 단계**: Step 3 시작

### Milestone 3: 최종 목표 달성 (Step 3)
- **목표**: 1.5분 이내 처리, GPU 85% 활용
- **검증**: 전체 테스트 및 품질 게이트 통과
- **다음 단계**: 프로덕션 배포

---

## 다음 단계

1. Step 1 구현 시작 (`/moai:2-run SPEC-WHISPERX-001 --step 1`)
2. Step 1 검증 후 Step 2 진행
3. Step 2 검증 후 Step 3 진행
4. 최종 성능 검증 및 문서화
5. 프로덕션 배포 및 모니터링

---

**문서 끝**
