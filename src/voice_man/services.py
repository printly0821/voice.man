"""
비즈니스 로직 서비스 모듈

파일 처리, 해시 생성, 오디오 전처리 등의 비즈니스 로직을 구현합니다.
"""

import hashlib
import logging
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class AudioFormat(str, Enum):
    """지원하는 오디오 파일 형식"""

    MP3 = "audio/mpeg"
    WAV = "audio/wav"
    M4A = "audio/mp4"
    FLAC = "audio/flac"
    OGG = "audio/ogg"


SUPPORTED_FORMATS = {format.value for format in AudioFormat}


def generate_file_id() -> str:
    """
    고유한 파일 ID를 생성합니다.

    Returns:
        str: UUID 기반 파일 ID
    """
    return str(uuid.uuid4())


def compute_sha256_hash(content: bytes) -> str:
    """
    파일 내용의 SHA-256 해시를 계산합니다.

    Args:
        content: 파일 내용 (bytes)

    Returns:
        str: SHA-256 해시값 (16진수 문자열)
    """
    return hashlib.sha256(content).hexdigest()


def is_supported_audio_format(content_type: str) -> bool:
    """
    지원되는 오디오 형식인지 확인합니다.

    Args:
        content_type: MIME 타입

    Returns:
        bool: 지원 여부
    """
    return content_type in SUPPORTED_FORMATS


@dataclass
class AudioMetadata:
    """오디오 파일 메타데이터"""

    duration_seconds: float
    sample_rate: int
    channels: int
    format: str


@dataclass
class TranscriptSegment:
    """STT 세그먼트 정보"""

    start_time: float
    end_time: float
    text: str
    confidence: float


@dataclass
class TranscriptionResult:
    """STT 변환 결과"""

    full_text: str
    segments: list[TranscriptSegment]
    language: str


async def extract_audio_metadata(file_path: Path) -> AudioMetadata:
    """
    FFmpeg를 사용하여 오디오 파일의 메타데이터를 추출합니다.

    Args:
        file_path: 오디오 파일 경로

    Returns:
        AudioMetadata: 추출된 메타데이터

    Raises:
        ValueError: 파일을 읽을 수 없는 경우
    """
    try:
        # FFmpeg-python 라이브러리를 사용하여 메타데이터 추출
        import ffmpeg

        probe = ffmpeg.probe(str(file_path))
        format_info = probe.get("format", {})
        streams = probe.get("streams", [])

        if not streams:
            raise ValueError("No audio streams found")

        # 첫 번째 오디오 스트림 정보 추출
        audio_stream = streams[0]

        duration = float(format_info.get("duration", 0))
        sample_rate = int(audio_stream.get("sample_rate", 0))
        channels = int(audio_stream.get("channels", 1))

        # 파일 형식 감지
        file_format = file_path.suffix.lstrip(".").lower()

        return AudioMetadata(
            duration_seconds=duration,
            sample_rate=sample_rate,
            channels=channels,
            format=file_format,
        )
    except Exception as e:
        logger.error(f"Failed to extract metadata from {file_path}: {e}")
        raise ValueError(f"Cannot read file: {e}")


async def detect_corrupted_file(file_path: Path) -> bool:
    """
    오디오 파일이 손상되었는지 감지합니다.

    Args:
        file_path: 오디오 파일 경로

    Returns:
        bool: 손상되었으면 True, 정상이면 False
    """
    try:
        metadata = await extract_audio_metadata(file_path)

        # 재생 시간이 0초이거나 매우 짧으면 손상된 것으로 간주
        if metadata.duration_seconds <= 0:
            return True

        # 샘플레이트가 비정상적으로 낮으면 손상된 것으로 간주
        if metadata.sample_rate < 8000:
            return True

        return False
    except Exception:
        # 메타데이터 추출 실패 시 손상된 것으로 간주
        return True


async def transcribe_audio(
    file_path: Path,
    language: str | None = None,
    model_size: str = "large-v3",
) -> TranscriptionResult:
    """
    Whisper 모델을 사용하여 오디오를 텍스트로 변환합니다.

    Args:
        file_path: 오디오 파일 경로
        language: 언어 코드 (예: "ko", "en"). None이면 자동 감지
        model_size: Whisper 모델 크기 (default: "large-v3")

    Returns:
        TranscriptionResult: 변환된 텍스트와 세그먼트 정보

    Raises:
        Exception: 변환 실패 시
    """
    try:
        import whisper

        # Whisper 모델 로드
        model = whisper.load_model(model_size)

        # 변환 옵션 설정
        options = {
            "task": "transcribe",
            "language": language,
            "word_timestamps": True,
        }

        # 오디오 변환
        result = model.transcribe(str(file_path), **options)

        # 세그먼트 변환
        segments = []
        for seg in result.get("segments", []):
            # 신뢰도 점수 계산 (avg_logprob를 0~1 사이로 변환)
            avg_logprob = seg.get("avg_logprob", -0.5)
            # logprob는 보통 -0.1 ~ -1.0 사이, 이를 0.9 ~ 0.0으로 변환
            confidence = max(0.0, min(1.0, 1.0 + avg_logprob))

            segments.append(
                TranscriptSegment(
                    start_time=seg["start"],
                    end_time=seg["end"],
                    text=seg["text"].strip(),
                    confidence=confidence,
                )
            )

        detected_language = result.get("language", "unknown")

        return TranscriptionResult(
            full_text=result["text"].strip(),
            segments=segments,
            language=detected_language,
        )
    except Exception as e:
        logger.error(f"Failed to transcribe audio {file_path}: {e}")
        raise
