"""
FastAPI 메인 애플리케이션 모듈

Voice Man 시스템의 진입점으로, 모든 API 엔드포인트를 관리합니다.
"""

from typing import Annotated, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from pydantic import BaseModel

from voice_man.schemas import AudioUploadResponse
from voice_man.services import (
    compute_sha256_hash,
    generate_file_id,
    is_supported_audio_format,
)

app = FastAPI(
    title="Voice Man API",
    description="음성 녹취 증거 분석 시스템",
    version="0.1.0",
)


class HealthResponse(BaseModel):
    """Health check 응답 모델"""

    status: str


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    헬스체크 엔드포인트

    애플리케이션이 정상적으로 실행 중인지 확인합니다.

    Returns:
        HealthResponse: 상태 정보를 포함한 응답
    """
    return HealthResponse(status="healthy")


@app.post("/api/v1/audio/upload", response_model=AudioUploadResponse)
async def upload_audio_file(
    file: Annotated[UploadFile, File(description="오디오 파일")],
) -> AudioUploadResponse:
    """
    오디오 파일 업로드 엔드포인트

    Args:
        file: 업로드할 오디오 파일

    Returns:
        AudioUploadResponse: 파일 ID와 메타데이터를 포함한 응답

    Raises:
        HTTPException: 지원하지 않는 파일 형식인 경우
    """
    # 파일 형식 검증
    if not is_supported_audio_format(file.content_type):
        raise HTTPException(
            status_code=400,
            detail="지원하지 않는 파일 형식입니다",
        )

    # 파일 내용 읽기
    content = await file.read()

    # 파일 ID 생성
    file_id = generate_file_id()

    # SHA-256 해시 계산
    sha256_hash = compute_sha256_hash(content)

    # 응답 생성
    return AudioUploadResponse(
        file_id=file_id,
        filename=file.filename,
        content_type=file.content_type,
        file_size=len(content),
        sha256_hash=sha256_hash,
    )


# Response Models for Psychology Analysis Endpoints


class CrimeAnalysisResponse(BaseModel):
    """범죄 분석 결과 응답 모델"""

    total_crimes: int = 0
    tags: list = []
    analysis_timestamp: str = ""


class EmotionProfileResponse(BaseModel):
    """감정 프로필 응답 모델"""

    speaker_id: str
    dominant_emotion: str
    average_intensity: float


class EmotionAnalysisResponse(BaseModel):
    """감정 분석 결과 응답 모델"""

    profiles: list[EmotionProfileResponse] = []


class GaslightingPatternResponse(BaseModel):
    """가스라이팅 패턴 응답 모델"""

    type: str
    text: str
    confidence: float
    speaker: str
    timestamp: str


class GaslightingAnalysisResponse(BaseModel):
    """가스라이팅 분석 결과 응답 모델"""

    total_patterns: int = 0
    patterns: list[GaslightingPatternResponse] = []


class ContextAnalysisResponse(BaseModel):
    """맥락 분석 결과 응답 모델"""

    crime_intent_score: float = 0.0
    contextual_factors: list[str] = []
    risk_assessment: str = "low"
    explanation: str = ""


class PsychologyAnalysisResponse(BaseModel):
    """심리 분석 통합 응답 모델"""

    gaslighting_analysis: Optional[GaslightingAnalysisResponse] = None
    emotion_analysis: Optional[EmotionAnalysisResponse] = None
    context_analysis: Optional[ContextAnalysisResponse] = None


class AnalysisStatusResponse(BaseModel):
    """분석 상태 응답 모델"""

    status: str  # pending, processing, completed, failed
    progress: float = 0.0  # 0.0 to 1.0


class AnalysisExecutionResponse(BaseModel):
    """분석 실행 응답 모델"""

    status: str
    message: str


# Psychology Analysis Endpoints


@app.get("/api/v1/audio/{audio_id}/analysis/crime", response_model=CrimeAnalysisResponse)
async def get_crime_analysis(audio_id: str) -> CrimeAnalysisResponse:
    """
    범죄 태깅 분석 결과 조회 엔드포인트

    Args:
        audio_id: 오디오 파일 ID

    Returns:
        CrimeAnalysisResponse: 범죄 태깅 분석 결과

    Raises:
        HTTPException: 오디오 파일을 찾을 수 없는 경우
    """
    # TODO: Implement actual database lookup
    # For now, return empty results
    return CrimeAnalysisResponse(total_crimes=0, tags=[], analysis_timestamp="")


@app.get("/api/v1/audio/{audio_id}/analysis/psychology", response_model=PsychologyAnalysisResponse)
async def get_psychology_analysis(audio_id: str) -> PsychologyAnalysisResponse:
    """
    심리 분석 결과 조회 엔드포인트

    가스라이팅, 감정, 맥락 분석 결과를 통합하여 반환합니다.

    Args:
        audio_id: 오디오 파일 ID

    Returns:
        PsychologyAnalysisResponse: 심리 분석 결과

    Raises:
        HTTPException: 오디오 파일을 찾을 수 없는 경우
    """
    # TODO: Implement actual database lookup
    # For now, return empty results
    return PsychologyAnalysisResponse(
        gaslighting_analysis=None, emotion_analysis=None, context_analysis=None
    )


@app.post(
    "/api/v1/audio/{audio_id}/analyze",
    response_model=AnalysisExecutionResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def start_analysis(audio_id: str) -> AnalysisExecutionResponse:
    """
    전체 분석 실행 엔드포인트

    범죄 태깅, 가스라이팅, 감정, 맥락 분석을 순차적으로 실행합니다.

    Args:
        audio_id: 오디오 파일 ID

    Returns:
        AnalysisExecutionResponse: 분석 실행 상태

    Raises:
        HTTPException: 오디오 파일을 찾을 수 없는 경우
    """
    # TODO: Implement actual analysis execution
    # For now, return pending status
    return AnalysisExecutionResponse(status="pending", message="Analysis queued")


@app.get("/api/v1/audio/{audio_id}/status", response_model=AnalysisStatusResponse)
async def get_analysis_status(audio_id: str) -> AnalysisStatusResponse:
    """
    분석 상태 조회 엔드포인트

    Args:
        audio_id: 오디오 파일 ID

    Returns:
        AnalysisStatusResponse: 분석 상태 정보

    Raises:
        HTTPException: 오디오 파일을 찾을 수 없는 경우
    """
    # TODO: Implement actual status lookup
    # For now, return pending status
    return AnalysisStatusResponse(status="pending", progress=0.0)
