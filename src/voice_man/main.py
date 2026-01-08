"""
FastAPI 메인 애플리케이션 모듈

Voice Man 시스템의 진입점으로, 모든 API 엔드포인트를 관리합니다.
"""

from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from src.voice_man.schemas import AudioUploadResponse
from src.voice_man.services import (
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
