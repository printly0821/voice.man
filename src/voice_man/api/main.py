"""
FastAPI Main Application

Voice Man 시스템의 진입점으로, 모든 API 엔드포인트를 관리합니다.
"""

from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from pydantic import BaseModel

from voice_man.schemas import AudioUploadResponse
from voice_man.services import (
    compute_sha256_hash,
    generate_file_id,
    is_supported_audio_format,
)
from voice_man.services.report_service import ReportService
from voice_man.models.audio_file import AudioFile, ReportStatus
from voice_man.services.pdf_service import PDFService
from voice_man.services.report_template_service import ReportTemplateService
from voice_man.services.chart_service import ChartService

# 포렌식 증거 API 라우터 import
from voice_man.api.web.evidence import router as evidence_router

app = FastAPI(
    title="Voice Man API",
    description="음성 녹취 증거 분석 시스템",
    version="0.1.0",
)

# 포렌식 증거 API 라우터 등록
app.include_router(evidence_router)

# Initialize services
report_service = ReportService()
pdf_service = PDFService()
template_service = ReportTemplateService()
chart_service = ChartService()


class HealthResponse(BaseModel):
    """Health check 응답 모델"""

    status: str


class ReportGenerationResponse(BaseModel):
    """보고서 생성 시작 응답 모델"""

    report_id: str
    status: str
    estimated_time_seconds: int


class ReportStatusResponse(BaseModel):
    """보고서 상태 조회 응답 모델"""

    status: str
    progress_percentage: int
    current_step: str
    completed_at: str | None = None
    file_path: str | None = None


class ReportVersionInfo(BaseModel):
    """보고서 버전 정보 모델"""

    version: int
    created_at: str
    file_path: str


class ReportVersionsResponse(BaseModel):
    """보고서 버전 목록 응답 모델"""

    versions: list[ReportVersionInfo]


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


@app.post(
    "/api/v1/audio/{audio_id}/report/generate",
    response_model=ReportGenerationResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def generate_report(
    audio_id: int,
    force: bool = False,
) -> ReportGenerationResponse:
    """
    보고서 생성 시작 엔드포인트

    비동기적으로 법적 보고서를 생성합니다.

    Args:
        audio_id: 오디오 파일 ID
        force: 기존 보고서가 있을 경우 강제 재생성

    Returns:
        ReportGenerationResponse: 생성 작업 정보

    Raises:
        HTTPException: 오디오 파일을 찾을 수 없거나 전사가 없는 경우
    """
    try:
        result = await report_service.start_report_generation(audio_id, force=force)
        return ReportGenerationResponse(
            report_id=result["report_id"],
            status=result["status"],
            estimated_time_seconds=result["estimated_time_seconds"],
        )
    except ValueError as e:
        if "not found" in str(e):
            raise HTTPException(status_code=404, detail=str(e))
        elif "transcript" in str(e):
            raise HTTPException(status_code=400, detail=str(e))
        elif "already exists" in str(e):
            raise HTTPException(status_code=409, detail=str(e))
        else:
            raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/v1/audio/{audio_id}/report/status", response_model=ReportStatusResponse)
async def get_report_status(audio_id: int) -> ReportStatusResponse:
    """
    보고서 생성 상태 조회 엔드포인트

    Args:
        audio_id: 오디오 파일 ID

    Returns:
        ReportStatusResponse: 상태 정보

    Raises:
        HTTPException: 보고서를 찾을 수 없는 경우
    """
    try:
        status_info = await report_service.get_report_status(audio_id)
        return ReportStatusResponse(**status_info)
    except ValueError as e:
        if "not found" in str(e):
            raise HTTPException(status_code=404, detail=str(e))
        else:
            raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/v1/audio/{audio_id}/report/download")
async def download_report(audio_id: int):
    """
    보고서 PDF 다운로드 엔드포인트

    Args:
        audio_id: 오디오 파일 ID

    Returns:
        FileResponse: PDF 파일

    Raises:
        HTTPException: 보고서를 찾을 수 없거나 준비되지 않은 경우
    """
    try:
        file_path = await report_service.get_report_file_path(audio_id)

        if not file_path or not Path(file_path).exists():
            raise HTTPException(status_code=500, detail="Report file not found on filesystem")

        return FileResponse(
            path=file_path,
            media_type="application/pdf",
            filename=f"report_{audio_id}.pdf",
        )
    except ValueError as e:
        if "not found" in str(e):
            raise HTTPException(status_code=404, detail=str(e))
        elif "not ready" in str(e):
            raise HTTPException(status_code=425, detail=str(e))
        else:
            raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/v1/audio/{audio_id}/report/versions", response_model=ReportVersionsResponse)
async def list_report_versions(audio_id: int) -> ReportVersionsResponse:
    """
    보고서 버전 목록 조회 엔드포인트

    Args:
        audio_id: 오디오 파일 ID

    Returns:
        ReportVersionsResponse: 버전 목록

    Raises:
        HTTPException: 오디오 파일을 찾을 수 없는 경우
    """
    try:
        versions = await report_service.get_report_versions(audio_id)
        return ReportVersionsResponse(versions=versions)
    except ValueError as e:
        if "not found" in str(e):
            raise HTTPException(status_code=404, detail=str(e))
        else:
            raise HTTPException(status_code=400, detail=str(e))
