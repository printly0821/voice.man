"""
Pydantic 스키마 정의

API 요청 및 응답 모델을 정의합니다.
"""

from pydantic import BaseModel


class AudioUploadResponse(BaseModel):
    """오디오 파일 업로드 응답 모델"""

    file_id: str
    filename: str
    content_type: str
    file_size: int
    sha256_hash: str


class ErrorResponse(BaseModel):
    """에러 응답 모델"""

    detail: str
