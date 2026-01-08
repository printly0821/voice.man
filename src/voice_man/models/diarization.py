"""
화자 분리 (Speaker Diarization) 모델

pyannote-audio를 사용한 화자 분리 결과 모델입니다.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import List


class Speaker(BaseModel):
    """개별 화자 정보"""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "speaker_id": "SPEAKER_00",
                "start_time": 0.0,
                "end_time": 150.0,
                "duration": 150.0,
                "confidence": 0.95,
            }
        }
    )

    speaker_id: str = Field(..., description="화자 ID (예: SPEAKER_00, SPEAKER_01)")
    start_time: float = Field(..., ge=0, description="발화 시작 시간 (초)")
    end_time: float = Field(..., gt=0, description="발화 종료 시간 (초)")
    duration: float = Field(..., gt=0, description="발화 지속 시간 (초)")
    confidence: float = Field(..., ge=0, le=1, description="신뢰도 점수")


class SpeakerTurn(BaseModel):
    """화자 교대 (Turn-taking) 정보"""

    speaker_id: str = Field(..., description="화자 ID")
    start_time: float = Field(..., ge=0, description="턴 시작 시간 (초)")
    end_time: float = Field(..., gt=0, description="턴 종료 시간 (초)")
    duration: float = Field(..., gt=0, description="턴 지속 시간 (초)")


class SpeakerStats(BaseModel):
    """화자 통계 정보"""

    total_speakers: int = Field(..., ge=1, description="총 화자 수")
    total_speech_duration: float = Field(..., ge=0, description="총 발화 시간 (초)")
    speaker_details: List[Speaker] = Field(..., description="각 화자별 상세 정보")


class DiarizationResult(BaseModel):
    """화자 분리 결과"""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "speakers": [
                    {
                        "speaker_id": "SPEAKER_00",
                        "start_time": 0.0,
                        "end_time": 150.0,
                        "duration": 150.0,
                        "confidence": 0.95,
                    }
                ],
                "total_duration": 300.0,
                "num_speakers": 2,
            }
        }
    )

    speakers: List[Speaker] = Field(..., description="화자별 세그먼트 목록")
    total_duration: float = Field(..., ge=0, description="전체 오디오 길이 (초)")
    num_speakers: int = Field(..., ge=1, description="감지된 화자 수")
