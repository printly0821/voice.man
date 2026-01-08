"""
범죄 발언 태깅 모델

CrimeTag, CrimeType, 및 관련 데이터 모델 정의
"""

from enum import Enum
from typing import List
from pydantic import BaseModel, Field, ConfigDict


class CrimeType(str, Enum):
    """범죄 유형 열거형"""

    THREAT = "협박"  # 형법 제283조
    INTIMIDATION = "공갈"  # 형법 제350조
    FRAUD = "사기"  # 형법 제347조
    INSULT = "모욕"  # 형법 제311조


class CrimeTag(BaseModel):
    """범죄 발언 태그 모델"""

    model_config = ConfigDict(use_enum_values=True)

    type: CrimeType = Field(..., description="범죄 유형")
    confidence: float = Field(..., ge=0.0, le=1.0, description="신뢰도 점수 (0.0 ~ 1.0)")
    keywords: List[str] = Field(default_factory=list, description="탐지된 키워드 목록")
    legal_reference: str = Field(..., description="관련 법조문")
    needs_review: bool = Field(default=False, description="검토 필요 여부")


class CrimeAnalysisResult(BaseModel):
    """범죄 분석 결과 모델"""

    total_crimes: int = Field(..., description="탐지된 총 범죄 발언 수")
    tags: List[CrimeTag] = Field(default_factory=list, description="범죄 태그 목록")
    analysis_timestamp: str = Field(..., description="분석 시간")
