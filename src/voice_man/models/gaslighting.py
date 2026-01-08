"""
가스라이팅 패턴 모델

GaslightingPattern, GaslightingPatternType, GaslightingAnalysisResult 정의
"""

from enum import Enum
from typing import List, Dict
from pydantic import BaseModel, Field, ConfigDict


class GaslightingPatternType(str, Enum):
    """가스라이팅 패턴 유형 열거형"""

    DENIAL = "부정"  # 상대방의 경험/기억 부정
    BLAME_SHIFTING = "전가"  # 책임을 상대방에게 전가
    MINIMIZING = "축소"  # 상대방의 감정/경험 축소
    CONFUSION = "혼란"  # 의도적 혼란 유발


class GaslightingPattern(BaseModel):
    """가스라이팅 패턴 모델"""

    model_config = ConfigDict(use_enum_values=True)

    type: GaslightingPatternType = Field(..., description="패턴 유형")
    confidence: float = Field(..., ge=0.0, le=1.0, description="신뢰도 점수 (0.0 ~ 1.0)")
    intensity: float = Field(default=0.5, ge=0.0, le=1.0, description="패턴 강도 (0.0 ~ 1.0)")
    speaker: str = Field(..., description="화자 ID")
    text: str = Field(..., description="발언 텍스트")
    timestamp: str = Field(..., description="발언 시간 (HH:MM:SS)")
    matched_keywords: List[str] = Field(default_factory=list, description="매칭된 키워드")


class GaslightingAnalysisResult(BaseModel):
    """가스라이팅 분석 결과 모델"""

    patterns: List[GaslightingPattern] = Field(default_factory=list, description="감지된 패턴 목록")
    pattern_frequency: Dict[GaslightingPatternType, int] = Field(
        default_factory=dict, description="패턴 빈도수"
    )
    speaker_patterns: Dict[str, List[GaslightingPattern]] = Field(
        default_factory=dict, description="화자별 패턴"
    )
    risk_level: str = Field(default="낮음", description="종합 위험도 (낮음, 중간, 높음, 매우 높음)")
    recommendations: List[str] = Field(default_factory=list, description="권고 조치")
    analysis_timestamp: str = Field(..., description="분석 시간")
