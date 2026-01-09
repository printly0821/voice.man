"""
범죄 언어 패턴 데이터 모델
SPEC-FORENSIC-001 Phase 2-A: Crime Language Pattern Database

가스라이팅, 협박, 강압, 기만 언어 패턴을 위한 Pydantic 모델 정의
학술 연구 기반 (Armenian Folia Anglistika 2024, Speech Acts analysis 2025)
"""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict


# ==============================================================================
# Enums - 패턴 유형 정의
# ==============================================================================


class GaslightingType(str, Enum):
    """
    가스라이팅 패턴 유형 (7가지)
    Based on: Armenian Folia Anglistika 2024, Speech Acts analysis 2025
    """

    DENIAL = "denial"  # 부정: 상대방의 경험/기억 부정
    COUNTERING = "countering"  # 반박: 상대방의 기억이 틀렸다고 주장
    TRIVIALIZING = "trivializing"  # 최소화: 상대방의 감정/경험 축소
    DIVERTING = "diverting"  # 화제전환: 대화 주제를 바꿔버림
    BLOCKING = "blocking"  # 차단: 대화 자체를 막음
    BLAME_SHIFTING = "blame_shifting"  # 책임전가: 책임을 상대방에게 돌림
    REALITY_DISTORTION = "reality_distortion"  # 현실왜곡: 현실을 왜곡하여 인식


class ThreatType(str, Enum):
    """
    협박 패턴 유형 (5가지)
    """

    DIRECT = "direct"  # 직접적 협박
    CONDITIONAL = "conditional"  # 조건부 협박
    VEILED = "veiled"  # 암시적 협박
    ECONOMIC = "economic"  # 경제적 협박
    SOCIAL = "social"  # 사회적 협박


class CoercionType(str, Enum):
    """
    강압 패턴 유형 (3가지)
    """

    EMOTIONAL = "emotional"  # 감정적 강압
    GUILT_INDUCTION = "guilt_induction"  # 죄책감 유발
    ISOLATION = "isolation"  # 고립화


class DeceptionCategory(str, Enum):
    """
    기만 언어 지표 카테고리 (5가지)
    """

    HEDGING = "hedging"  # 회피어
    DISTANCING = "distancing"  # 거리두기
    NEGATIVE_EMOTION = "negative_emotion"  # 부정 감정어
    EXCLUSIVE = "exclusive"  # 배타적 표현
    COGNITIVE_COMPLEXITY = "cognitive_complexity"  # 인지 복잡성


# ==============================================================================
# Pattern Definition Models - 패턴 정의 모델
# ==============================================================================


class GaslightingPattern(BaseModel):
    """
    가스라이팅 패턴 정의 모델
    JSON DB에서 로드되는 패턴 데이터 구조
    """

    model_config = ConfigDict(use_enum_values=True, str_strip_whitespace=True)

    type: GaslightingType = Field(..., description="가스라이팅 유형")
    patterns_ko: List[str] = Field(..., min_length=1, description="한국어 패턴 목록")
    patterns_en: List[str] = Field(default_factory=list, description="영어 패턴 목록")
    severity_weight: float = Field(..., ge=0.0, le=1.0, description="심각도 가중치 (0.0 ~ 1.0)")
    description_ko: str = Field(..., description="한국어 설명")
    description_en: str = Field(default="", description="영어 설명")


class ThreatPattern(BaseModel):
    """
    협박 패턴 정의 모델
    """

    model_config = ConfigDict(use_enum_values=True, str_strip_whitespace=True)

    type: ThreatType = Field(..., description="협박 유형")
    patterns_ko: List[str] = Field(..., min_length=1, description="한국어 패턴 목록")
    patterns_en: List[str] = Field(default_factory=list, description="영어 패턴 목록")
    severity_weight: float = Field(..., ge=0.0, le=1.0, description="심각도 가중치 (0.0 ~ 1.0)")
    description_ko: str = Field(..., description="한국어 설명")
    description_en: str = Field(default="", description="영어 설명")


class CoercionPattern(BaseModel):
    """
    강압 패턴 정의 모델
    """

    model_config = ConfigDict(use_enum_values=True, str_strip_whitespace=True)

    type: CoercionType = Field(..., description="강압 유형")
    patterns_ko: List[str] = Field(..., min_length=1, description="한국어 패턴 목록")
    patterns_en: List[str] = Field(default_factory=list, description="영어 패턴 목록")
    severity_weight: float = Field(..., ge=0.0, le=1.0, description="심각도 가중치 (0.0 ~ 1.0)")
    description_ko: str = Field(..., description="한국어 설명")
    description_en: str = Field(default="", description="영어 설명")


class DeceptionMarker(BaseModel):
    """
    기만 언어 지표 정의 모델
    """

    model_config = ConfigDict(use_enum_values=True, str_strip_whitespace=True)

    category: DeceptionCategory = Field(..., description="기만 지표 카테고리")
    markers_ko: List[str] = Field(..., min_length=1, description="한국어 지표 목록")
    markers_en: List[str] = Field(default_factory=list, description="영어 지표 목록")
    description_ko: str = Field(..., description="한국어 설명")
    description_en: str = Field(default="", description="영어 설명")


# ==============================================================================
# Match Result Models - 매칭 결과 모델
# ==============================================================================


class GaslightingMatch(BaseModel):
    """
    가스라이팅 패턴 매칭 결과
    """

    model_config = ConfigDict(use_enum_values=True)

    type: GaslightingType = Field(..., description="매칭된 가스라이팅 유형")
    matched_pattern: str = Field(..., description="매칭된 패턴 문자열")
    text: str = Field(..., description="원본 텍스트")
    start_position: int = Field(..., ge=0, description="매칭 시작 위치")
    end_position: int = Field(..., ge=0, description="매칭 종료 위치")
    confidence: float = Field(..., ge=0.0, le=1.0, description="신뢰도 (0.0 ~ 1.0)")
    severity_weight: float = Field(..., ge=0.0, le=1.0, description="심각도 가중치 (0.0 ~ 1.0)")
    speaker: Optional[str] = Field(default=None, description="화자 ID")
    timestamp: Optional[str] = Field(default=None, description="발화 시간")


class ThreatMatch(BaseModel):
    """
    협박 패턴 매칭 결과
    """

    model_config = ConfigDict(use_enum_values=True)

    type: ThreatType = Field(..., description="매칭된 협박 유형")
    matched_pattern: str = Field(..., description="매칭된 패턴 문자열")
    text: str = Field(..., description="원본 텍스트")
    start_position: int = Field(..., ge=0, description="매칭 시작 위치")
    end_position: int = Field(..., ge=0, description="매칭 종료 위치")
    confidence: float = Field(..., ge=0.0, le=1.0, description="신뢰도 (0.0 ~ 1.0)")
    severity_weight: float = Field(..., ge=0.0, le=1.0, description="심각도 가중치 (0.0 ~ 1.0)")
    speaker: Optional[str] = Field(default=None, description="화자 ID")
    timestamp: Optional[str] = Field(default=None, description="발화 시간")


class CoercionMatch(BaseModel):
    """
    강압 패턴 매칭 결과
    """

    model_config = ConfigDict(use_enum_values=True)

    type: CoercionType = Field(..., description="매칭된 강압 유형")
    matched_pattern: str = Field(..., description="매칭된 패턴 문자열")
    text: str = Field(..., description="원본 텍스트")
    start_position: int = Field(..., ge=0, description="매칭 시작 위치")
    end_position: int = Field(..., ge=0, description="매칭 종료 위치")
    confidence: float = Field(..., ge=0.0, le=1.0, description="신뢰도 (0.0 ~ 1.0)")
    severity_weight: float = Field(..., ge=0.0, le=1.0, description="심각도 가중치 (0.0 ~ 1.0)")
    speaker: Optional[str] = Field(default=None, description="화자 ID")
    timestamp: Optional[str] = Field(default=None, description="발화 시간")


class DeceptionMarkerMatch(BaseModel):
    """
    기만 언어 지표 매칭 결과
    """

    model_config = ConfigDict(use_enum_values=True)

    category: DeceptionCategory = Field(..., description="매칭된 기만 지표 카테고리")
    marker: str = Field(..., description="매칭된 지표 문자열")
    count: int = Field(..., ge=0, description="발생 횟수")


# ==============================================================================
# Analysis Result Models - 분석 결과 모델
# ==============================================================================


class DeceptionAnalysis(BaseModel):
    """
    기만 언어 분석 결과
    """

    model_config = ConfigDict(use_enum_values=True)

    marker_matches: List[DeceptionMarkerMatch] = Field(
        default_factory=list, description="매칭된 기만 지표 목록"
    )
    hedging_score: float = Field(default=0.0, ge=0.0, le=1.0, description="회피어 점수")
    distancing_score: float = Field(default=0.0, ge=0.0, le=1.0, description="거리두기 점수")
    negative_emotion_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="부정 감정어 점수"
    )
    exclusive_score: float = Field(default=0.0, ge=0.0, le=1.0, description="배타적 표현 점수")
    cognitive_complexity_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="인지 복잡성 점수"
    )
    overall_deception_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="종합 기만 점수"
    )


class CrimeLanguageScore(BaseModel):
    """
    범죄 언어 종합 점수
    """

    model_config = ConfigDict(use_enum_values=True)

    gaslighting_score: float = Field(default=0.0, ge=0.0, le=1.0, description="가스라이팅 점수")
    threat_score: float = Field(default=0.0, ge=0.0, le=1.0, description="협박 점수")
    coercion_score: float = Field(default=0.0, ge=0.0, le=1.0, description="강압 점수")
    deception_score: float = Field(default=0.0, ge=0.0, le=1.0, description="기만 점수")
    overall_risk_score: float = Field(default=0.0, ge=0.0, le=1.0, description="종합 위험 점수")
    risk_level: str = Field(default="낮음", description="위험 수준 (낮음, 중간, 높음, 매우 높음)")
    gaslighting_count: int = Field(default=0, ge=0, description="가스라이팅 패턴 수")
    threat_count: int = Field(default=0, ge=0, description="협박 패턴 수")
    coercion_count: int = Field(default=0, ge=0, description="강압 패턴 수")
