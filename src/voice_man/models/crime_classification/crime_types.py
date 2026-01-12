"""
Crime Type Data Models
SPEC-CRIME-CLASS-001 Phase 1: Data Models & Pattern Extension
Defines 11 crime types and related data structures
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List


class CrimeType(Enum):
    """
    범죄 유형 열거형 (11개 죄명)

    11 Crime Types for Classification System
    """

    DOMESTIC_VIOLENCE = "가정폭력"
    STALKING = "스토킹"
    THREAT = "협박"
    GASLIGHTING = "가스라이팅"
    FRAUD = "사기"
    EXTORTION = "공갈"
    COERCION = "강요"
    INSULT = "모욕"
    EMBEZZLEMENT = "횡령"
    BREACH_OF_TRUST = "배임"
    TAX_EVASION = "조세포탈"


@dataclass
class ModalityScore:
    """
    모달리티별 점수 (Modality Scores)

    Attributes:
        text_score: 텍스트 분석 기반 점수 (0.0-1.0)
        audio_score: 음성 분석 기반 점수 (0.0-1.0)
        psychological_score: 심리 분석 기반 점수 (0.0-1.0)
    """

    text_score: float
    audio_score: float
    psychological_score: float

    def __post_init__(self):
        """Validate that scores are between 0 and 1"""
        for name, value in [
            ("text_score", self.text_score),
            ("audio_score", self.audio_score),
            ("psychological_score", self.psychological_score),
        ]:
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1, got {value}")


@dataclass
class CrimeClassification:
    """
    범죄 분류 결과 (Crime Classification Result)

    Attributes:
        crime_type: 범죄 유형
        confidence: 분류 신뢰도 (0.0-1.0)
        confidence_interval: 95% 신뢰 구간 {"lower_95": 0.7, "upper_95": 0.9}
        modality_scores: 모달리티별 점수
        weighted_score: 가중 앙상블 점수
        legal_reference: 형법 조문 참조 (예: "형법 제347조")
        evidence_items: 증거 항목 목록
        requires_review: 전문가 검토 필요 여부
    """

    crime_type: CrimeType
    confidence: float
    confidence_interval: Dict[str, float]
    modality_scores: ModalityScore
    weighted_score: float
    legal_reference: str
    evidence_items: List[str]
    requires_review: bool
