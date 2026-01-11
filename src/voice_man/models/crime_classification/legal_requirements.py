"""
Legal Requirements Data Models
SPEC-CRIME-CLASS-001 Phase 3: Legal Evidence Mapping
Defines legal requirements and evidence mapping structures
"""

from dataclasses import dataclass
from typing import List


@dataclass
class LegalRequirement:
    """
    법적 구성요건 (Legal Requirement)

    Attributes:
        name: 구성요건 이름 (예: "기망행위")
        description: 구성요건 설명
        indicators: 관련 지표 목록
        satisfied: 충족 여부
        evidence: 증거 항목 목록
    """

    name: str
    description: str
    indicators: List[str]
    satisfied: bool
    evidence: List[str]


@dataclass
class LegalEvidenceMapping:
    """
    법적 증거 매핑 결과 (Legal Evidence Mapping Result)

    Attributes:
        crime_type: 범죄 유형명 (Korean string)
        legal_code: 형법 조문 (예: "형법 제347조")
        requirements: 구성요건 목록
        fulfillment_rate: 충족률 (0.0-1.0)
        legal_viability: 법적 타당성 ("높음", "보통", "낮음")
    """

    crime_type: str
    legal_code: str
    requirements: List[LegalRequirement]
    fulfillment_rate: float
    legal_viability: str
