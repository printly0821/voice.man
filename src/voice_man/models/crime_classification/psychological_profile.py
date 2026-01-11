"""
Psychological Profile Data Model
SPEC-CRIME-CLASS-001 Phase 2: Psychological Profiling
Defines psychological profile structure
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class PsychologicalProfile:
    """
    심리 프로파일 (Psychological Profile)

    Dark Triad personality traits and crime propensity analysis

    Attributes:
        dark_triad_scores: 다크 트라이어드 점수
            - narcissism: 자기애성 점수
            - machiavellianism: 권력지향성 점수
            - psychopathy: 심리병질적 성향 점수
        attachment_style: 애착 유형 (예: "anxious_avoidant")
        dominant_traits: 주요 성격 특성 목록
        crime_propensity: 범죄 성향 예측 점수 (범죄 유형별)
    """

    dark_triad_scores: Dict[str, float]
    attachment_style: str
    dominant_traits: List[str]
    crime_propensity: Dict[str, float]
