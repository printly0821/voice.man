"""
Classification Result Data Model
SPEC-CRIME-CLASS-001 Phase 4: Fusion & Format
Defines complete classification result structure
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from voice_man.models.crime_classification.crime_types import CrimeClassification
from voice_man.models.crime_classification.legal_requirements import LegalEvidenceMapping
from voice_man.models.crime_classification.psychological_profile import PsychologicalProfile


@dataclass
class CrimeClassificationResult:
    """
    범죄 분류 종합 결과 (Crime Classification Result)

    Complete result of multimodal crime classification analysis

    Attributes:
        analysis_id: 분석 ID
        analyzed_at: 분석 시간
        speaker_id: 화자 ID
        classifications: 범죄 분류 결과 목록 (신뢰도 순)
        legal_mappings: 법적 증거 매핑 결과 목록
        psychological_profile: 심리 프로파일 (선택적)
        summary: 분석 요약
        recommendations: 권고사항 목록
    """

    analysis_id: str
    analyzed_at: datetime
    speaker_id: str
    classifications: List[CrimeClassification]
    legal_mappings: List[LegalEvidenceMapping]
    psychological_profile: Optional[PsychologicalProfile]
    summary: str
    recommendations: List[str]
