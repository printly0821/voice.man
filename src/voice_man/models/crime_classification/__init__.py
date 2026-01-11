"""
Crime Classification Models
SPEC-CRIME-CLASS-001 Phase 1: Data Models & Pattern Extension
"""

from voice_man.models.crime_classification.crime_types import (
    CrimeType,
    ModalityScore,
    CrimeClassification,
)
from voice_man.models.crime_classification.legal_requirements import (
    LegalRequirement,
    LegalEvidenceMapping,
)
from voice_man.models.crime_classification.psychological_profile import (
    PsychologicalProfile,
)
from voice_man.models.crime_classification.classification_result import (
    CrimeClassificationResult,
)

__all__ = [
    "CrimeType",
    "ModalityScore",
    "CrimeClassification",
    "LegalRequirement",
    "LegalEvidenceMapping",
    "PsychologicalProfile",
    "CrimeClassificationResult",
]
