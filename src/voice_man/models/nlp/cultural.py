"""
Cultural Analysis Data Models
SPEC-NLP-KOBERT-001 TAG-003: Cultural Context Analysis
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


class SpeechLevel(str, Enum):
    """Korean speech level (politeness level)"""

    FORMAL = "formal"  # 존댓말 (합쇼체, 해요체)
    INFORMAL = "informal"  # 반말 (해체, 해라체)
    MIXED = "mixed"  # 혼용


class HierarchyType(str, Enum):
    """Hierarchy relationship types"""

    JOB_TITLE = "job_titles"  # 직책 (부장님, 사장님)
    KINSHIP = "kinship"  # 친족 (할머니, 아버지)
    SOCIAL = "social"  # 사회적 (선생님, 고객님)
    PEER = "peer"  # 동료/친구


@dataclass
class LevelTransition:
    """
    Speech level transition within text

    Attributes:
        position: Character position of transition
        from_level: Source speech level
        to_level: Target speech level
        context: Context around transition
    """

    position: int
    from_level: SpeechLevel
    to_level: SpeechLevel
    context: str


@dataclass
class SpeechLevelResult:
    """
    Speech level analysis result

    Attributes:
        level: Dominant speech level
        formal_ratio: Ratio of formal speech (0-1)
        informal_ratio: Ratio of informal speech (0-1)
        level_transitions: List of speech level transitions
        detected_markers: Detected hierarchy/speech markers
    """

    level: SpeechLevel
    formal_ratio: float
    informal_ratio: float
    level_transitions: List[LevelTransition]
    detected_markers: Optional[List[str]] = None

    def __post_init__(self):
        """Validate speech level result"""
        if not 0 <= self.formal_ratio <= 1:
            raise ValueError(f"formal_ratio must be between 0 and 1, got {self.formal_ratio}")

        if not 0 <= self.informal_ratio <= 1:
            raise ValueError(f"informal_ratio must be between 0 and 1, got {self.informal_ratio}")

        # Ratios should sum to approximately 1
        total = self.formal_ratio + self.informal_ratio
        if abs(total - 1.0) > 0.1:  # Allow 10% tolerance
            raise ValueError(f"formal_ratio + informal_ratio should sum to ~1.0, got {total}")
