"""
Korean Cultural Context Analyzer
SPEC-NLP-KOBERT-001 TASK-003: Korean cultural context analyzer
"""

import logging
import re
from dataclasses import dataclass, field
from typing import List, Literal, Optional

logger = logging.getLogger(__name__)


@dataclass
class LevelTransition:
    """Speech level transition point"""

    position: int  # Character position in text
    from_level: str  # Source level
    to_level: str  # Target level
    context: str  # Context around transition


@dataclass
class SpeechLevelResult:
    """존댓말/반말 분석 결과"""

    level: Literal["formal", "informal", "mixed"]
    formal_ratio: float
    informal_ratio: float
    level_transitions: List[LevelTransition]


@dataclass
class HierarchyContext:
    """위계 관계 맥락"""

    has_family_markers: bool
    has_job_title_markers: bool
    has_social_markers: bool
    detected_relationships: List[str]


@dataclass
class ManipulationPattern:
    """조작적 표현 패턴"""

    pattern: str
    category: str
    confidence: float
    position: Optional[int] = None


@dataclass
class ComprehensiveAnalysisResult:
    """종합 문화적 맥락 분석 결과"""

    speech_level: SpeechLevelResult
    hierarchy_context: HierarchyContext
    manipulation_patterns: List[ManipulationPattern]


class KoreanCulturalAnalyzer:
    """
    한국어 문화적 맥락 분석기

    Features:
        - 존댓말/반말 분석
        - 위계 관계 패턴 탐지
        - 한국어 특유 조작 표현 패턴 탐지
        - 종합 문화적 맥락 분석
    """

    # Formal speech patterns (존댓말)
    FORMAL_PATTERNS = [
        r"입니다\b",  # 입니다체
        r"세요\b",  # 해요체
        r"거든요\b",
        r"군요\b",
        r"나요\b",
        r"까요\b",
        r"지요\b",
        r"답\b",  # 합쇼체
    ]

    # Informal speech patterns (반말)
    INFORMAL_PATTERNS = [
        r"아\b",  # 해체
        r"어\b",
        r"야\b",  # 해라체
        r"다\b",
        r"니\b",
        r"까\b",
        r"지\b",
        r"넨\b",
    ]

    # Family hierarchy markers
    FAMILY_MARKERS = [
        "할머니",
        "할아버지",
        "어머니",
        "아버지",
        "누나",
        "형",
        "오빠",
        "언니",
        "동생",
        "조카",
        "사촌",
        "삼촌",
        "고모",
        "숙모",
    ]

    # Job title markers
    JOB_TITLE_MARKERS = [
        "사장님",
        "부장님",
        "과장님",
        "대리님",
        "사원님",
        "팀장님",
        "실장님",
        "지점장님",
        "원장님",
        "국장님",
    ]

    # Social hierarchy markers
    SOCIAL_MARKERS = [
        "선생님",
        "교수님",
        "변호사님",
        "의사선생님",
        "고객님",
        "손님",
        "회원님",
        "회장님",
        "총리님",
        "대통령님",
    ]

    # Korean manipulation patterns
    MANIPULATION_PATTERNS = {
        "gaslighting": [
            r"네가\s*(잘못|틀린)\s*했으니까",
            r"다\s*널\s*위해서야",
            r"그렇게\s*느끼는\s*게\s*이상한\s*거야",
            r"기억이\s*나\s*못하네",
            r"너랑\s*상관없어",
        ],
        "threat": [
            r"큰일\s*날\s*거야",
            r"후회하게\s*될\s*거야",
            r"신\s*중하게\s*생각해",
            r"그러면\s*안돼",
        ],
        "coercion": [
            r"너라면\s*할\s*수\s*있잖아",
            r"나를\s*실망시키지\s*마",
            r"너\s*때문에야",
            r"너의\s*책임이야",
            r"부탁할\s*게",
        ],
    }

    def __init__(self) -> None:
        """Initialize Korean cultural analyzer"""
        # Compile regex patterns for efficiency
        self._formal_regex = re.compile("|".join(self.FORMAL_PATTERNS), re.IGNORECASE)
        self._informal_regex = re.compile("|".join(self.INFORMAL_PATTERNS), re.IGNORECASE)

        logger.info("KoreanCulturalAnalyzer initialized")

    def analyze_speech_level(self, text: str) -> SpeechLevelResult:
        """
        Analyze speech level (formal/informal/mixed)

        Args:
            text: Input text

        Returns:
            SpeechLevelResult with analysis results
        """
        if not text or not text.strip():
            return SpeechLevelResult(
                level="mixed", formal_ratio=0.0, informal_ratio=0.0, level_transitions=[]
            )

        # Count formal and informal markers
        formal_matches = list(self._formal_regex.finditer(text))
        informal_matches = list(self._informal_regex.finditer(text))

        total_markers = len(formal_matches) + len(informal_matches)

        if total_markers == 0:
            # No explicit markers, assume informal (default in casual speech)
            return SpeechLevelResult(
                level="informal", formal_ratio=0.0, informal_ratio=1.0, level_transitions=[]
            )

        formal_ratio = len(formal_matches) / total_markers
        informal_ratio = len(informal_matches) / total_markers

        # Determine speech level
        if formal_ratio > 0.7:
            level = "formal"
        elif informal_ratio > 0.7:
            level = "informal"
        else:
            level = "mixed"

        # Detect level transitions
        level_transitions = self._detect_level_transitions(text, formal_matches, informal_matches)

        return SpeechLevelResult(
            level=level,
            formal_ratio=formal_ratio,
            informal_ratio=informal_ratio,
            level_transitions=level_transitions,
        )

    def _detect_level_transitions(
        self,
        text: str,
        formal_matches: List[re.Match],
        informal_matches: List[re.Match],
    ) -> List[LevelTransition]:
        """
        Detect speech level transitions in text

        Args:
            text: Input text
            formal_matches: List of formal marker matches
            informal_matches: List of informal marker matches

        Returns:
            List of LevelTransition objects
        """
        transitions = []
        all_matches = []

        for match in formal_matches:
            all_matches.append((match.start(), "formal"))

        for match in informal_matches:
            all_matches.append((match.start(), "informal"))

        # Sort by position
        all_matches.sort(key=lambda x: x[0])

        # Detect transitions
        for i in range(1, len(all_matches)):
            prev_pos, prev_level = all_matches[i - 1]
            curr_pos, curr_level = all_matches[i]

            if prev_level != curr_level:
                # Extract context (10 chars before and after)
                context_start = max(0, curr_pos - 10)
                context_end = min(len(text), curr_pos + 10)
                context = text[context_start:context_end]

                transitions.append(
                    LevelTransition(
                        position=curr_pos,
                        from_level=prev_level,
                        to_level=curr_level,
                        context=context,
                    )
                )

        return transitions

    def detect_hierarchy_context(self, text: str) -> HierarchyContext:
        """
        Detect hierarchy context markers

        Args:
            text: Input text

        Returns:
            HierarchyContext with detected markers
        """
        if not text or not text.strip():
            return HierarchyContext(
                has_family_markers=False,
                has_job_title_markers=False,
                has_social_markers=False,
                detected_relationships=[],
            )

        # Detect markers
        has_family_markers = any(marker in text for marker in self.FAMILY_MARKERS)
        has_job_title_markers = any(marker in text for marker in self.JOB_TITLE_MARKERS)
        has_social_markers = any(marker in text for marker in self.SOCIAL_MARKERS)

        # Collect detected relationships
        detected_relationships = []

        if has_family_markers:
            for marker in self.FAMILY_MARKERS:
                if marker in text:
                    detected_relationships.append(f"family:{marker}")

        if has_job_title_markers:
            for marker in self.JOB_TITLE_MARKERS:
                if marker in text:
                    detected_relationships.append(f"job:{marker}")

        if has_social_markers:
            for marker in self.SOCIAL_MARKERS:
                if marker in text:
                    detected_relationships.append(f"social:{marker}")

        return HierarchyContext(
            has_family_markers=has_family_markers,
            has_job_title_markers=has_job_title_markers,
            has_social_markers=has_social_markers,
            detected_relationships=detected_relationships,
        )

    def detect_manipulation_patterns(self, text: str) -> List[ManipulationPattern]:
        """
        Detect Korean manipulation patterns

        Args:
            text: Input text

        Returns:
            List of detected ManipulationPattern objects
        """
        if not text or not text.strip():
            return []

        detected_patterns = []

        for category, patterns in self.MANIPULATION_PATTERNS.items():
            for pattern in patterns:
                regex = re.compile(pattern, re.IGNORECASE)
                matches = regex.finditer(text)

                for match in matches:
                    # Calculate confidence based on pattern specificity
                    confidence = self._calculate_pattern_confidence(match.group())

                    detected_patterns.append(
                        ManipulationPattern(
                            pattern=match.group(),
                            category=category,
                            confidence=confidence,
                            position=match.start(),
                        )
                    )

        return detected_patterns

    def _calculate_pattern_confidence(self, pattern: str) -> float:
        """
        Calculate confidence score for a pattern

        Args:
            pattern: Detected pattern string

        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Base confidence
        confidence = 0.5

        # Increase confidence for longer, more specific patterns
        if len(pattern) > 10:
            confidence += 0.2
        if len(pattern) > 15:
            confidence += 0.2

        # Increase confidence for patterns with specific Korean endings
        if any(ending in pattern for ending in ["그래", "야", "잖아", "위해서", "이상한", "거야"]):
            confidence += 0.1

        return min(confidence, 1.0)

    def analyze_comprehensive(self, text: str) -> ComprehensiveAnalysisResult:
        """
        Perform comprehensive cultural context analysis

        Args:
            text: Input text

        Returns:
            ComprehensiveAnalysisResult with all analyses
        """
        speech_level = self.analyze_speech_level(text)
        hierarchy_context = self.detect_hierarchy_context(text)
        manipulation_patterns = self.detect_manipulation_patterns(text)

        return ComprehensiveAnalysisResult(
            speech_level=speech_level,
            hierarchy_context=hierarchy_context,
            manipulation_patterns=manipulation_patterns,
        )
