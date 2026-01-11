"""
Psychological Profiling Service
SPEC-CRIME-CLASS-001 Phase 4: Psychological Profiling (Dark Triad)
"""

from typing import Dict, List

from voice_man.models.crime_classification.psychological_profile import (
    PsychologicalProfile,
)


class PsychologicalProfiler:
    """
    심리 프로파일링 서비스

    Dark Triad personality traits analysis and crime propensity prediction
    """

    def __init__(self) -> None:
        """Initialize psychological profiler"""
        self._trait_indicators = self._load_trait_indicators()

    def _load_trait_indicators(self) -> Dict[str, List[str]]:
        """Load psychological trait indicators"""
        return {
            "narcissism": [
                "내가",
                "내가해",
                "내말",
                "내권리",
                "나",
                "내자신",
                "내능력",
                "내성격",
                "자신있",
                "자부",
                "자랑",
            ],
            "machiavellianism": [
                "이득",
                "이용",
                "조작",
                "속임",
                "기만",
                "계산",
                "전략",
                "목적",
                "수단",
                "성공",
            ],
            "psychopathy": [
                "상관",
                "안",
                "문제",
                "책임",
                "후회",
                "감정",
                "공감",
                "양심",
                "죄책감",
                "미안",
            ],
        }

    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analyze text for psychological traits

        Args:
            text: Text to analyze

        Returns:
            Dictionary with dark triad scores
        """
        dark_triad_scores = {
            "narcissism": 0.0,
            "machiavellianism": 0.0,
            "psychopathy": 0.0,
        }

        for trait, indicators in self._trait_indicators.items():
            match_count = sum(1 for indicator in indicators if indicator in text)
            # Normalize score: matches / total indicators
            dark_triad_scores[trait] = min(match_count / len(indicators), 1.0)

        return dark_triad_scores

    def infer_attachment_style(self, dark_triad_scores: Dict[str, float]) -> str:
        """
        Infer attachment style from dark triad scores

        Args:
            dark_triad_scores: Dark triad personality scores

        Returns:
            Attachment style classification
        """
        narcissism = dark_triad_scores.get("narcissism", 0)
        psychopathy = dark_triad_scores.get("psychopathy", 0)

        if narcissism > 0.6 or psychopathy > 0.6:
            return "anxious_avoidant"
        elif narcissism > 0.4:
            return "avoidant"
        else:
            return "secure"

    def extract_dominant_traits(self, dark_triad_scores: Dict[str, float]) -> List[str]:
        """
        Extract dominant personality traits

        Args:
            dark_triad_scores: Dark triad personality scores

        Returns:
            List of dominant trait names
        """
        dominant_traits = []

        trait_mapping = {
            "narcissism": ["grandiosity", "superiority", "self_admiration"],
            "machiavellianism": ["manipulation", "strategic", "opportunism"],
            "psychopathy": ["lack_of_empathy", "impulsivity", "coldness"],
        }

        for trait, score in dark_triad_scores.items():
            if score > 0.5:
                dominant_traits.extend(trait_mapping.get(trait, []))

        return dominant_traits if dominant_traits else ["no_dominant_traits"]

    def predict_crime_propensity(self, dark_triad_scores: Dict[str, float]) -> Dict[str, float]:
        """
        Predict crime propensity based on psychological profile

        Args:
            dark_triad_scores: Dark triad personality scores

        Returns:
            Dictionary mapping crime types to propensity scores
        """
        # Crime propensity based on psychological research
        narcissism = dark_triad_scores.get("narcissism", 0)
        machiavellianism = dark_triad_scores.get("machiavellianism", 0)
        psychopathy = dark_triad_scores.get("psychopathy", 0)

        return {
            "gaslighting": narcissism * 0.8 + machiavellianism * 0.6,
            "fraud": machiavellianism * 0.9 + narcissism * 0.4,
            "coercion": psychopathy * 0.7 + machiavellianism * 0.5,
            "threat": psychopathy * 0.8,
            "extortion": machiavellianism * 0.7 + psychopathy * 0.6,
        }

    def create_profile(self, text: str) -> PsychologicalProfile:
        """
        Create complete psychological profile from text

        Args:
            text: Text to analyze

        Returns:
            PsychologicalProfile object
        """
        dark_triad_scores = self.analyze_text(text)
        attachment_style = self.infer_attachment_style(dark_triad_scores)
        dominant_traits = self.extract_dominant_traits(dark_triad_scores)
        crime_propensity = self.predict_crime_propensity(dark_triad_scores)

        return PsychologicalProfile(
            dark_triad_scores=dark_triad_scores,
            attachment_style=attachment_style,
            dominant_traits=dominant_traits,
            crime_propensity=crime_propensity,
        )
