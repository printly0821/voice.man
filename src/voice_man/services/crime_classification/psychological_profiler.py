"""
Psychological Profiling Service
SPEC-CRIME-CLASS-001 Phase 4: Psychological Profiling (Dark Triad)

Phase 1 Improvement (from gaslighting-forensic-ai-guide.md):
- Expanded SD3 (Short Dark Triad) pattern keywords
- Added 20 additional keywords per trait (30 total per trait)
- Enhanced sub-trait classification

Phase 2 Enhancement: BERT + Random Forest Hybrid
- Added ML-based classification using KoBERT embeddings
- Maintains backward compatibility with keyword matching
- Automatic fallback to keyword-based if ML model unavailable
"""

from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Union

from voice_man.models.crime_classification.psychological_profile import (
    PsychologicalProfile,
)

if TYPE_CHECKING:
    from voice_man.services.crime_classification.bert_dark_triad_classifier import (
        BERTDarkTriadClassifier,
    )


class PsychologicalProfiler:
    """
    심리 프로파일링 서비스

    Dark Triad personality traits analysis and crime propensity prediction

    Phase 2 Enhancement:
    - Supports both keyword-based and ML-based (BERT+RF) classification
    - Automatic fallback to keyword-based if ML model unavailable
    - Backward compatible with existing code
    """

    def __init__(
        self,
        use_ml_classifier: bool = False,
        model_dir: Optional[Path] = None,
        auto_train_ml: bool = False,
    ) -> None:
        """
        Initialize psychological profiler

        Args:
            use_ml_classifier: Whether to use BERT+RF ML classifier
            model_dir: Directory for ML model persistence
            auto_train_ml: Automatically train ML model if not found
        """
        self._trait_indicators = self._load_trait_indicators()
        self.use_ml_classifier = use_ml_classifier
        self._ml_classifier: Optional["BERTDarkTriadClassifier"] = None

        # Initialize ML classifier if requested
        if use_ml_classifier:
            self._init_ml_classifier(model_dir, auto_train_ml)

    def _init_ml_classifier(self, model_dir: Optional[Path], auto_train: bool) -> None:
        """
        Initialize ML-based classifier

        Args:
            model_dir: Model directory path
            auto_train: Whether to auto-train if model not found
        """
        try:
            from voice_man.services.crime_classification.bert_dark_triad_classifier import (
                BERTDarkTriadClassifier,
            )

            self._ml_classifier = BERTDarkTriadClassifier(
                model_dir=model_dir,
                auto_load=True,
                auto_train=auto_train,
            )

            # Only use ML if successfully loaded/trained
            if self._ml_classifier.is_loaded:
                self.use_ml_classifier = True
            else:
                self.use_ml_classifier = False

        except Exception:
            # ML initialization failed - fall back to keyword-based
            self.use_ml_classifier = False
            self._ml_classifier = None

    def _load_trait_indicators(self) -> Dict[str, List[str]]:
        """
        Load psychological trait indicators

        Phase 1 Improvement: Expanded SD3 (Short Dark Triad) based keywords.
        Total: 30 keywords per trait (was 10), 90 total keywords.

        SD3 Reference:
        - Jones & Paulhus (2014) Short Dark Triad
        - 27 items, 9 per trait
        - Validated across multiple studies
        """
        return {
            "narcissism": [
                # Original keywords (10)
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
                # SD3 expansion: Grandiosity & Self-Admiration (20 additional)
                "특별해",
                "대단해",
                "인정해",
                "최고야",
                "천재야",
                "완벽해",
                "자만심",
                "우월감",
                "특권의식",
                "칭찬원해",
                "인정받아",
                "자기자랑",
                "자기과시",
                "리더십",
                "카리스마",
                "매력적",
                "인기있",
                "존경받아",
                "중요해",
            ],
            "machiavellianism": [
                # Original keywords (10)
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
                # SD3 expansion: Manipulation & Strategic Thinking (20 additional)
                "이용가치",
                "활용해",
                "이용먹",
                "수단방",
                "목적달성",
                "결과중시",
                "실리적",
                "냉정해",
                "계산적",
                "전략적",
                "처세수",
                "현명해",
                "교묘해",
                "술책",
                "작전",
                "유리해",
                "필살묵정",
                "원리주의",
                "효율적",
                "성공주의",
                "승부패",
                "목표달성",
            ],
            "psychopathy": [
                # Original keywords (10)
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
                # SD3 expansion: Callousness & Lack of Empathy (20 additional)
                "무감정",
                "냉혹해",
                "잔인해",
                "비인격적",
                "상관없",
                "신경쓰",
                "괜찮아",
                "별일",
                "아니야",
                "상관안",
                "자책없",
                "죄책",
                "책임회피",
                "후회없",
                "양심없",
                "공감부족",
                "불쌍",
                "무자비",
                "가학적",
                "임혹",
                "냉소",
                "무관심",
                "이기적",
                "자기중심",
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
        # Use ML classifier if available and enabled
        if self.use_ml_classifier and self._ml_classifier is not None:
            return self._ml_classifier.predict(
                text, return_probabilities=True, fallback_to_keywords=True
            )

        # Keyword-based analysis (fallback)
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
