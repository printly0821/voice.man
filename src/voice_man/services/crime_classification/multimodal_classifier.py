"""
Multimodal Fusion Classifier Service
SPEC-CRIME-CLASS-001 Phase 7: Multimodal Weighted Ensemble Classifier
"""

from typing import Dict, List

from voice_man.models.crime_classification.classification_result import (
    CrimeClassificationResult,
)
from voice_man.models.crime_classification.crime_types import (
    CrimeClassification,
    CrimeType,
    ModalityScore,
)
from voice_man.models.crime_classification.legal_requirements import (
    LegalEvidenceMapping,
)
from voice_man.models.crime_classification.psychological_profile import (
    PsychologicalProfile,
)
from voice_man.services.crime_classification.confidence_calculator import (
    ConfidenceCalculator,
)
from voice_man.services.crime_classification.extended_crime_patterns import (
    ExtendedCrimePatterns,
)
from voice_man.services.crime_classification.legal_evidence_mapper import (
    LegalEvidenceMapper,
)
from voice_man.services.crime_classification.psychological_profiler import (
    PsychologicalProfiler,
)


class MultimodalClassifier:
    """
    멀티모달 융합 분류기

    Weighted ensemble classification combining text, audio, and psychological scores
    """

    def __init__(self) -> None:
        """Initialize multimodal classifier"""
        self.patterns = ExtendedCrimePatterns()
        self.profiler = PsychologicalProfiler()
        self.confidence_calc = ConfidenceCalculator()
        self.legal_mapper = LegalEvidenceMapper()

    def classify(
        self,
        text: str,
        audio_features: Dict[str, float],
        speaker_id: str = "SPEAKER_00",
    ) -> CrimeClassificationResult:
        """
        Perform multimodal crime classification

        Args:
            text: Text transcription
            audio_features: Audio feature scores per crime type
            speaker_id: Speaker identifier

        Returns:
            CrimeClassificationResult object
        """
        # Get crime type weights
        weights = self.patterns.get_crime_weights()

        # Calculate text-based scores
        text_scores: Dict[CrimeType, float] = {}
        for crime_type in CrimeType:
            crime_name = crime_type.value
            text_scores[crime_type] = self.patterns.calculate_crime_score(crime_name, text)

        # Audio scores from input (already calculated per crime type)
        audio_scores: Dict[CrimeType, float] = {}
        for crime_type in CrimeType:
            crime_name = crime_type.value
            audio_scores[crime_type] = audio_features.get(crime_name, 0.0)

        # Calculate psychological scores
        psych_profile = self.profiler.create_profile(text)
        psych_propensity = psych_profile.crime_propensity
        psych_scores: Dict[CrimeType, float] = {}
        for crime_type in CrimeType:
            crime_name = crime_type.value
            psych_scores[crime_type] = psych_propensity.get(crime_name, 0.0)

        # Weighted ensemble classification
        classifications = []
        for crime_type in CrimeType:
            crime_name = crime_type.value
            w = weights.get(crime_name, {"text": 0.4, "audio": 0.35, "psych": 0.25})

            # Calculate weighted score
            weighted_score = (
                text_scores.get(crime_type, 0.0) * w["text"]
                + audio_scores.get(crime_type, 0.0) * w["audio"]
                + psych_scores.get(crime_type, 0.0) * w["psych"]
            )

            # Calculate confidence interval
            ci = self.confidence_calc.calculate_confidence_interval(
                text_score=text_scores.get(crime_type, 0.0),
                audio_score=audio_scores.get(crime_type, 0.0),
                psych_score=psych_scores.get(crime_type, 0.0),
                weights=w,
            )

            # Get legal reference
            legal_ref = self._get_legal_reference(crime_type)

            # Create classification
            classification = CrimeClassification(
                crime_type=crime_type,
                confidence=weighted_score,
                confidence_interval=ci,
                modality_scores=ModalityScore(
                    text_score=text_scores.get(crime_type, 0.0),
                    audio_score=audio_scores.get(crime_type, 0.0),
                    psychological_score=psych_scores.get(crime_type, 0.0),
                ),
                weighted_score=weighted_score,
                legal_reference=legal_ref,
                evidence_items=[],
                requires_review=weighted_score < 0.7,
            )
            classifications.append(classification)

        # Sort by confidence (descending)
        classifications.sort(key=lambda x: x.confidence, reverse=True)

        # Filter by threshold (confidence >= 0.3)
        classifications = [c for c in classifications if c.confidence >= 0.3]

        # Create legal mappings
        legal_mappings = []
        for classification in classifications:
            crime_name = classification.crime_type.value
            # Mock indicators and evidence (in real system, extract from analysis)
            mapping = self.legal_mapper.map_evidence(
                crime_type=crime_name, detected_indicators=[], evidence_items=[]
            )
            legal_mappings.append(mapping)

        # Create result
        from datetime import datetime

        result = CrimeClassificationResult(
            analysis_id=f"analysis-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            analyzed_at=datetime.now(),
            speaker_id=speaker_id,
            classifications=classifications,
            legal_mappings=legal_mappings,
            psychological_profile=psych_profile,
            summary=self._generate_summary(classifications),
            recommendations=self._generate_recommendations(classifications),
        )

        return result

    def _get_legal_reference(self, crime_type: CrimeType) -> str:
        """Get legal code reference for crime type"""
        legal_codes = {
            CrimeType.FRAUD: "형법 제347조",
            CrimeType.EXTORTION: "형법 제350조",
            CrimeType.COERCION: "형법 제324조",
            CrimeType.THREAT: "형법 제283조",
            CrimeType.INSULT: "형법 제311조",
            CrimeType.EMBEZZLEMENT: "형법 제355조 제1항",
            CrimeType.BREACH_OF_TRUST: "형법 제355조 제2항",
            CrimeType.GASLIGHTING: "형법 제283조(협박), 제311조(모욕)",
            CrimeType.DOMESTIC_VIOLENCE: "가정폭력범죄의 처벌 등에 관한 특례법",
            CrimeType.STALKING: "스토킹 범죄의 처벌 등에 관한 법률",
            CrimeType.TAX_EVASION: "조세범처벌법",
        }
        return legal_codes.get(crime_type, "미정")

    def _generate_summary(self, classifications: List[CrimeClassification]) -> str:
        """Generate analysis summary"""
        if not classifications:
            return "분류된 범죄 유형이 없습니다."

        top_crime = classifications[0]
        summary = f"가장 유력한 범죄 유형: {top_crime.crime_type.value} (신뢰도: {top_crime.confidence:.2f})\n"
        summary += f"텍스트 점수: {top_crime.modality_scores.text_score:.2f}, "
        summary += f"음성 점수: {top_crime.modality_scores.audio_score:.2f}, "
        summary += f"심리 점수: {top_crime.modality_scores.psychological_score:.2f}"

        return summary

    def _generate_recommendations(self, classifications: List[CrimeClassification]) -> List[str]:
        """Generate recommendations based on classification results"""
        recommendations = []

        for classification in classifications:
            if classification.requires_review:
                recommendations.append(
                    f"{classification.crime_type.value}: 전문가 검토 권장 (신뢰도 {classification.confidence:.2f})"
                )

        if not recommendations:
            recommendations.append("분류 결과가 신뢰할 수준입니다.")

        recommendations.append("법률 전문가 검토 필수")
        recommendations.append("본 결과는 참고 자료로만 활용 가능")

        return recommendations
