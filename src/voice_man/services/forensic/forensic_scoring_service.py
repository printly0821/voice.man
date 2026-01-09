"""
Forensic Scoring Service
SPEC-FORENSIC-001 Phase 2-D: Integrated forensic scoring system

This service combines all forensic analysis results (audio features, crime language,
emotion recognition, cross-validation) into a comprehensive risk assessment.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np

from voice_man.models.forensic.audio_features import StressFeatures
from voice_man.models.forensic.crime_language import (
    CrimeLanguageScore,
    GaslightingMatch,
    ThreatMatch,
    CoercionMatch,
)
from voice_man.models.forensic.cross_validation import CrossValidationResult
from voice_man.models.forensic.emotion_recognition import MultiModelEmotionResult
from voice_man.models.forensic.forensic_score import (
    CategoryScore,
    DeceptionAnalysis,
    ForensicCategory,
    ForensicEvidence,
    ForensicScoreResult,
    GaslightingAnalysis,
    RiskLevel,
    ThreatAssessment,
)


class ForensicScoringService:
    """
    Forensic comprehensive scoring service.

    Integrates all analysis results to provide a forensic assessment
    from a legal/investigative perspective.

    Scoring Weights:
        - Gaslighting: 25%
        - Threat: 25%
        - Coercion: 20%
        - Deception: 20%
        - Emotional Manipulation: 10%

    Risk Level Mapping:
        - 0-20: MINIMAL
        - 21-40: LOW
        - 41-60: MODERATE
        - 61-80: HIGH
        - 81-100: CRITICAL
    """

    # Category weights for overall score calculation
    CATEGORY_WEIGHTS = {
        ForensicCategory.GASLIGHTING: 0.25,
        ForensicCategory.THREAT: 0.25,
        ForensicCategory.COERCION: 0.20,
        ForensicCategory.DECEPTION: 0.20,
        ForensicCategory.EMOTIONAL_MANIPULATION: 0.10,
    }

    # Deception score weights
    DECEPTION_WEIGHTS = {
        "voice_text_inconsistency": 0.35,
        "deception_markers": 0.30,
        "emotional_inconsistency": 0.25,
        "stress_patterns": 0.10,
    }

    # Gaslighting score weights
    GASLIGHTING_WEIGHTS = {
        "pattern_detection": 0.40,
        "emotional_manipulation": 0.30,
        "recurring_patterns": 0.20,
        "cross_validation_inconsistency": 0.10,
    }

    def __init__(
        self,
        audio_feature_service,
        stress_analysis_service,
        crime_language_service,
        ser_service,
        cross_validation_service,
    ):
        """
        Initialize the ForensicScoringService.

        Args:
            audio_feature_service: Service for audio feature extraction.
            stress_analysis_service: Service for stress analysis.
            crime_language_service: Service for crime language pattern detection.
            ser_service: Service for speech emotion recognition.
            cross_validation_service: Service for text-voice cross-validation.
        """
        self._audio_feature_service = audio_feature_service
        self._stress_analysis_service = stress_analysis_service
        self._crime_language_service = crime_language_service
        self._ser_service = ser_service
        self._cross_validation_service = cross_validation_service

    def _map_score_to_risk_level(self, score: float) -> RiskLevel:
        """
        Map a numeric score (0-100) to a RiskLevel enum.

        Args:
            score: Numeric score from 0 to 100.

        Returns:
            Corresponding RiskLevel.
        """
        if score <= 20:
            return RiskLevel.MINIMAL
        elif score <= 40:
            return RiskLevel.LOW
        elif score <= 60:
            return RiskLevel.MODERATE
        elif score <= 80:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL

    def calculate_gaslighting_score(
        self,
        crime_analysis: CrimeLanguageScore,
        cross_validation: CrossValidationResult,
    ) -> GaslightingAnalysis:
        """
        Calculate gaslighting intensity score.

        Weights:
            - Gaslighting pattern detection: 40%
            - Emotional manipulation indicators: 30%
            - Recurring patterns: 20%
            - Cross-validation inconsistency: 10%

        Args:
            crime_analysis: Crime language analysis result.
            cross_validation: Cross-validation result.

        Returns:
            GaslightingAnalysis with intensity score and details.
        """
        # Pattern detection contribution (40%)
        pattern_score = (
            crime_analysis.gaslighting_score * 100 * self.GASLIGHTING_WEIGHTS["pattern_detection"]
        )

        # Emotional manipulation contribution (30%)
        # Higher coercion and lower consistency indicate manipulation
        emotional_score = (
            (
                crime_analysis.coercion_score * 0.5
                + (1 - cross_validation.overall_consistency_score) * 0.5
            )
            * 100
            * self.GASLIGHTING_WEIGHTS["emotional_manipulation"]
        )

        # Recurring patterns contribution (20%)
        pattern_count = crime_analysis.gaslighting_count
        recurring_score = (
            min(pattern_count / 5.0, 1.0) * 100 * self.GASLIGHTING_WEIGHTS["recurring_patterns"]
        )

        # Cross-validation inconsistency contribution (10%)
        inconsistency_score = (
            (1 - cross_validation.overall_consistency_score)
            * 100
            * self.GASLIGHTING_WEIGHTS["cross_validation_inconsistency"]
        )

        # Total intensity score
        intensity_score = pattern_score + emotional_score + recurring_score + inconsistency_score
        intensity_score = min(100.0, max(0.0, intensity_score))

        # Determine victim impact level
        victim_impact = self._map_score_to_risk_level(intensity_score)

        # Extract patterns detected
        patterns_detected = []
        if crime_analysis.gaslighting_score > 0.3:
            patterns_detected.append("denial")
        if crime_analysis.gaslighting_score > 0.5:
            patterns_detected.append("countering")
        if crime_analysis.gaslighting_score > 0.7:
            patterns_detected.append("trivializing")
            patterns_detected.append("blame_shifting")

        # Manipulation techniques
        manipulation_techniques = []
        if crime_analysis.gaslighting_score > 0.4:
            manipulation_techniques.append("reality distortion")
        if crime_analysis.coercion_score > 0.3:
            manipulation_techniques.append("emotional coercion")

        return GaslightingAnalysis(
            intensity_score=intensity_score,
            patterns_detected=patterns_detected,
            manipulation_techniques=manipulation_techniques,
            victim_impact_level=victim_impact,
            recurring_phrases=[],  # Would require text analysis for specific phrases
        )

    def calculate_deception_score(
        self,
        cross_validation: CrossValidationResult,
        emotion_result: MultiModelEmotionResult,
        crime_analysis: CrimeLanguageScore,
    ) -> DeceptionAnalysis:
        """
        Calculate deception probability score.

        Weights:
            - Text-voice inconsistency: 35%
            - Deception language markers: 30%
            - Emotional inconsistency: 25%
            - Stress patterns: 10%

        Args:
            cross_validation: Cross-validation result.
            emotion_result: Emotion recognition result.
            crime_analysis: Crime language analysis result.

        Returns:
            DeceptionAnalysis with deception probability and details.
        """
        # Voice-text consistency (inverted for deception)
        voice_text_consistency = cross_validation.overall_consistency_score

        # Text-voice inconsistency contribution (35%)
        inconsistency_contribution = (1 - voice_text_consistency) * self.DECEPTION_WEIGHTS[
            "voice_text_inconsistency"
        ]

        # Deception markers contribution (30%)
        deception_markers_contribution = (
            crime_analysis.deception_score * self.DECEPTION_WEIGHTS["deception_markers"]
        )

        # Emotional inconsistency contribution (25%)
        # Compare text sentiment with voice emotion
        text_sentiment = cross_validation.text_analysis.sentiment_score
        voice_valence = cross_validation.voice_analysis.valence

        # Convert sentiment (-1 to 1) to 0-1 scale
        normalized_sentiment = (text_sentiment + 1) / 2

        # Calculate emotional inconsistency
        emotional_diff = abs(normalized_sentiment - voice_valence)
        emotional_inconsistency = emotional_diff * self.DECEPTION_WEIGHTS["emotional_inconsistency"]

        # Stress patterns contribution (10%)
        stress_contribution = (
            cross_validation.voice_analysis.stress_level * self.DECEPTION_WEIGHTS["stress_patterns"]
        )

        # Total deception probability
        deception_probability = (
            inconsistency_contribution
            + deception_markers_contribution
            + emotional_inconsistency
            + stress_contribution
        )
        deception_probability = min(1.0, max(0.0, deception_probability))

        # Emotional authenticity (inverse of emotional inconsistency)
        emotional_authenticity = 1.0 - emotional_diff
        emotional_authenticity = min(1.0, max(0.0, emotional_authenticity))

        # Count linguistic markers (approximation based on deception score)
        linguistic_markers_count = int(crime_analysis.deception_score * 10)

        # Behavioral indicators
        behavioral_indicators = []
        if voice_text_consistency < 0.5:
            behavioral_indicators.append("voice-text inconsistency")
        if crime_analysis.deception_score > 0.3:
            behavioral_indicators.append("hedging language")
        if crime_analysis.deception_score > 0.5:
            behavioral_indicators.append("distancing language")
        if emotional_diff > 0.3:
            behavioral_indicators.append("emotional incongruence")
        if cross_validation.voice_analysis.stress_level > 0.6:
            behavioral_indicators.append("elevated stress")

        return DeceptionAnalysis(
            deception_probability=deception_probability,
            voice_text_consistency=voice_text_consistency,
            emotional_authenticity=emotional_authenticity,
            linguistic_markers_count=linguistic_markers_count,
            behavioral_indicators=behavioral_indicators,
        )

    def assess_threat_level(
        self,
        crime_analysis: CrimeLanguageScore,
        stress_result: StressFeatures,
    ) -> ThreatAssessment:
        """
        Assess threat level from crime language and stress analysis.

        Assessment criteria:
            - Direct threat expressions
            - Conditional threats
            - Implicit/veiled threats
            - Stress correlation

        Args:
            crime_analysis: Crime language analysis result.
            stress_result: Stress analysis result.

        Returns:
            ThreatAssessment with threat level and details.
        """
        # Calculate threat score (0-100)
        base_threat_score = crime_analysis.threat_score * 100

        # Stress amplification (high stress with threats increases credibility)
        stress_factor = 1.0
        if stress_result.risk_level == "high":
            stress_factor = 1.2
        elif stress_result.risk_level == "medium":
            stress_factor = 1.1

        threat_score = min(100.0, base_threat_score * stress_factor)

        # Determine threat level
        threat_level = self._map_score_to_risk_level(threat_score)

        # Determine threat types
        threat_types = []
        if crime_analysis.threat_score > 0.7:
            threat_types.append("direct")
        if crime_analysis.threat_score > 0.4:
            threat_types.append("conditional")
        if crime_analysis.threat_score > 0.2:
            threat_types.append("veiled")
        if crime_analysis.coercion_score > 0.5:
            threat_types.append("coercive")

        # Determine immediacy based on threat score and stress
        if crime_analysis.threat_score > 0.8 and stress_result.risk_level == "high":
            immediacy = "immediate"
        elif crime_analysis.threat_score > 0.5:
            immediacy = "near-term"
        else:
            immediacy = "long-term"

        # Determine specificity
        if crime_analysis.threat_count >= 3:
            specificity = "detailed"
        elif crime_analysis.threat_count >= 1:
            specificity = "specific"
        else:
            specificity = "vague"

        # Calculate credibility based on consistency of threat patterns
        credibility_score = min(
            1.0, crime_analysis.threat_score + 0.1 * crime_analysis.threat_count
        )
        if stress_result.risk_level == "high":
            credibility_score = min(1.0, credibility_score + 0.1)

        return ThreatAssessment(
            threat_level=threat_level,
            threat_types=threat_types,
            immediacy=immediacy,
            specificity=specificity,
            credibility_score=credibility_score,
        )

    def calculate_overall_score(
        self,
        category_scores: List[CategoryScore],
    ) -> Tuple[float, RiskLevel]:
        """
        Calculate overall risk score from category scores.

        Weighted average:
            - Gaslighting: 25%
            - Threat: 25%
            - Coercion: 20%
            - Deception: 20%
            - Emotional Manipulation: 10%

        Args:
            category_scores: List of category scores.

        Returns:
            Tuple of (overall_score, risk_level).
        """
        if not category_scores:
            return 0.0, RiskLevel.MINIMAL

        # Build a mapping of category to score
        score_map = {cs.category: cs.score for cs in category_scores}

        # Calculate weighted sum
        total_weight = 0.0
        weighted_sum = 0.0

        for category, weight in self.CATEGORY_WEIGHTS.items():
            if category in score_map:
                weighted_sum += score_map[category] * weight
                total_weight += weight

        # Normalize if not all categories present
        if total_weight > 0 and total_weight < 1.0:
            # Redistribute weights proportionally
            weighted_sum = weighted_sum / total_weight

        overall_score = min(100.0, max(0.0, weighted_sum))
        risk_level = self._map_score_to_risk_level(overall_score)

        return overall_score, risk_level

    def generate_evidence_items(
        self,
        all_results: Dict[str, Any],
    ) -> List[ForensicEvidence]:
        """
        Generate forensic evidence items from analysis results.

        Args:
            all_results: Dictionary containing all analysis results.

        Returns:
            List of ForensicEvidence items.
        """
        evidence_items = []

        # Process gaslighting matches
        gaslighting_matches = all_results.get("gaslighting_matches", [])
        for match in gaslighting_matches:
            if isinstance(match, GaslightingMatch):
                evidence_items.append(
                    ForensicEvidence(
                        timestamp=0.0,  # Would need audio timestamp
                        evidence_type="gaslighting_pattern",
                        description=f"Gaslighting pattern detected: {match.type}",
                        severity=self._map_score_to_risk_level(match.severity_weight * 100),
                        supporting_data={
                            "pattern": match.matched_pattern,
                            "confidence": match.confidence,
                            "type": str(match.type),
                        },
                    )
                )

        # Process threat matches
        threat_matches = all_results.get("threat_matches", [])
        for match in threat_matches:
            if isinstance(match, ThreatMatch):
                evidence_items.append(
                    ForensicEvidence(
                        timestamp=0.0,
                        evidence_type="threat_expression",
                        description=f"Threat detected: {match.type}",
                        severity=self._map_score_to_risk_level(match.severity_weight * 100),
                        supporting_data={
                            "pattern": match.matched_pattern,
                            "confidence": match.confidence,
                            "type": str(match.type),
                        },
                    )
                )

        # Process coercion matches
        coercion_matches = all_results.get("coercion_matches", [])
        for match in coercion_matches:
            if isinstance(match, CoercionMatch):
                evidence_items.append(
                    ForensicEvidence(
                        timestamp=0.0,
                        evidence_type="coercion_pattern",
                        description=f"Coercion pattern detected: {match.type}",
                        severity=self._map_score_to_risk_level(match.severity_weight * 100),
                        supporting_data={
                            "pattern": match.matched_pattern,
                            "confidence": match.confidence,
                            "type": str(match.type),
                        },
                    )
                )

        return evidence_items

    def generate_summary_and_recommendations(
        self,
        score_result,
    ) -> Tuple[str, List[str]]:
        """
        Generate human-readable summary and recommendations.

        Args:
            score_result: The forensic score result (or mock with required attributes).

        Returns:
            Tuple of (summary, recommendations).
        """
        risk_level = score_result.overall_risk_level
        risk_score = score_result.overall_risk_score
        category_scores = score_result.category_scores

        # Generate summary based on risk level
        if risk_level == RiskLevel.CRITICAL:
            summary = f"심각한 위험 수준 감지됨 (점수: {risk_score:.1f}/100). 즉각적인 전문가 개입이 필요합니다."
        elif risk_level == RiskLevel.HIGH:
            summary = (
                f"높은 위험 수준 감지됨 (점수: {risk_score:.1f}/100). 전문가 상담을 권장합니다."
            )
        elif risk_level == RiskLevel.MODERATE:
            summary = (
                f"중간 위험 수준 감지됨 (점수: {risk_score:.1f}/100). 주의 깊은 관찰이 필요합니다."
            )
        elif risk_level == RiskLevel.LOW:
            summary = (
                f"낮은 위험 수준 (점수: {risk_score:.1f}/100). 일부 주의 패턴이 감지되었습니다."
            )
        else:
            summary = f"최소 위험 수준 (점수: {risk_score:.1f}/100). 특별한 위험 패턴이 감지되지 않았습니다."

        # Add category-specific details to summary
        high_risk_categories = [cs for cs in category_scores if cs.score >= 60.0]
        if high_risk_categories:
            category_names = [str(cs.category) for cs in high_risk_categories]
            summary += f" 주요 위험 영역: {', '.join(category_names)}"

        # Generate recommendations
        recommendations = []

        if risk_level in [RiskLevel.CRITICAL, RiskLevel.HIGH]:
            recommendations.append("전문 상담사 또는 법률 전문가와 상담하시기 바랍니다")
            recommendations.append("모든 대화 기록을 안전하게 보관하세요")
            recommendations.append("신뢰할 수 있는 가족이나 친구에게 상황을 공유하세요")

        if risk_level == RiskLevel.CRITICAL:
            recommendations.append("즉시 안전한 장소로 이동하는 것을 고려하세요")
            recommendations.append("필요시 관련 기관(경찰, 상담센터)에 연락하세요")

        # Category-specific recommendations
        for cs in category_scores:
            if cs.category == ForensicCategory.GASLIGHTING and cs.score >= 50:
                recommendations.append("가스라이팅 패턴이 감지되었습니다. 심리 상담을 고려해보세요")
            if cs.category == ForensicCategory.THREAT and cs.score >= 50:
                recommendations.append("위협 패턴이 감지되었습니다. 법적 조언을 구하시기 바랍니다")
            if cs.category == ForensicCategory.COERCION and cs.score >= 50:
                recommendations.append("강압 패턴이 감지되었습니다. 지원 네트워크를 활용하세요")

        if not recommendations:
            recommendations.append("현재 특별한 조치가 필요하지 않습니다")
            recommendations.append("정기적인 자기 점검을 권장합니다")

        return summary, recommendations

    async def analyze(
        self,
        audio_path: str,
        transcript: str,
    ) -> ForensicScoreResult:
        """
        Perform complete forensic analysis.

        Analysis pipeline:
            1. Audio feature analysis (Phase 1)
            2. Crime language analysis (Phase 2-A)
            3. SER emotion analysis (Phase 2-B)
            4. Cross-validation (Phase 2-C)
            5. Comprehensive scoring (Phase 2-D)

        Args:
            audio_path: Path to the audio file.
            transcript: Text transcript of the audio.

        Returns:
            ForensicScoreResult with complete analysis.
        """

        # Generate analysis ID
        analysis_id = str(uuid.uuid4())[:8]
        analyzed_at = datetime.now()

        # Load audio
        audio, sr = librosa.load(audio_path, sr=None)
        audio_duration = len(audio) / sr

        # Phase 1: Audio feature analysis
        audio_features = self._audio_feature_service.analyze_audio_features(audio, sr, audio_path)

        # Phase 1: Stress analysis
        stress_result = self._stress_analysis_service.analyze_stress(
            audio, sr, jitter_percent=audio_features.pitch_features.jitter_percent
        )

        # Phase 2-A: Crime language analysis
        crime_analysis = self._crime_language_service.analyze_comprehensive(transcript)
        gaslighting_matches = self._crime_language_service.detect_gaslighting(transcript)
        threat_matches = self._crime_language_service.detect_threats(transcript)
        coercion_matches = self._crime_language_service.detect_coercion(transcript)

        # Phase 2-B: SER emotion analysis
        emotion_result = self._ser_service.analyze_ensemble(audio, sr)
        forensic_indicators = self._ser_service.get_forensic_emotion_indicators(emotion_result)

        # Phase 2-C: Cross-validation
        cross_validation = self._cross_validation_service.cross_validate(transcript, audio_path)

        # Phase 2-D: Calculate individual scores
        gaslighting_analysis = self.calculate_gaslighting_score(crime_analysis, cross_validation)
        deception_analysis = self.calculate_deception_score(
            cross_validation, emotion_result, crime_analysis
        )
        threat_assessment = self.assess_threat_level(crime_analysis, stress_result)

        # Build category scores
        category_scores = [
            CategoryScore(
                category=ForensicCategory.GASLIGHTING,
                score=gaslighting_analysis.intensity_score,
                confidence=min(1.0, crime_analysis.gaslighting_score + 0.3),
                evidence_count=crime_analysis.gaslighting_count,
                key_indicators=gaslighting_analysis.patterns_detected,
            ),
            CategoryScore(
                category=ForensicCategory.THREAT,
                score=crime_analysis.threat_score * 100,
                confidence=min(1.0, crime_analysis.threat_score + 0.3),
                evidence_count=crime_analysis.threat_count,
                key_indicators=threat_assessment.threat_types,
            ),
            CategoryScore(
                category=ForensicCategory.COERCION,
                score=crime_analysis.coercion_score * 100,
                confidence=min(1.0, crime_analysis.coercion_score + 0.3),
                evidence_count=crime_analysis.coercion_count,
                key_indicators=[],
            ),
            CategoryScore(
                category=ForensicCategory.DECEPTION,
                score=deception_analysis.deception_probability * 100,
                confidence=cross_validation.overall_consistency_score,
                evidence_count=deception_analysis.linguistic_markers_count,
                key_indicators=deception_analysis.behavioral_indicators,
            ),
            CategoryScore(
                category=ForensicCategory.EMOTIONAL_MANIPULATION,
                score=(crime_analysis.coercion_score + crime_analysis.gaslighting_score) * 50,
                confidence=0.7,
                evidence_count=0,
                key_indicators=[],
            ),
        ]

        # Calculate overall score
        overall_score, risk_level = self.calculate_overall_score(category_scores)

        # Calculate confidence
        avg_confidence = sum(cs.confidence for cs in category_scores) / len(category_scores)
        evidence_factor = min(1.0, sum(cs.evidence_count for cs in category_scores) / 10)
        confidence_level = (
            avg_confidence * 0.4
            + evidence_factor * 0.3
            + cross_validation.overall_consistency_score * 0.3
        )
        confidence_level = min(1.0, max(0.0, confidence_level))

        # Generate evidence items
        all_results = {
            "gaslighting_matches": gaslighting_matches,
            "threat_matches": threat_matches,
            "coercion_matches": coercion_matches,
            "audio_duration": audio_duration,
        }
        evidence_items = self.generate_evidence_items(all_results)

        # Generate summary and recommendations
        temp_result = type(
            "TempResult",
            (),
            {
                "overall_risk_level": risk_level,
                "overall_risk_score": overall_score,
                "category_scores": category_scores,
            },
        )()
        summary, recommendations = self.generate_summary_and_recommendations(temp_result)

        # Generate flags
        flags = []
        if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            flags.append("HIGH_RISK_DETECTED")
        if gaslighting_analysis.intensity_score >= 60:
            flags.append("GASLIGHTING_PATTERN")
        if threat_assessment.threat_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            flags.append("THREAT_DETECTED")
        if deception_analysis.deception_probability >= 0.6:
            flags.append("DECEPTION_INDICATOR")
        if forensic_indicators.stress_indicator:
            flags.append("HIGH_STRESS")

        return ForensicScoreResult(
            analysis_id=analysis_id,
            analyzed_at=analyzed_at,
            audio_duration_seconds=audio_duration,
            overall_risk_score=overall_score,
            overall_risk_level=risk_level,
            confidence_level=confidence_level,
            category_scores=category_scores,
            deception_analysis=deception_analysis,
            gaslighting_analysis=gaslighting_analysis,
            threat_assessment=threat_assessment,
            evidence_items=evidence_items,
            summary=summary,
            recommendations=recommendations,
            flags=flags,
        )
