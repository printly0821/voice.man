"""
범죄 언어 분석 서비스
SPEC-FORENSIC-001 Phase 2-A: Crime Language Analysis Service

가스라이팅, 협박, 강압, 기만 언어 패턴을 감지하고 분석하는 서비스
"""

from typing import Dict, List, Optional, Any

from voice_man.models.forensic.crime_language import (
    GaslightingType,
    GaslightingMatch,
    ThreatType,
    ThreatMatch,
    CoercionType,
    CoercionMatch,
    DeceptionCategory,
    DeceptionMarkerMatch,
    DeceptionAnalysis,
    CrimeLanguageScore,
)
from voice_man.services.forensic.crime_language_pattern_db import CrimeLanguagePatternDB


class CrimeLanguageAnalysisService:
    """
    범죄 언어 분석 서비스

    텍스트에서 가스라이팅, 협박, 강압, 기만 언어 패턴을 감지하고 분석
    """

    def __init__(self) -> None:
        """서비스 초기화"""
        self._pattern_db = CrimeLanguagePatternDB()

    # ==========================================================================
    # Gaslighting Detection
    # ==========================================================================

    def detect_gaslighting(
        self,
        text: str,
        speaker: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> List[GaslightingMatch]:
        """
        텍스트에서 가스라이팅 패턴 감지

        Args:
            text: 분석할 텍스트
            speaker: 화자 ID (선택)
            timestamp: 발화 시간 (선택)

        Returns:
            감지된 가스라이팅 매칭 결과 목록
        """
        if not text or not text.strip():
            return []

        matches: List[GaslightingMatch] = []

        for pattern in self._pattern_db.get_gaslighting_patterns():
            for pattern_str in pattern.patterns_ko:
                if pattern_str in text:
                    start_pos = text.find(pattern_str)
                    end_pos = start_pos + len(pattern_str)

                    match = GaslightingMatch(
                        type=pattern.type,
                        matched_pattern=pattern_str,
                        text=text,
                        start_position=start_pos,
                        end_position=end_pos,
                        confidence=self._calculate_match_confidence(text, pattern_str),
                        severity_weight=pattern.severity_weight,
                        speaker=speaker,
                        timestamp=timestamp,
                    )
                    matches.append(match)

        return matches

    # ==========================================================================
    # Threat Detection
    # ==========================================================================

    def detect_threats(
        self,
        text: str,
        speaker: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> List[ThreatMatch]:
        """
        텍스트에서 협박 패턴 감지

        Args:
            text: 분석할 텍스트
            speaker: 화자 ID (선택)
            timestamp: 발화 시간 (선택)

        Returns:
            감지된 협박 매칭 결과 목록
        """
        if not text or not text.strip():
            return []

        matches: List[ThreatMatch] = []

        for pattern in self._pattern_db.get_threat_patterns():
            for pattern_str in pattern.patterns_ko:
                if pattern_str in text:
                    start_pos = text.find(pattern_str)
                    end_pos = start_pos + len(pattern_str)

                    match = ThreatMatch(
                        type=pattern.type,
                        matched_pattern=pattern_str,
                        text=text,
                        start_position=start_pos,
                        end_position=end_pos,
                        confidence=self._calculate_match_confidence(text, pattern_str),
                        severity_weight=pattern.severity_weight,
                        speaker=speaker,
                        timestamp=timestamp,
                    )
                    matches.append(match)

        return matches

    # ==========================================================================
    # Coercion Detection
    # ==========================================================================

    def detect_coercion(
        self,
        text: str,
        speaker: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> List[CoercionMatch]:
        """
        텍스트에서 강압 패턴 감지

        Args:
            text: 분석할 텍스트
            speaker: 화자 ID (선택)
            timestamp: 발화 시간 (선택)

        Returns:
            감지된 강압 매칭 결과 목록
        """
        if not text or not text.strip():
            return []

        matches: List[CoercionMatch] = []

        for pattern in self._pattern_db.get_coercion_patterns():
            for pattern_str in pattern.patterns_ko:
                if pattern_str in text:
                    start_pos = text.find(pattern_str)
                    end_pos = start_pos + len(pattern_str)

                    match = CoercionMatch(
                        type=pattern.type,
                        matched_pattern=pattern_str,
                        text=text,
                        start_position=start_pos,
                        end_position=end_pos,
                        confidence=self._calculate_match_confidence(text, pattern_str),
                        severity_weight=pattern.severity_weight,
                        speaker=speaker,
                        timestamp=timestamp,
                    )
                    matches.append(match)

        return matches

    # ==========================================================================
    # Deception Analysis
    # ==========================================================================

    def analyze_deception(self, text: str) -> DeceptionAnalysis:
        """
        텍스트에서 기만 언어 지표 분석

        Args:
            text: 분석할 텍스트

        Returns:
            기만 분석 결과
        """
        if not text or not text.strip():
            return DeceptionAnalysis(
                marker_matches=[],
                hedging_score=0.0,
                distancing_score=0.0,
                negative_emotion_score=0.0,
                exclusive_score=0.0,
                cognitive_complexity_score=0.0,
                overall_deception_score=0.0,
            )

        marker_matches: List[DeceptionMarkerMatch] = []
        category_counts: Dict[DeceptionCategory, int] = {
            category: 0 for category in DeceptionCategory
        }

        # Count markers by category
        for marker in self._pattern_db.get_deception_markers():
            for marker_str in marker.markers_ko:
                count = text.count(marker_str)
                if count > 0:
                    category_counts[marker.category] += count
                    marker_matches.append(
                        DeceptionMarkerMatch(
                            category=marker.category,
                            marker=marker_str,
                            count=count,
                        )
                    )

        # Calculate scores (normalized by word count)
        word_count = max(len(text.split()), 1)

        hedging_score = min(category_counts[DeceptionCategory.HEDGING] / word_count * 5, 1.0)
        distancing_score = min(category_counts[DeceptionCategory.DISTANCING] / word_count * 5, 1.0)
        negative_emotion_score = min(
            category_counts[DeceptionCategory.NEGATIVE_EMOTION] / word_count * 5, 1.0
        )
        exclusive_score = min(category_counts[DeceptionCategory.EXCLUSIVE] / word_count * 5, 1.0)
        cognitive_complexity_score = min(
            category_counts[DeceptionCategory.COGNITIVE_COMPLEXITY] / word_count * 5,
            1.0,
        )

        # Calculate overall deception score
        # Weighted average: hedging and distancing increase deception
        # Exclusive and cognitive complexity decrease deception when present
        overall_deception_score = (
            hedging_score * 0.3
            + distancing_score * 0.3
            + negative_emotion_score * 0.2
            + (1 - exclusive_score) * 0.1
            + (1 - cognitive_complexity_score) * 0.1
        )

        return DeceptionAnalysis(
            marker_matches=marker_matches,
            hedging_score=hedging_score,
            distancing_score=distancing_score,
            negative_emotion_score=negative_emotion_score,
            exclusive_score=exclusive_score,
            cognitive_complexity_score=cognitive_complexity_score,
            overall_deception_score=overall_deception_score,
        )

    # ==========================================================================
    # Comprehensive Analysis
    # ==========================================================================

    def analyze_comprehensive(self, text: str) -> CrimeLanguageScore:
        """
        텍스트 종합 분석

        Args:
            text: 분석할 텍스트

        Returns:
            범죄 언어 종합 점수
        """
        if not text or not text.strip():
            return CrimeLanguageScore(
                gaslighting_score=0.0,
                threat_score=0.0,
                coercion_score=0.0,
                deception_score=0.0,
                overall_risk_score=0.0,
                risk_level="낮음",
                gaslighting_count=0,
                threat_count=0,
                coercion_count=0,
            )

        # Detect patterns
        gaslighting_matches = self.detect_gaslighting(text)
        threat_matches = self.detect_threats(text)
        coercion_matches = self.detect_coercion(text)
        deception_analysis = self.analyze_deception(text)

        # Calculate scores
        gaslighting_score = self._calculate_pattern_score(gaslighting_matches)
        threat_score = self._calculate_pattern_score(threat_matches)
        coercion_score = self._calculate_pattern_score(coercion_matches)
        deception_score = deception_analysis.overall_deception_score

        # Calculate overall risk score
        overall_risk_score = (
            gaslighting_score * 0.3
            + threat_score * 0.35
            + coercion_score * 0.25
            + deception_score * 0.1
        )

        risk_level = self._calculate_risk_level(overall_risk_score)

        return CrimeLanguageScore(
            gaslighting_score=gaslighting_score,
            threat_score=threat_score,
            coercion_score=coercion_score,
            deception_score=deception_score,
            overall_risk_score=overall_risk_score,
            risk_level=risk_level,
            gaslighting_count=len(gaslighting_matches),
            threat_count=len(threat_matches),
            coercion_count=len(coercion_matches),
        )

    def _calculate_pattern_score(
        self, matches: List[GaslightingMatch | ThreatMatch | CoercionMatch]
    ) -> float:
        """패턴 매칭 기반 점수 계산"""
        if not matches:
            return 0.0

        # Weight by severity and count
        total_weight = sum(m.severity_weight * m.confidence for m in matches)
        # Normalize to 0-1 range (cap at 5 matches for max score)
        return min(total_weight / 5.0, 1.0)

    def _calculate_risk_level(self, score: float) -> str:
        """위험 수준 계산"""
        if score >= 0.75:
            return "매우 높음"
        elif score >= 0.5:
            return "높음"
        elif score >= 0.25:
            return "중간"
        else:
            return "낮음"

    def _calculate_match_confidence(self, text: str, pattern: str) -> float:
        """매칭 신뢰도 계산"""
        # Base confidence
        base_confidence = 0.8

        # Exact match bonus
        if text.strip() == pattern:
            return 0.95

        # Longer patterns have higher confidence
        pattern_len = len(pattern)
        if pattern_len >= 8:
            base_confidence += 0.1
        elif pattern_len >= 5:
            base_confidence += 0.05

        return min(base_confidence, 1.0)

    # ==========================================================================
    # Conversation Analysis
    # ==========================================================================

    def analyze_conversation(self, conversation: List[Dict[str, str]]) -> CrimeLanguageScore:
        """
        대화 전체 분석

        Args:
            conversation: 대화 목록 [{"speaker": str, "text": str, "time": str}, ...]

        Returns:
            범죄 언어 종합 점수
        """
        if not conversation:
            return CrimeLanguageScore(
                gaslighting_score=0.0,
                threat_score=0.0,
                coercion_score=0.0,
                deception_score=0.0,
                overall_risk_score=0.0,
                risk_level="낮음",
                gaslighting_count=0,
                threat_count=0,
                coercion_count=0,
            )

        all_gaslighting = []
        all_threats = []
        all_coercion = []

        for utterance in conversation:
            text = utterance.get("text", "")
            speaker = utterance.get("speaker")
            time = utterance.get("time")

            all_gaslighting.extend(self.detect_gaslighting(text, speaker, time))
            all_threats.extend(self.detect_threats(text, speaker, time))
            all_coercion.extend(self.detect_coercion(text, speaker, time))

        # Calculate scores
        gaslighting_score = self._calculate_pattern_score(all_gaslighting)
        threat_score = self._calculate_pattern_score(all_threats)
        coercion_score = self._calculate_pattern_score(all_coercion)

        # Combine all text for deception analysis
        all_text = " ".join(u.get("text", "") for u in conversation)
        deception_analysis = self.analyze_deception(all_text)

        overall_risk_score = (
            gaslighting_score * 0.3
            + threat_score * 0.35
            + coercion_score * 0.25
            + deception_analysis.overall_deception_score * 0.1
        )

        return CrimeLanguageScore(
            gaslighting_score=gaslighting_score,
            threat_score=threat_score,
            coercion_score=coercion_score,
            deception_score=deception_analysis.overall_deception_score,
            overall_risk_score=overall_risk_score,
            risk_level=self._calculate_risk_level(overall_risk_score),
            gaslighting_count=len(all_gaslighting),
            threat_count=len(all_threats),
            coercion_count=len(all_coercion),
        )

    def analyze_conversation_by_speaker(
        self, conversation: List[Dict[str, str]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        화자별 대화 분석

        Args:
            conversation: 대화 목록

        Returns:
            화자별 분석 결과 딕셔너리
        """
        speaker_results: Dict[str, Dict[str, Any]] = {}

        for utterance in conversation:
            speaker = utterance.get("speaker", "Unknown")
            text = utterance.get("text", "")
            time = utterance.get("time")

            if speaker not in speaker_results:
                speaker_results[speaker] = {
                    "gaslighting_count": 0,
                    "threat_count": 0,
                    "coercion_count": 0,
                    "total_utterances": 0,
                }

            speaker_results[speaker]["total_utterances"] += 1
            speaker_results[speaker]["gaslighting_count"] += len(
                self.detect_gaslighting(text, speaker, time)
            )
            speaker_results[speaker]["threat_count"] += len(
                self.detect_threats(text, speaker, time)
            )
            speaker_results[speaker]["coercion_count"] += len(
                self.detect_coercion(text, speaker, time)
            )

        return speaker_results

    # ==========================================================================
    # Report Generation
    # ==========================================================================

    def generate_analysis_report(self, text: str) -> Dict[str, Any]:
        """
        분석 보고서 생성

        Args:
            text: 분석할 텍스트

        Returns:
            분석 보고서 딕셔너리
        """
        gaslighting_matches = self.detect_gaslighting(text)
        threat_matches = self.detect_threats(text)
        coercion_matches = self.detect_coercion(text)
        deception_analysis = self.analyze_deception(text)
        crime_language_score = self.analyze_comprehensive(text)

        recommendations = self._generate_recommendations(
            gaslighting_count=len(gaslighting_matches),
            threat_count=len(threat_matches),
            coercion_count=len(coercion_matches),
            risk_level=crime_language_score.risk_level,
        )

        return {
            "gaslighting_matches": [
                {
                    "type": m.type.value if hasattr(m.type, "value") else str(m.type),
                    "matched_pattern": m.matched_pattern,
                    "confidence": m.confidence,
                    "severity_weight": m.severity_weight,
                }
                for m in gaslighting_matches
            ],
            "threat_matches": [
                {
                    "type": m.type.value if hasattr(m.type, "value") else str(m.type),
                    "matched_pattern": m.matched_pattern,
                    "confidence": m.confidence,
                    "severity_weight": m.severity_weight,
                }
                for m in threat_matches
            ],
            "coercion_matches": [
                {
                    "type": m.type.value if hasattr(m.type, "value") else str(m.type),
                    "matched_pattern": m.matched_pattern,
                    "confidence": m.confidence,
                    "severity_weight": m.severity_weight,
                }
                for m in coercion_matches
            ],
            "deception_analysis": {
                "hedging_score": deception_analysis.hedging_score,
                "distancing_score": deception_analysis.distancing_score,
                "negative_emotion_score": deception_analysis.negative_emotion_score,
                "exclusive_score": deception_analysis.exclusive_score,
                "cognitive_complexity_score": deception_analysis.cognitive_complexity_score,
                "overall_deception_score": deception_analysis.overall_deception_score,
            },
            "crime_language_score": {
                "gaslighting_score": crime_language_score.gaslighting_score,
                "threat_score": crime_language_score.threat_score,
                "coercion_score": crime_language_score.coercion_score,
                "deception_score": crime_language_score.deception_score,
                "overall_risk_score": crime_language_score.overall_risk_score,
                "risk_level": crime_language_score.risk_level,
                "gaslighting_count": crime_language_score.gaslighting_count,
                "threat_count": crime_language_score.threat_count,
                "coercion_count": crime_language_score.coercion_count,
            },
            "recommendations": recommendations,
        }

    def _generate_recommendations(
        self,
        gaslighting_count: int,
        threat_count: int,
        coercion_count: int,
        risk_level: str,
    ) -> List[str]:
        """권고사항 생성"""
        recommendations = []

        if risk_level in ["높음", "매우 높음"]:
            recommendations.append("전문가 상담을 권장합니다")
            recommendations.append("대화 기록을 보존하세요")

        if gaslighting_count >= 3:
            recommendations.append(
                "가스라이팅 패턴이 다수 감지되었습니다. 심리 상담을 고려해보세요"
            )

        if threat_count >= 2:
            recommendations.append("협박 패턴이 감지되었습니다. 법적 조언을 구하시기 바랍니다")

        if coercion_count >= 2:
            recommendations.append(
                "강압 패턴이 감지되었습니다. 신뢰할 수 있는 사람과 상황을 공유하세요"
            )

        if not recommendations:
            recommendations.append("현재 특별한 위험 패턴이 감지되지 않았습니다")

        return recommendations
