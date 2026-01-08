"""
Context Analysis Service with LLM Integration
TASK-011: LLM-based context analysis with LangChain and Claude API
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json

from voice_man.models.crime_tag import CrimeTag
from voice_man.models.gaslighting import GaslightingPattern, GaslightingAnalysisResult
from voice_man.models.emotion import EmotionLabel, EmotionAnalysis


@dataclass
class ContextAnalysisResult:
    """Result of LLM-based context analysis"""

    crime_intent_score: float  # 0.0-1.0 likelihood of criminal intent
    contextual_factors: List[str]  # List of identified context factors
    risk_assessment: str  # "low", "medium", "high", "critical"
    explanation: str  # Natural language explanation
    confidence: float  # 0.0-1.0 confidence in analysis


class ContextAnalysisService:
    """
    Service for LLM-based context analysis using LangChain and Claude API.

    Note: Mock implementation for testing. Real implementation requires:
    - LangChain for prompt management
    - Anthropic Claude API for inference
    - Few-shot examples for consistency
    """

    # Few-shot examples for crime intent classification
    FEW_SHOT_EXAMPLES = [
        {
            "text": "죽여버린다",
            "context": "Argument between spouses",
            "intent": "high",
            "reasoning": "Direct threat with violent language in domestic context",
        },
        {
            "text": "너 때문에 내 인생이 망졌어",
            "context": "Argument between ex-partners",
            "intent": "low",
            "reasoning": "Expression of distress without explicit threat of harm",
        },
    ]

    async def analyze_crime_intent(
        self,
        text: str,
        crime_tags: List[CrimeTag],
        conversation_context: Optional[List[Dict[str, Any]]] = None,
    ) -> ContextAnalysisResult:
        """
        Analyze criminal intent using LLM context understanding.

        Args:
            text: Text utterance to analyze
            crime_tags: Previously detected crime tags
            conversation_context: Previous conversation history

        Returns:
            ContextAnalysisResult with intent score and explanation
        """
        # Mock implementation - in production, call Claude API via LangChain
        crime_score = self._calculate_crime_score(crime_tags)

        if crime_score > 0.8:
            risk_assessment = "critical"
            explanation = (
                "High probability of criminal intent detected. "
                "Explicit threats or intimidation present. "
                "Immediate legal action recommended."
            )
        elif crime_score > 0.6:
            risk_assessment = "high"
            explanation = (
                "Significant indicators of criminal intent. "
                "Threatening language or intimidation detected. "
                "Legal consultation advised."
            )
        elif crime_score > 0.3:
            risk_assessment = "medium"
            explanation = (
                "Moderate risk factors present. "
                "Context suggests potential for escalation. "
                "Monitor and document interactions."
            )
        else:
            risk_assessment = "low"
            explanation = (
                "Low probability of criminal intent. "
                "No explicit threats or intimidation detected. "
                "Continue monitoring."
            )

        contextual_factors = self._extract_contextual_factors(
            text, crime_tags, conversation_context
        )

        return ContextAnalysisResult(
            crime_intent_score=crime_score,
            contextual_factors=contextual_factors,
            risk_assessment=risk_assessment,
            explanation=explanation,
            confidence=0.85,  # Mock confidence
        )

    async def analyze_gaslighting_context(
        self,
        text: str,
        gaslighting_patterns: List[GaslightingPattern],
        conversation_context: Optional[List[Dict[str, Any]]] = None,
    ) -> ContextAnalysisResult:
        """
        Analyze gaslighting patterns in conversation context.

        Args:
            text: Text utterance to analyze
            gaslighting_patterns: Detected gaslighting patterns
            conversation_context: Previous conversation history

        Returns:
            ContextAnalysisResult with gaslighting severity assessment
        """
        pattern_count = len(gaslighting_patterns)
        pattern_types = {p.type for p in gaslighting_patterns}

        # Calculate gaslighting severity
        if pattern_count >= 4:
            severity = "critical"
            explanation = (
                "Severe gaslighting detected. Multiple pattern types present. "
                "High psychological manipulation risk. "
                "Professional intervention strongly recommended."
            )
            score = 0.95
        elif pattern_count >= 2:
            severity = "high"
            explanation = (
                "Significant gaslighting patterns detected. "
                "Multiple manipulation tactics present. "
                "Psychological support recommended."
            )
            score = 0.75
        elif pattern_count == 1:
            severity = "medium"
            explanation = (
                "Gaslighting pattern detected. "
                "Monitor for escalation and manipulation tactics. "
                "Document all interactions."
            )
            score = 0.5
        else:
            severity = "low"
            explanation = "No significant gaslighting patterns detected."
            score = 0.1

        # Convert pattern types to strings for display
        pattern_type_strs = [str(pt) for pt in pattern_types]

        contextual_factors = [
            f"Pattern types: {', '.join(pattern_type_strs)}",
            f"Pattern frequency: {pattern_count}",
        ]

        return ContextAnalysisResult(
            crime_intent_score=score,
            contextual_factors=contextual_factors,
            risk_assessment=severity,
            explanation=explanation,
            confidence=0.80,
        )

    async def analyze_emotional_escalation(
        self,
        emotion_timeline: List[EmotionAnalysis],
        speaker_id: str,
    ) -> Dict[str, Any]:
        """
        Analyze emotional escalation patterns over time.

        Args:
            emotion_timeline: Timeline of emotion analyses
            speaker_id: Speaker to analyze

        Returns:
            Dict with escalation metrics and recommendations
        """
        if len(emotion_timeline) < 2:
            return {
                "escalation_detected": False,
                "reason": "Insufficient data points",
                "recommendation": "Continue monitoring",
            }

        # Check for escalation to negative emotions
        negative_emotions = {
            EmotionLabel.ANGER,
            EmotionLabel.FEAR,
            EmotionLabel.SADNESS,
            EmotionLabel.DISGUST,
        }

        recent_emotions = emotion_timeline[-3:]  # Last 3 utterances
        negative_count = sum(1 for e in recent_emotions if e.primary_emotion in negative_emotions)

        # Calculate average intensity
        avg_intensity = sum(e.intensity for e in recent_emotions) / len(recent_emotions)

        escalation_detected = negative_count >= 2 and avg_intensity > 0.6

        if escalation_detected:
            recommendation = (
                "Emotional escalation detected. "
                "Speaker showing sustained negative emotions with high intensity. "
                "De-escalation techniques recommended. "
                "Consider professional mediation."
            )
        else:
            recommendation = "No significant emotional escalation detected."

        return {
            "escalation_detected": escalation_detected,
            "negative_emotion_ratio": negative_count / len(recent_emotions),
            "average_intensity": avg_intensity,
            "recommendation": recommendation,
        }

    def _calculate_crime_score(self, crime_tags: List[CrimeTag]) -> float:
        """Calculate crime intent score from tags."""
        if not crime_tags:
            return 0.0

        # Weight by crime type and confidence
        weights = {
            "협박": 0.9,
            "공갈": 0.8,
            "사기": 0.6,
            "모욕": 0.3,
        }

        total_score = 0.0
        for tag in crime_tags:
            weight = weights.get(tag.type, 0.5)
            total_score += weight * tag.confidence

        # Normalize to 0-1
        return min(1.0, total_score / len(crime_tags))

    def _extract_contextual_factors(
        self,
        text: str,
        crime_tags: List[CrimeTag],
        conversation_context: Optional[List[Dict[str, Any]]],
    ) -> List[str]:
        """Extract relevant contextual factors."""
        factors = []

        # Add crime types as factors
        crime_types = {tag.type for tag in crime_tags}
        if crime_types:
            factors.append(f"Crime types: {', '.join(crime_types)}")

        # Add conversation length context
        if conversation_context:
            factors.append(f"Conversation length: {len(conversation_context)} utterances")

        # Add text length factor
        text_length = len(text)
        if text_length > 100:
            factors.append("Long utterance (>100 chars)")
        elif text_length < 20:
            factors.append("Short utterance (<20 chars)")

        return factors
