"""
TASK-011: LLM-based Context Analysis Tests
Test context analysis with LangChain and Claude API integration
"""

import pytest
from voice_man.models.crime_tag import CrimeTag, CrimeType
from voice_man.models.gaslighting import GaslightingPattern, GaslightingPatternType
from voice_man.models.emotion import EmotionLabel, EmotionAnalysis
from voice_man.services.context_analysis_service import (
    ContextAnalysisService,
    ContextAnalysisResult,
)


@pytest.mark.asyncio
class TestContextAnalysisService:
    """Test ContextAnalysisService"""

    async def test_analyze_crime_intent_high_risk(self):
        """Test analyzing high-risk criminal intent"""
        service = ContextAnalysisService()

        crime_tags = [
            CrimeTag(
                type=CrimeType.THREAT,
                confidence=0.95,
                keywords=["죽여버린다"],
                legal_reference="형법 제250조",
            )
        ]

        result = await service.analyze_crime_intent("죽여버린다", crime_tags)

        assert isinstance(result, ContextAnalysisResult)
        assert result.crime_intent_score > 0.8
        assert result.risk_assessment in ["low", "medium", "high", "critical"]
        assert len(result.contextual_factors) > 0
        assert len(result.explanation) > 0
        assert 0.0 <= result.confidence <= 1.0

    async def test_analyze_crime_intent_no_crime(self):
        """Test analyzing text with no criminal intent"""
        service = ContextAnalysisService()

        result = await service.analyze_crime_intent("안녕하세요", [])

        assert result.crime_intent_score == 0.0
        assert result.risk_assessment == "low"

    async def test_analyze_crime_intent_multiple_crimes(self):
        """Test analyzing text with multiple crime types"""
        service = ContextAnalysisService()

        crime_tags = [
            CrimeTag(
                type=CrimeType.THREAT,
                confidence=0.9,
                keywords=["죽여버린다"],
                legal_reference="형법 제250조",
            ),
            CrimeTag(
                type=CrimeType.INSULT,
                confidence=0.8,
                keywords=["미친년"],
                legal_reference="형법 제311조",
            ),
        ]

        result = await service.analyze_crime_intent("죽여버린다 미친년아", crime_tags)

        assert result.crime_intent_score > 0.5
        assert "협박" in str(result.contextual_factors)

    async def test_analyze_gaslighting_severe(self):
        """Test analyzing severe gaslighting"""
        service = ContextAnalysisService()

        patterns = [
            GaslightingPattern(
                type=GaslightingPatternType.DENIAL,
                text="그런 적 없어",
                confidence=0.9,
                speaker="A",
                timestamp="00:00:01",
            ),
            GaslightingPattern(
                type=GaslightingPatternType.BLAME_SHIFTING,
                text="네 때문이야",
                confidence=0.85,
                speaker="A",
                timestamp="00:00:02",
            ),
            GaslightingPattern(
                type=GaslightingPatternType.MINIMIZING,
                text="과민 반응이야",
                confidence=0.8,
                speaker="A",
                timestamp="00:00:03",
            ),
            GaslightingPattern(
                type=GaslightingPatternType.CONFUSION,
                text="혼나게 하는 거야?",
                confidence=0.75,
                speaker="A",
                timestamp="00:00:04",
            ),
        ]

        result = await service.analyze_gaslighting_context("혼란스러워...", patterns)

        assert result.risk_assessment in ["low", "medium", "high", "critical"]
        assert len(result.contextual_factors) > 0
        assert result.crime_intent_score > 0.8  # 4 patterns = severe

    async def test_analyze_gaslighting_no_patterns(self):
        """Test analyzing with no gaslighting patterns"""
        service = ContextAnalysisService()

        result = await service.analyze_gaslighting_context("정상적인 대화입니다", [])

        assert result.crime_intent_score < 0.3
        assert result.risk_assessment == "low"

    async def test_analyze_emotional_escalation(self):
        """Test emotional escalation detection"""
        service = ContextAnalysisService()

        emotion_timeline = [
            EmotionAnalysis(
                primary_emotion=EmotionLabel.JOY,
                intensity=0.5,
                emotion_distribution={
                    "joy": 0.8,
                    "neutral": 0.2,
                    "sadness": 0.0,
                    "anger": 0.0,
                    "fear": 0.0,
                    "disgust": 0.0,
                    "surprise": 0.0,
                },
                confidence=0.9,
            ),
            EmotionAnalysis(
                primary_emotion=EmotionLabel.ANGER,
                intensity=0.8,
                emotion_distribution={
                    "anger": 0.9,
                    "neutral": 0.1,
                    "sadness": 0.0,
                    "joy": 0.0,
                    "fear": 0.0,
                    "disgust": 0.0,
                    "surprise": 0.0,
                },
                confidence=0.85,
            ),
            EmotionAnalysis(
                primary_emotion=EmotionLabel.ANGER,
                intensity=0.9,
                emotion_distribution={
                    "anger": 0.95,
                    "neutral": 0.05,
                    "sadness": 0.0,
                    "joy": 0.0,
                    "fear": 0.0,
                    "disgust": 0.0,
                    "surprise": 0.0,
                },
                confidence=0.9,
            ),
        ]

        result = await service.analyze_emotional_escalation(emotion_timeline, "speaker-1")

        assert "escalation_detected" in result
        assert "negative_emotion_ratio" in result
        assert "average_intensity" in result
        assert "recommendation" in result
        assert result["escalation_detected"] is True

    async def test_analyze_emotional_escalation_no_escalation(self):
        """Test emotional escalation when not present"""
        service = ContextAnalysisService()

        emotion_timeline = [
            EmotionAnalysis(
                primary_emotion=EmotionLabel.JOY,
                intensity=0.5,
                emotion_distribution={
                    "joy": 0.8,
                    "neutral": 0.2,
                    "sadness": 0.0,
                    "anger": 0.0,
                    "fear": 0.0,
                    "disgust": 0.0,
                    "surprise": 0.0,
                },
                confidence=0.9,
            ),
            EmotionAnalysis(
                primary_emotion=EmotionLabel.NEUTRAL,
                intensity=0.3,
                emotion_distribution={
                    "neutral": 0.9,
                    "joy": 0.1,
                    "sadness": 0.0,
                    "anger": 0.0,
                    "fear": 0.0,
                    "disgust": 0.0,
                    "surprise": 0.0,
                },
                confidence=0.85,
            ),
        ]

        result = await service.analyze_emotional_escalation(emotion_timeline, "speaker-1")

        assert result["escalation_detected"] is False

    async def test_analyze_emotional_escalation_insufficient_data(self):
        """Test emotional escalation with insufficient data"""
        service = ContextAnalysisService()

        emotion_timeline = [
            EmotionAnalysis(
                primary_emotion=EmotionLabel.ANGER,
                intensity=0.9,
                emotion_distribution={
                    "anger": 1.0,
                    "neutral": 0.0,
                    "sadness": 0.0,
                    "joy": 0.0,
                    "fear": 0.0,
                    "disgust": 0.0,
                    "surprise": 0.0,
                },
                confidence=0.9,
            )
        ]

        result = await service.analyze_emotional_escalation(emotion_timeline, "speaker-1")

        assert result["escalation_detected"] is False
        assert "Insufficient data" in result["reason"]

    async def test_contextual_factors_extraction(self):
        """Test contextual factor extraction"""
        service = ContextAnalysisService()

        crime_tags = [
            CrimeTag(
                type=CrimeType.THREAT,
                confidence=0.9,
                keywords=["threat"],
                legal_reference="형법 제250조",
            )
        ]

        conversation_context = [{"speaker": "A", "text": "previous utterance"}] * 10

        result = await service.analyze_crime_intent(
            "죽여버린다" * 21,  # Long text > 100 chars (105 chars)
            crime_tags,
            conversation_context,
        )

        assert any("Crime types" in str(f) for f in result.contextual_factors)
        assert any("Conversation length" in str(f) for f in result.contextual_factors)
        assert any("(>100 chars)" in str(f) for f in result.contextual_factors)

    async def test_risk_assessment_levels(self):
        """Test all risk assessment levels"""
        service = ContextAnalysisService()

        # Critical risk
        critical_tags = [
            CrimeTag(
                type=CrimeType.THREAT,
                confidence=0.95,
                keywords=["죽여버린다"],
                legal_reference="형법 제250조",
            )
        ] * 3
        critical_result = await service.analyze_crime_intent("죽여버린다", critical_tags)
        assert critical_result.risk_assessment == "critical"

        # Low risk
        low_result = await service.analyze_crime_intent("안녕하세요", [])
        assert low_result.risk_assessment == "low"


class TestEdgeCases:
    """Test edge cases and error handling"""

    @pytest.mark.asyncio
    async def test_empty_text(self):
        """Test analyzing empty text"""
        service = ContextAnalysisService()

        result = await service.analyze_crime_intent("", [])

        assert result.crime_intent_score == 0.0
        assert result.risk_assessment == "low"

    @pytest.mark.asyncio
    async def test_very_long_text(self):
        """Test analyzing very long text"""
        service = ContextAnalysisService()

        long_text = "위협적 발언 " * 1000
        crime_tags = [
            CrimeTag(
                type=CrimeType.THREAT,
                confidence=0.9,
                keywords=["threat"],
                legal_reference="형법 제250조",
            )
        ]

        result = await service.analyze_crime_intent(long_text, crime_tags)

        assert result.crime_intent_score > 0.0
        assert len(result.explanation) > 0

    @pytest.mark.asyncio
    async def test_mixed_crime_types(self):
        """Test analyzing mixed crime types"""
        service = ContextAnalysisService()

        crime_tags = [
            CrimeTag(
                type=CrimeType.THREAT,
                confidence=0.9,
                keywords=["threat"],
                legal_reference="형법 제250조",
            ),
            CrimeTag(
                type=CrimeType.INTIMIDATION,
                confidence=0.8,
                keywords=["intimidation"],
                legal_reference="형법 제283조",
            ),
            CrimeTag(
                type=CrimeType.FRAUD,
                confidence=0.7,
                keywords=["fraud"],
                legal_reference="형법 제347조",
            ),
        ]

        result = await service.analyze_crime_intent("mixed crimes", crime_tags)

        assert result.crime_intent_score > 0.5
        assert len(result.contextual_factors) > 0
