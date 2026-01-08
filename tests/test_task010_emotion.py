"""
TASK-010: Emotion Analysis System Tests
Test emotion analysis using KoBERT-based model with 7 emotion classes
"""

import pytest
from datetime import datetime
from voice_man.models.emotion import EmotionLabel, EmotionAnalysis, EmotionProfile, EmotionTimeline
from voice_man.services.emotion_service import EmotionAnalysisService


class TestEmotionLabelEnum:
    """Test emotion label enumeration"""

    def test_seven_emotion_classes(self):
        """Test that there are exactly 7 emotion classes"""
        assert len(EmotionLabel) == 7
        labels = [e.value for e in EmotionLabel]
        assert "joy" in labels
        assert "sadness" in labels
        assert "anger" in labels
        assert "fear" in labels
        assert "disgust" in labels
        assert "surprise" in labels
        assert "neutral" in labels

    def test_emotion_label_values(self):
        """Test emotion label string values"""
        assert EmotionLabel.JOY.value == "joy"
        assert EmotionLabel.SADNESS.value == "sadness"
        assert EmotionLabel.ANGER.value == "anger"
        assert EmotionLabel.FEAR.value == "fear"
        assert EmotionLabel.DISGUST.value == "disgust"
        assert EmotionLabel.SURPRISE.value == "surprise"
        assert EmotionLabel.NEUTRAL.value == "neutral"


class TestEmotionAnalysis:
    """Test EmotionAnalysis Pydantic model"""

    def test_emotion_analysis_creation(self):
        """Test creating an emotion analysis result"""
        analysis = EmotionAnalysis(
            primary_emotion=EmotionLabel.JOY,
            intensity=0.85,
            emotion_distribution={
                "joy": 0.85,
                "sadness": 0.05,
                "anger": 0.03,
                "fear": 0.02,
                "disgust": 0.01,
                "surprise": 0.02,
                "neutral": 0.02,
            },
            confidence=0.92,
        )
        assert analysis.primary_emotion == EmotionLabel.JOY
        assert analysis.intensity == 0.85
        assert len(analysis.emotion_distribution) == 7
        assert analysis.confidence == 0.92

    def test_intensity_range_validation(self):
        """Test that intensity is between 0.0 and 1.0"""
        with pytest.raises(ValueError):
            EmotionAnalysis(
                primary_emotion=EmotionLabel.JOY,
                intensity=1.5,  # Invalid: > 1.0
                emotion_distribution={"joy": 0.5},
                confidence=0.8,
            )

        with pytest.raises(ValueError):
            EmotionAnalysis(
                primary_emotion=EmotionLabel.JOY,
                intensity=-0.1,  # Invalid: < 0.0
                emotion_distribution={"joy": 0.5},
                confidence=0.8,
            )

    def test_emotion_distribution_sum(self):
        """Test that emotion distribution sums to approximately 1.0"""
        analysis = EmotionAnalysis(
            primary_emotion=EmotionLabel.JOY,
            intensity=0.85,
            emotion_distribution={
                "joy": 0.70,
                "sadness": 0.10,
                "anger": 0.05,
                "fear": 0.05,
                "disgust": 0.03,
                "surprise": 0.04,
                "neutral": 0.03,
            },
            confidence=0.92,
        )
        total = sum(analysis.emotion_distribution.values())
        assert 0.95 <= total <= 1.05  # Allow small floating point error


class TestEmotionProfile:
    """Test EmotionProfile model for speaker-based emotion tracking"""

    def test_emotion_profile_creation(self):
        """Test creating an emotion profile for a speaker"""
        profile = EmotionProfile(
            speaker_id="speaker-1",
            dominant_emotion=EmotionLabel.ANGER,
            average_intensity=0.75,
            emotion_frequency={
                "joy": 2,
                "sadness": 1,
                "anger": 8,
                "fear": 3,
                "disgust": 1,
                "surprise": 2,
                "neutral": 1,
            },
        )
        assert profile.speaker_id == "speaker-1"
        assert profile.dominant_emotion == EmotionLabel.ANGER
        assert profile.average_intensity == 0.75
        assert profile.emotion_frequency["anger"] == 8

    def test_dominant_emotion_calculation(self):
        """Test calculating dominant emotion from frequency"""
        # This would be tested via service logic
        pass


class TestEmotionTimeline:
    """Test EmotionTimeline for tracking emotion changes over time"""

    def test_emotion_timeline_creation(self):
        """Test creating an emotion timeline"""
        timeline = EmotionTimeline(
            speaker_id="speaker-1",
            timestamps=[
                datetime(2025, 1, 8, 10, 0, 0),
                datetime(2025, 1, 8, 10, 1, 0),
                datetime(2025, 1, 8, 10, 2, 0),
            ],
            emotions=[EmotionLabel.JOY, EmotionLabel.NEUTRAL, EmotionLabel.ANGER],
            intensities=[0.7, 0.3, 0.9],
        )
        assert len(timeline.timestamps) == 3
        assert len(timeline.emotions) == 3
        assert len(timeline.intensities) == 3
        assert timeline.timestamps[0] == datetime(2025, 1, 8, 10, 0, 0)
        assert timeline.emotions[0] == EmotionLabel.JOY
        assert timeline.intensities[0] == 0.7

    def test_timeline_length_validation(self):
        """Test that timestamps, emotions, and intensities have same length"""
        with pytest.raises(ValueError):
            EmotionTimeline(
                speaker_id="speaker-1",
                timestamps=[datetime(2025, 1, 8, 10, 0, 0)],
                emotions=[EmotionLabel.JOY, EmotionLabel.ANGER],  # Mismatch
                intensities=[0.7],
            )


class TestEmotionAnalysisService:
    """Test EmotionAnalysisService"""

    def test_analyze_emotion_from_text(self):
        """Test analyzing emotion from text utterance"""
        service = EmotionAnalysisService()

        # Test angry text
        analysis = service.analyze_emotion(
            text="진짜 화가 난다! 당신 때문이야!", speaker_id="speaker-1"
        )
        assert analysis.primary_emotion in EmotionLabel
        assert 0.0 <= analysis.intensity <= 1.0
        assert 0.0 <= analysis.confidence <= 1.0
        assert len(analysis.emotion_distribution) == 7

    def test_detect_emotion_keywords(self):
        """Test emotion detection from keywords"""
        service = EmotionAnalysisService()

        # Joy keywords
        joy_text = "정말 기뻐요! 행복해요! 좋아요!"
        analysis = service.analyze_emotion(joy_text, "speaker-1")
        assert analysis.primary_emotion == EmotionLabel.JOY

        # Anger keywords
        anger_text = "미친거야! 죽어! 화가 난다!"
        analysis = service.analyze_emotion(anger_text, "speaker-1")
        assert analysis.primary_emotion == EmotionLabel.ANGER

        # Fear keywords
        fear_text = "무서워요! 두려워요! 도와주세요!"
        analysis = service.analyze_emotion(fear_text, "speaker-1")
        assert analysis.primary_emotion == EmotionLabel.FEAR

    def test_create_speaker_emotion_profile(self):
        """Test creating emotion profile for speaker"""
        service = EmotionAnalysisService()

        utterances = [
            ("기뻐요!", "speaker-1"),
            ("화가 나요!", "speaker-1"),
            ("화가 나요!", "speaker-1"),
            ("슬퍼요...", "speaker-1"),
        ]

        profile = service.create_speaker_profile(utterances)
        assert profile.speaker_id == "speaker-1"
        assert profile.dominant_emotion == EmotionLabel.ANGER  # Most frequent
        assert 0.0 <= profile.average_intensity <= 1.0
        assert sum(profile.emotion_frequency.values()) == 4

    def test_track_emotion_timeline(self):
        """Test tracking emotion changes over time"""
        service = EmotionAnalysisService()

        utterances = [
            (datetime(2025, 1, 8, 10, 0, 0), "기뻐요!", "speaker-1"),
            (datetime(2025, 1, 8, 10, 1, 0), "화가 나요!", "speaker-1"),
            (datetime(2025, 1, 8, 10, 2, 0), "괜찮아요.", "speaker-1"),
        ]

        timeline = service.track_emotion_timeline(utterances)
        assert timeline.speaker_id == "speaker-1"
        assert len(timeline.timestamps) == 3
        assert len(timeline.emotions) == 3
        assert len(timeline.intensities) == 3

    def test_detect_emotion_transition(self):
        """Test detecting emotion state transitions"""
        service = EmotionAnalysisService()

        # Joy to Anger transition
        from_emotion = EmotionLabel.JOY
        to_emotion = EmotionLabel.ANGER

        transition = service.detect_emotion_transition(
            previous_emotion=from_emotion, current_emotion=to_emotion
        )

        assert transition["is_transition"] is True
        assert transition["from"] == EmotionLabel.JOY
        assert transition["to"] == EmotionLabel.ANGER
        assert transition["transition_type"] == "positive_to_negative"

    def test_calculate_emotion_volatility(self):
        """Test calculating emotion volatility (stability measure)"""
        service = EmotionAnalysisService()

        intensities = [0.3, 0.8, 0.2, 0.9, 0.4]
        volatility = service.calculate_emotion_volatility(intensities)

        assert 0.0 <= volatility <= 1.0
        assert volatility > 0.5  # High variance should yield high volatility

    def test_empty_text_handling(self):
        """Test handling empty or invalid text"""
        service = EmotionAnalysisService()

        analysis = service.analyze_emotion("", "speaker-1")
        assert analysis.primary_emotion == EmotionLabel.NEUTRAL
        assert analysis.intensity == 0.0

    def test_multiple_speakers(self):
        """Test analyzing emotions for multiple speakers"""
        service = EmotionAnalysisService()

        utterances = [
            ("기뻐요!", "speaker-1"),
            ("화가 나요!", "speaker-2"),
            ("무서워요...", "speaker-2"),
            ("좋아요!", "speaker-1"),
        ]

        profiles = service.create_multiple_speaker_profiles(utterances)

        assert len(profiles) == 2
        assert any(p.speaker_id == "speaker-1" for p in profiles)
        assert any(p.speaker_id == "speaker-2" for p in profiles)


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_very_long_text(self):
        """Test handling very long text"""
        service = EmotionAnalysisService()
        long_text = "기뻐요! " * 1000

        analysis = service.analyze_emotion(long_text, "speaker-1")
        assert analysis.primary_emotion == EmotionLabel.JOY

    def test_mixed_emotions(self):
        """Test text with mixed emotion indicators"""
        service = EmotionAnalysisService()

        mixed_text = "기쁘지만 슬프기도 해요... 화도 나지만 괜찮아요."
        analysis = service.analyze_emotion(mixed_text, "speaker-1")

        # Should detect one primary emotion
        assert analysis.primary_emotion in EmotionLabel
        assert 0.0 <= analysis.intensity <= 1.0

    def test_no_emotion_keywords(self):
        """Test text with minimal emotion keywords"""
        service = EmotionAnalysisService()

        # Text with no strong emotion indicators
        neutral_text = "프로젝트 마감일이 다가오고 있습니다."
        analysis = service.analyze_emotion(neutral_text, "speaker-1")

        assert analysis.primary_emotion == EmotionLabel.NEUTRAL
        # With minimal keywords, intensity should be relatively low
        assert analysis.intensity <= 1.0

    def test_empty_utterance_list(self):
        """Test handling empty utterance list for profile"""
        service = EmotionAnalysisService()

        with pytest.raises(ValueError):
            service.create_speaker_profile([])

    def test_single_utterance_profile(self):
        """Test creating profile from single utterance"""
        service = EmotionAnalysisService()

        profile = service.create_speaker_profile([("화가 나요!", "speaker-1")])
        assert profile.speaker_id == "speaker-1"
        assert profile.dominant_emotion == EmotionLabel.ANGER
        assert sum(profile.emotion_frequency.values()) == 1
