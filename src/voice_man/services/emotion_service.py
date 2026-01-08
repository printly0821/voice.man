"""
Emotion Analysis Service
TASK-010: Emotion analysis using keyword-based approach with KoBERT integration planned
"""

from collections import Counter
from datetime import datetime
from typing import Dict, List, Tuple
import statistics
import re

from voice_man.models.emotion import (
    EmotionLabel,
    EmotionAnalysis,
    EmotionProfile,
    EmotionTimeline,
)


class EmotionAnalysisService:
    """
    Service for analyzing emotions in text utterances.

    Note: Currently uses keyword-based analysis.
    KoBERT integration planned for production:
    - Model: KoBERT-base (skt/kobert-base-v1)
    - Fine-tuning: Korean emotion dataset (7 classes)
    - Inference: PyTorch with HuggingFace Transformers
    """

    # Emotion keyword dictionary for Korean text analysis
    EMOTION_KEYWORDS = {
        EmotionLabel.JOY: [
            "기뻐",
            "행복",
            "좋아",
            "즐거워",
            "신나",
            "기쁘",
            "반가워",
            "사랑",
            "행운",
            "축하",
            "웃겨",
            "재미있",
        ],
        EmotionLabel.SADNESS: [
            "슬퍼",
            "우울",
            "비참",
            "속상",
            "마음 아파",
            "눈물",
            "울어",
            "가슴 아파",
            "힘들어",
            "외로워",
            "그리워",
            "미안",
        ],
        EmotionLabel.ANGER: [
            "화가",
            "화나",
            "미친",
            "죽어",
            "답답",
            "짜증",
            "분노",
            "성나",
            "억울",
            "넌더리",
            "때려",
            "죽이",
        ],
        EmotionLabel.FEAR: [
            "무서워",
            "두려워",
            "겁나",
            "공포",
            "떨려",
            "불안",
            "아슬아슬",
            "위험",
            "도와",
            "구해",
            "도망",
        ],
        EmotionLabel.DISGUST: [
            "역겨워",
            "구역질",
            "징그러",
            "귀찮",
            "싫어",
            "혐오",
            "역겹",
            "소름",
            "끔찍",
        ],
        EmotionLabel.SURPRISE: [
            "놀랐",
            "깜짝",
            "대박",
            "헐",
            "진짜",
            "말도 안 돼",
            "설마",
            "이럴 수가",
            "어머나",
            "천만에",
        ],
        EmotionLabel.NEUTRAL: [
            "그래",
            "알겠",
            "네",
            "예",
            "괜찮",
            "이해",
            "확인",
            "알겠어",
            "그렇",
            "맞아",
            "아니",
            "물론",
        ],
    }

    def analyze_emotion(self, text: str, speaker_id: str) -> EmotionAnalysis:
        """
        Analyze emotion from text utterance.

        Args:
            text: Text utterance to analyze
            speaker_id: Speaker identifier

        Returns:
            EmotionAnalysis with primary emotion, intensity, distribution, and confidence
        """
        if not text or not text.strip():
            return EmotionAnalysis(
                primary_emotion=EmotionLabel.NEUTRAL,
                intensity=0.0,
                emotion_distribution=self._get_neutral_distribution(),
                confidence=1.0,
            )

        # Count emotion keyword matches
        emotion_scores = self._count_emotion_keywords(text)

        # Determine primary emotion
        max_score = max(emotion_scores.values())

        if max_score == 0:
            # No emotion keywords found - neutral with low intensity
            primary_emotion = EmotionLabel.NEUTRAL
            intensity = 0.1
            confidence = 0.3
        else:
            primary_emotion = max(emotion_scores, key=emotion_scores.get)
            total_score = sum(emotion_scores.values())

            # Calculate intensity based on keyword frequency
            intensity = min(1.0, (max_score / max(1, total_score)) + 0.3)

            # Calculate confidence based on how clear the emotion is
            if total_score > 0:
                confidence = max_score / total_score
            else:
                confidence = 0.5

        # Build emotion distribution
        distribution = self._build_emotion_distribution(emotion_scores)

        return EmotionAnalysis(
            primary_emotion=primary_emotion,
            intensity=intensity,
            emotion_distribution=distribution,
            confidence=confidence,
        )

    def create_speaker_profile(self, utterances: List[Tuple[str, str]]) -> EmotionProfile:
        """
        Create emotion profile for a speaker from multiple utterances.

        Args:
            utterances: List of (text, speaker_id) tuples

        Returns:
            EmotionProfile with dominant emotion and statistics

        Raises:
            ValueError: If utterances list is empty
        """
        if not utterances:
            raise ValueError("utterances list cannot be empty")

        speaker_id = utterances[0][1]

        # Analyze all utterances
        emotions = []
        intensities = []

        for text, spk_id in utterances:
            analysis = self.analyze_emotion(text, spk_id)
            emotions.append(analysis.primary_emotion)
            intensities.append(analysis.intensity)

        # Calculate emotion frequency
        emotion_counter = Counter(emotions)

        # Determine dominant emotion
        if emotion_counter:
            dominant_emotion = emotion_counter.most_common(1)[0][0]
        else:
            dominant_emotion = EmotionLabel.NEUTRAL

        # Calculate average intensity
        average_intensity = statistics.mean(intensities) if intensities else 0.0

        # Build emotion frequency dict
        emotion_frequency = {emotion.value: 0 for emotion in EmotionLabel}
        for emotion, count in emotion_counter.items():
            emotion_frequency[emotion.value] = count

        return EmotionProfile(
            speaker_id=speaker_id,
            dominant_emotion=dominant_emotion,
            average_intensity=average_intensity,
            emotion_frequency=emotion_frequency,
        )

    def track_emotion_timeline(
        self, utterances: List[Tuple[datetime, str, str]]
    ) -> EmotionTimeline:
        """
        Track emotion changes over time for a speaker.

        Args:
            utterances: List of (timestamp, text, speaker_id) tuples

        Returns:
            EmotionTimeline with timestamps, emotions, and intensities
        """
        timestamps = []
        emotions = []
        intensities = []

        for timestamp, text, speaker_id in utterances:
            analysis = self.analyze_emotion(text, speaker_id)
            timestamps.append(timestamp)
            emotions.append(analysis.primary_emotion)
            intensities.append(analysis.intensity)

        return EmotionTimeline(
            speaker_id=utterances[0][2] if utterances else "unknown",
            timestamps=timestamps,
            emotions=emotions,
            intensities=intensities,
        )

    def detect_emotion_transition(
        self, previous_emotion: EmotionLabel, current_emotion: EmotionLabel
    ) -> Dict:
        """
        Detect emotion state transition.

        Args:
            previous_emotion: Previous emotion state
            current_emotion: Current emotion state

        Returns:
            Dict with transition info: is_transition, from, to, transition_type
        """
        positive_emotions = {EmotionLabel.JOY, EmotionLabel.SURPRISE}
        negative_emotions = {
            EmotionLabel.ANGER,
            EmotionLabel.FEAR,
            EmotionLabel.SADNESS,
            EmotionLabel.DISGUST,
        }
        neutral_emotions = {EmotionLabel.NEUTRAL}

        is_transition = previous_emotion != current_emotion

        transition_type = "none"
        if is_transition:
            if previous_emotion in positive_emotions and current_emotion in negative_emotions:
                transition_type = "positive_to_negative"
            elif previous_emotion in negative_emotions and current_emotion in positive_emotions:
                transition_type = "negative_to_positive"
            elif previous_emotion in neutral_emotions and current_emotion in negative_emotions:
                transition_type = "neutral_to_negative"
            elif previous_emotion in neutral_emotions and current_emotion in positive_emotions:
                transition_type = "neutral_to_positive"
            elif current_emotion in neutral_emotions:
                transition_type = "to_neutral"
            else:
                transition_type = "other"

        return {
            "is_transition": is_transition,
            "from": previous_emotion,
            "to": current_emotion,
            "transition_type": transition_type,
        }

    def calculate_emotion_volatility(self, intensities: List[float]) -> float:
        """
        Calculate emotion volatility (emotion stability measure).

        Args:
            intensities: List of emotion intensity values

        Returns:
            Volatility score between 0.0 (stable) and 1.0 (highly volatile)
        """
        if len(intensities) < 2:
            return 0.0

        # Calculate standard deviation as volatility measure
        if len(intensities) > 1:
            stdev = statistics.stdev(intensities)
            # Normalize to 0-1 range (max expected stdev is ~0.5)
            volatility = min(1.0, stdev * 2)
        else:
            volatility = 0.0

        return volatility

    def create_multiple_speaker_profiles(
        self, utterances: List[Tuple[str, str]]
    ) -> List[EmotionProfile]:
        """
        Create emotion profiles for multiple speakers.

        Args:
            utterances: List of (text, speaker_id) tuples

        Returns:
            List of EmotionProfile objects, one per speaker
        """
        # Group utterances by speaker
        speaker_utterances: Dict[str, List[Tuple[str, str]]] = {}

        for text, speaker_id in utterances:
            if speaker_id not in speaker_utterances:
                speaker_utterances[speaker_id] = []
            speaker_utterances[speaker_id].append((text, speaker_id))

        # Create profile for each speaker
        profiles = []
        for speaker_id, speaker_utts in speaker_utterances.items():
            profile = self.create_speaker_profile(speaker_utts)
            profiles.append(profile)

        return profiles

    def _count_emotion_keywords(self, text: str) -> Dict[EmotionLabel, int]:
        """Count emotion keyword matches in text."""
        emotion_scores = {emotion: 0 for emotion in EmotionLabel}

        # Split text into words for more accurate matching
        words = text.split()

        for emotion, keywords in self.EMOTION_KEYWORDS.items():
            for keyword in keywords:
                # Count exact word matches
                for word in words:
                    if keyword in word:
                        emotion_scores[emotion] += 1

        return emotion_scores

    def _build_emotion_distribution(
        self, emotion_scores: Dict[EmotionLabel, int]
    ) -> Dict[str, float]:
        """Build normalized emotion distribution from scores."""
        total_score = sum(emotion_scores.values())

        if total_score == 0:
            return self._get_neutral_distribution()

        distribution = {}
        for emotion in EmotionLabel:
            score = emotion_scores.get(emotion, 0)
            # Normalize to sum to 1.0
            distribution[emotion.value] = score / total_score

        return distribution

    def _get_neutral_distribution(self) -> Dict[str, float]:
        """Get neutral emotion distribution (all neutral)."""
        return {
            "joy": 0.0,
            "sadness": 0.0,
            "anger": 0.0,
            "fear": 0.0,
            "disgust": 0.0,
            "surprise": 0.0,
            "neutral": 1.0,
        }
