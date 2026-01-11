"""
Cross-Validation Service for Text-Voice Analysis
SPEC-FORENSIC-001 Phase 2-C: Text-Voice cross-validation service

Detects discrepancies between text content (what was said) and voice emotion
(how it was said) to identify potential deception.
"""

from typing import List, Optional, Set

from voice_man.models.forensic.cross_validation import (
    CrossValidationResult,
    Discrepancy,
    DiscrepancySeverity,
    DiscrepancyType,
    TextAnalysisResult,
    VoiceAnalysisResult,
)
from voice_man.models.forensic.emotion_recognition import MultiModelEmotionResult
from voice_man.services.forensic.crime_language_service import CrimeLanguageAnalysisService
from voice_man.services.forensic.ser_service import SERService


class CrossValidationService:
    """Text-Voice cross-validation service.

    Analyzes text content and voice emotion to detect discrepancies
    that may indicate deception.
    """

    # Sentiment keywords for basic sentiment analysis
    POSITIVE_KEYWORDS: Set[str] = {
        "happy",
        "love",
        "great",
        "wonderful",
        "amazing",
        "excellent",
        "joy",
        "excited",
        "glad",
        "pleased",
        "thankful",
        "grateful",
        "thrilled",
        "delighted",
        "fantastic",
        "beautiful",
        "good",
        "nice",
        "best",
        # Korean positive words
        "행복",
        "좋아",
        "사랑",
        "기쁘",
        "감사",
        "최고",
        "훌륭",
        "멋지",
        "즐거",
    }

    NEGATIVE_KEYWORDS: Set[str] = {
        "hate",
        "terrible",
        "awful",
        "horrible",
        "disgusting",
        "angry",
        "sad",
        "fear",
        "hurt",
        "destroy",
        "kill",
        "die",
        "dead",
        "worst",
        "bad",
        "ugly",
        # Korean negative words
        "싫어",
        "미워",
        "짜증",
        "화나",
        "슬프",
        "무서",
        "죽",
        "나쁘",
    }

    APOLOGY_KEYWORDS: Set[str] = {
        "sorry",
        "apologize",
        "forgive",
        "regret",
        "mistake",
        "my fault",
        # Korean apology words
        "미안",
        "죄송",
        "용서",
        "실수",
    }

    COMFORT_KEYWORDS: Set[str] = {
        "okay",
        "alright",
        "don't worry",
        "it's fine",
        "no problem",
        "calm down",
        "relax",
        # Korean comfort words
        "괜찮",
        "걱정",
        "안심",
        "진정",
    }

    # Thresholds for discrepancy detection
    POSITIVE_SENTIMENT_THRESHOLD = 0.3
    NEGATIVE_SENTIMENT_THRESHOLD = -0.3
    HIGH_VALENCE_THRESHOLD = 0.7
    LOW_VALENCE_THRESHOLD = 0.3
    HIGH_INTENSITY_THRESHOLD = 0.7
    LOW_AROUSAL_THRESHOLD = 0.3
    HIGH_STRESS_THRESHOLD = 0.7
    LOW_INTENSITY_THRESHOLD = 0.3

    # Severity weights for deception calculation
    SEVERITY_WEIGHTS = {
        DiscrepancySeverity.LOW: 0.15,
        DiscrepancySeverity.MEDIUM: 0.35,
        DiscrepancySeverity.HIGH: 0.6,
        DiscrepancySeverity.CRITICAL: 0.85,
    }

    def __init__(
        self,
        crime_language_service: CrimeLanguageAnalysisService,
        ser_service: SERService,
    ):
        """Initialize the cross-validation service.

        Args:
            crime_language_service: Service for crime language pattern detection.
            ser_service: Service for speech emotion recognition.
        """
        self._crime_language_service = crime_language_service
        self._ser_service = ser_service

    def analyze_text(self, text: str) -> TextAnalysisResult:
        """Analyze text for sentiment, emotions, and crime patterns.

        Args:
            text: The text to analyze.

        Returns:
            TextAnalysisResult with analysis results.

        Raises:
            ValueError: If text is empty.
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        text_lower = text.lower()

        # Get crime language analysis
        crime_analysis = self._crime_language_service.analyze_comprehensive(text)

        # Detect crime patterns
        crime_patterns: List[str] = []
        gaslighting_matches = self._crime_language_service.detect_gaslighting(text)
        threat_matches = self._crime_language_service.detect_threats(text)
        coercion_matches = self._crime_language_service.detect_coercion(text)

        if gaslighting_matches:
            crime_patterns.append("gaslighting")
        if threat_matches:
            crime_patterns.append("threat")
        if coercion_matches:
            crime_patterns.append("coercion")

        # Calculate sentiment score
        sentiment_score = self._calculate_sentiment_score(text_lower, crime_analysis)

        # Determine sentiment category
        if sentiment_score > self.POSITIVE_SENTIMENT_THRESHOLD:
            detected_sentiment = "positive"
        elif sentiment_score < self.NEGATIVE_SENTIMENT_THRESHOLD:
            detected_sentiment = "negative"
        else:
            detected_sentiment = "neutral"

        # Detect emotions from text
        detected_emotions = self._detect_text_emotions(text_lower, crime_analysis)

        # Calculate intensity level
        intensity_level = self._calculate_intensity_level(text, crime_analysis)

        return TextAnalysisResult(
            text=text,
            detected_sentiment=detected_sentiment,
            sentiment_score=sentiment_score,
            detected_emotions=detected_emotions,
            crime_patterns_found=crime_patterns,
            intensity_level=intensity_level,
        )

    def _calculate_sentiment_score(self, text_lower: str, crime_analysis) -> float:
        """Calculate sentiment score from text.

        Args:
            text_lower: Lowercase text.
            crime_analysis: Crime language analysis result.

        Returns:
            Sentiment score from -1.0 to 1.0.
        """
        positive_count = sum(1 for word in self.POSITIVE_KEYWORDS if word in text_lower)
        negative_count = sum(1 for word in self.NEGATIVE_KEYWORDS if word in text_lower)

        # Factor in crime patterns (negative influence)
        crime_factor = (
            crime_analysis.gaslighting_score * 0.3
            + crime_analysis.threat_score * 0.4
            + crime_analysis.coercion_score * 0.3
        )

        total = positive_count + negative_count
        if total == 0:
            base_score = -crime_factor
        else:
            base_score = (positive_count - negative_count) / max(total, 1)
            base_score = base_score - crime_factor

        # Clamp to valid range
        return max(-1.0, min(1.0, base_score))

    def _detect_text_emotions(self, text_lower: str, crime_analysis) -> List[str]:
        """Detect emotions expressed in text.

        Args:
            text_lower: Lowercase text.
            crime_analysis: Crime language analysis result.

        Returns:
            List of detected emotion names.
        """
        emotions: List[str] = []

        # Check for joy/happiness
        joy_words = {"happy", "joy", "excited", "thrilled", "delighted", "glad", "행복", "기쁘"}
        if any(word in text_lower for word in joy_words):
            emotions.append("joy")

        # Check for anger
        anger_words = {"angry", "hate", "furious", "rage", "화나", "짜증", "미워"}
        if any(word in text_lower for word in anger_words):
            emotions.append("anger")

        # Check for sadness
        sad_words = {"sad", "cry", "tears", "depressed", "슬프", "울"}
        if any(word in text_lower for word in sad_words):
            emotions.append("sadness")

        # Check for fear
        fear_words = {"afraid", "scared", "fear", "terrified", "무서", "두려"}
        if any(word in text_lower for word in fear_words):
            emotions.append("fear")

        # Check for remorse (apology)
        if any(word in text_lower for word in self.APOLOGY_KEYWORDS):
            emotions.append("remorse")

        # Check for comfort/reassurance
        if any(word in text_lower for word in self.COMFORT_KEYWORDS):
            emotions.append("comfort")

        # Add emotions based on crime patterns
        if crime_analysis.threat_score > 0.5:
            if "anger" not in emotions:
                emotions.append("anger")

        return emotions

    def _calculate_intensity_level(self, text: str, crime_analysis) -> float:
        """Calculate intensity level of text.

        Args:
            text: Original text.
            crime_analysis: Crime language analysis result.

        Returns:
            Intensity level from 0.0 to 1.0.
        """
        intensity = 0.0

        # Check for uppercase (shouting)
        uppercase_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        intensity += uppercase_ratio * 0.3

        # Check for exclamation marks
        exclamation_ratio = text.count("!") / max(len(text.split()), 1)
        intensity += min(exclamation_ratio * 0.2, 0.2)

        # Factor in crime patterns
        intensity += crime_analysis.threat_score * 0.3
        intensity += crime_analysis.coercion_score * 0.2

        # Check for intense words
        intense_words = {
            "destroy",
            "kill",
            "absolutely",
            "completely",
            "totally",
            "죽",
            "완전",
        }
        text_lower = text.lower()
        intense_count = sum(1 for word in intense_words if word in text_lower)
        intensity += min(intense_count * 0.1, 0.3)

        return min(1.0, intensity)

    def analyze_voice(
        self,
        audio_path: str,
        precomputed_emotion_result: Optional[MultiModelEmotionResult] = None,
    ) -> VoiceAnalysisResult:
        """Analyze voice for emotion and stress.

        Args:
            audio_path: Path to the audio file.
            precomputed_emotion_result: Optional pre-computed SER result to avoid
                duplicate analysis. If provided, skips SER inference entirely.
                SPEC-PERFOPT-001: Performance optimization to eliminate redundant SER calls.

        Returns:
            VoiceAnalysisResult with analysis results.

        Raises:
            FileNotFoundError: If audio file does not exist.
        """
        # SPEC-PERFOPT-001: Use precomputed result if available (avoids duplicate SER)
        if precomputed_emotion_result is not None:
            emotion_result = precomputed_emotion_result
        else:
            # Fallback: compute SER from file (original behavior)
            emotion_result = self._ser_service.analyze_ensemble_from_file(audio_path)
        forensic_indicators = self._ser_service.get_forensic_emotion_indicators(emotion_result)

        # Extract dominant emotion
        dominant_emotion = forensic_indicators.dominant_emotion

        # Extract confidence
        emotion_confidence = forensic_indicators.confidence
        if emotion_result.secondary_result and emotion_result.secondary_result.categorical:
            emotion_confidence = emotion_result.secondary_result.categorical.confidence

        # Extract arousal and valence
        arousal = 0.5
        valence = 0.5
        if emotion_result.primary_result and emotion_result.primary_result.dimensions:
            arousal = emotion_result.primary_result.dimensions.arousal
            valence = emotion_result.primary_result.dimensions.valence

        # Calculate stress level from arousal, valence, and forensic indicators
        stress_level = self._calculate_stress_level(arousal, valence, forensic_indicators)

        return VoiceAnalysisResult(
            dominant_emotion=dominant_emotion,
            emotion_confidence=emotion_confidence,
            arousal=arousal,
            valence=valence,
            stress_level=stress_level,
        )

    def _calculate_stress_level(self, arousal: float, valence: float, indicators) -> float:
        """Calculate stress level from voice analysis.

        Args:
            arousal: Arousal level (0-1).
            valence: Valence level (0-1).
            indicators: Forensic emotion indicators.

        Returns:
            Stress level from 0.0 to 1.0.
        """
        stress = 0.0

        # High arousal contributes to stress
        stress += arousal * 0.4

        # Low valence (negative emotion) contributes to stress
        stress += (1 - valence) * 0.3

        # Forensic stress indicator
        if indicators.stress_indicator:
            stress += 0.3

        return min(1.0, stress)

    def detect_discrepancies(
        self,
        text_result: TextAnalysisResult,
        voice_result: VoiceAnalysisResult,
    ) -> List[Discrepancy]:
        """Detect discrepancies between text and voice analysis.

        Args:
            text_result: Results from text analysis.
            voice_result: Results from voice analysis.

        Returns:
            List of detected discrepancies.
        """
        discrepancies: List[Discrepancy] = []

        # Check for EMOTION_TEXT_MISMATCH
        emotion_mismatch = self._check_emotion_text_mismatch(text_result, voice_result)
        if emotion_mismatch:
            discrepancies.append(emotion_mismatch)

        # Check for INTENSITY_MISMATCH
        intensity_mismatch = self._check_intensity_mismatch(text_result, voice_result)
        if intensity_mismatch:
            discrepancies.append(intensity_mismatch)

        # Check for SENTIMENT_CONTRADICTION
        sentiment_contradiction = self._check_sentiment_contradiction(text_result, voice_result)
        if sentiment_contradiction:
            discrepancies.append(sentiment_contradiction)

        # Check for STRESS_CONTENT_MISMATCH
        stress_mismatch = self._check_stress_content_mismatch(text_result, voice_result)
        if stress_mismatch:
            discrepancies.append(stress_mismatch)

        return discrepancies

    def _check_emotion_text_mismatch(
        self,
        text_result: TextAnalysisResult,
        voice_result: VoiceAnalysisResult,
    ) -> Optional[Discrepancy]:
        """Check for emotion-text mismatch.

        Rules:
        - Positive text (> 0.3) + low voice valence (< 0.3) -> HIGH
        - Negative text (< -0.3) + high voice valence (> 0.7) -> HIGH
        - Joy in text + sad/angry voice -> CRITICAL
        """
        text_sentiment = text_result.sentiment_score
        voice_valence = voice_result.valence
        voice_emotion = voice_result.dominant_emotion

        # Check for joy text + sad/angry voice (CRITICAL)
        joy_in_text = (
            "joy" in text_result.detected_emotions or "excitement" in text_result.detected_emotions
        )
        if joy_in_text and voice_emotion in ["sad", "angry"]:
            return Discrepancy(
                discrepancy_type=DiscrepancyType.EMOTION_TEXT_MISMATCH,
                severity=DiscrepancySeverity.CRITICAL,
                description=f"Joy expressed in text but {voice_emotion} emotion detected in voice",
                text_evidence=f"Detected emotions: {text_result.detected_emotions}",
                voice_evidence=f"Dominant emotion: {voice_emotion}",
                confidence=0.9,
            )

        # Check positive text + negative voice
        if (
            text_sentiment > self.POSITIVE_SENTIMENT_THRESHOLD
            and voice_valence < self.LOW_VALENCE_THRESHOLD
        ):
            return Discrepancy(
                discrepancy_type=DiscrepancyType.EMOTION_TEXT_MISMATCH,
                severity=DiscrepancySeverity.HIGH,
                description="Positive text sentiment with negative voice emotion",
                text_evidence=f"Sentiment score: {text_sentiment:.2f}",
                voice_evidence=f"Valence: {voice_valence:.2f}, Emotion: {voice_emotion}",
                confidence=0.85,
            )

        # Check negative text + positive voice
        if (
            text_sentiment < self.NEGATIVE_SENTIMENT_THRESHOLD
            and voice_valence > self.HIGH_VALENCE_THRESHOLD
        ):
            return Discrepancy(
                discrepancy_type=DiscrepancyType.EMOTION_TEXT_MISMATCH,
                severity=DiscrepancySeverity.HIGH,
                description="Negative text sentiment with positive voice emotion",
                text_evidence=f"Sentiment score: {text_sentiment:.2f}",
                voice_evidence=f"Valence: {voice_valence:.2f}, Emotion: {voice_emotion}",
                confidence=0.85,
            )

        return None

    def _check_intensity_mismatch(
        self,
        text_result: TextAnalysisResult,
        voice_result: VoiceAnalysisResult,
    ) -> Optional[Discrepancy]:
        """Check for intensity mismatch.

        Rules:
        - High text intensity (> 0.7) + low arousal (< 0.3) -> HIGH
        - Threat/coercion pattern + low arousal -> CRITICAL
        """
        text_intensity = text_result.intensity_level
        voice_arousal = voice_result.arousal

        # Check for threat with low arousal (CRITICAL)
        has_threat = "threat" in text_result.crime_patterns_found
        has_coercion = "coercion" in text_result.crime_patterns_found
        if (has_threat or has_coercion) and voice_arousal < self.LOW_AROUSAL_THRESHOLD:
            return Discrepancy(
                discrepancy_type=DiscrepancyType.INTENSITY_MISMATCH,
                severity=DiscrepancySeverity.CRITICAL,
                description="Threatening/coercive text delivered with calm voice",
                text_evidence=f"Crime patterns: {text_result.crime_patterns_found}",
                voice_evidence=f"Arousal: {voice_arousal:.2f}",
                confidence=0.9,
            )

        # Check high intensity text + low arousal
        if (
            text_intensity > self.HIGH_INTENSITY_THRESHOLD
            and voice_arousal < self.LOW_AROUSAL_THRESHOLD
        ):
            return Discrepancy(
                discrepancy_type=DiscrepancyType.INTENSITY_MISMATCH,
                severity=DiscrepancySeverity.HIGH,
                description="High intensity text with low voice arousal",
                text_evidence=f"Intensity level: {text_intensity:.2f}",
                voice_evidence=f"Arousal: {voice_arousal:.2f}",
                confidence=0.8,
            )

        return None

    def _check_sentiment_contradiction(
        self,
        text_result: TextAnalysisResult,
        voice_result: VoiceAnalysisResult,
    ) -> Optional[Discrepancy]:
        """Check for sentiment contradiction.

        Rules:
        - Apology text + angry/contempt voice -> HIGH
        - Comfort text + fear/disgust voice -> MEDIUM
        """
        voice_emotion = voice_result.dominant_emotion
        text_emotions = text_result.detected_emotions

        # Check apology text + angry voice (HIGH)
        has_apology = "remorse" in text_emotions
        if has_apology and voice_emotion in ["angry", "contempt"]:
            return Discrepancy(
                discrepancy_type=DiscrepancyType.SENTIMENT_CONTRADICTION,
                severity=DiscrepancySeverity.HIGH,
                description="Apology text delivered with angry voice",
                text_evidence="Apology/remorse detected in text",
                voice_evidence=f"Voice emotion: {voice_emotion}",
                confidence=0.85,
            )

        # Check comfort text + fear/disgust voice (MEDIUM)
        has_comfort = "comfort" in text_emotions
        if has_comfort and voice_emotion in ["fear", "disgust"]:
            return Discrepancy(
                discrepancy_type=DiscrepancyType.SENTIMENT_CONTRADICTION,
                severity=DiscrepancySeverity.MEDIUM,
                description="Comforting text delivered with fearful/disgusted voice",
                text_evidence="Comfort/reassurance detected in text",
                voice_evidence=f"Voice emotion: {voice_emotion}",
                confidence=0.75,
            )

        return None

    def _check_stress_content_mismatch(
        self,
        text_result: TextAnalysisResult,
        voice_result: VoiceAnalysisResult,
    ) -> Optional[Discrepancy]:
        """Check for stress-content mismatch.

        Rules:
        - Casual content (low intensity) + high stress (> 0.7) -> MEDIUM
        - Important content + very low stress (< 0.2) -> LOW
        """
        text_intensity = text_result.intensity_level
        voice_stress = voice_result.stress_level

        # Check casual content + high stress (MEDIUM)
        if (
            text_intensity < self.LOW_INTENSITY_THRESHOLD
            and voice_stress > self.HIGH_STRESS_THRESHOLD
        ):
            return Discrepancy(
                discrepancy_type=DiscrepancyType.STRESS_CONTENT_MISMATCH,
                severity=DiscrepancySeverity.MEDIUM,
                description="Casual content with high voice stress",
                text_evidence=f"Intensity level: {text_intensity:.2f}",
                voice_evidence=f"Stress level: {voice_stress:.2f}",
                confidence=0.7,
            )

        # Check important content + very low stress (LOW)
        if text_intensity > self.HIGH_INTENSITY_THRESHOLD and voice_stress < 0.2:
            return Discrepancy(
                discrepancy_type=DiscrepancyType.STRESS_CONTENT_MISMATCH,
                severity=DiscrepancySeverity.LOW,
                description="Important content delivered with very low stress",
                text_evidence=f"Intensity level: {text_intensity:.2f}",
                voice_evidence=f"Stress level: {voice_stress:.2f}",
                confidence=0.6,
            )

        return None

    def calculate_deception_probability(self, discrepancies: List[Discrepancy]) -> float:
        """Calculate deception probability from discrepancies.

        Args:
            discrepancies: List of detected discrepancies.

        Returns:
            Deception probability from 0.0 to 1.0.
        """
        if not discrepancies:
            return 0.05  # Base probability when no discrepancies

        total_weight = 0.0
        total_confidence = 0.0

        for discrepancy in discrepancies:
            # Get severity weight
            severity_key = DiscrepancySeverity(discrepancy.severity)
            severity_weight = self.SEVERITY_WEIGHTS.get(severity_key, 0.3)

            # Weight by confidence
            weighted_contribution = severity_weight * discrepancy.confidence
            total_weight += weighted_contribution
            total_confidence += discrepancy.confidence

        # Calculate weighted average
        if total_confidence > 0:
            probability = total_weight / len(discrepancies) * (1 + len(discrepancies) * 0.1)
        else:
            probability = 0.1

        return min(1.0, probability)

    def _calculate_consistency_score(self, discrepancies: List[Discrepancy]) -> float:
        """Calculate overall consistency score.

        Args:
            discrepancies: List of detected discrepancies.

        Returns:
            Consistency score from 0.0 to 1.0 (higher = more consistent).
        """
        if not discrepancies:
            return 0.95

        # Each discrepancy reduces consistency
        reduction = 0.0
        for d in discrepancies:
            severity_key = DiscrepancySeverity(d.severity)
            severity_weight = self.SEVERITY_WEIGHTS.get(severity_key, 0.3)
            reduction += severity_weight * d.confidence * 0.3

        consistency = 1.0 - min(reduction, 0.9)
        return max(0.1, consistency)

    def _determine_risk_level(self, discrepancies: List[Discrepancy]) -> DiscrepancySeverity:
        """Determine overall risk level from discrepancies.

        Args:
            discrepancies: List of detected discrepancies.

        Returns:
            Overall risk level.
        """
        if not discrepancies:
            return DiscrepancySeverity.LOW

        # Find highest severity
        severities = [DiscrepancySeverity(d.severity) for d in discrepancies]

        if DiscrepancySeverity.CRITICAL in severities:
            return DiscrepancySeverity.CRITICAL
        elif DiscrepancySeverity.HIGH in severities:
            return DiscrepancySeverity.HIGH
        elif DiscrepancySeverity.MEDIUM in severities:
            return DiscrepancySeverity.MEDIUM
        else:
            return DiscrepancySeverity.LOW

    def cross_validate(
        self,
        text: str,
        audio_path: str,
        precomputed_emotion_result: Optional[MultiModelEmotionResult] = None,
    ) -> CrossValidationResult:
        """Perform complete cross-validation analysis.

        Args:
            text: The text transcript to analyze.
            audio_path: Path to the audio file.
            precomputed_emotion_result: Optional pre-computed SER result to avoid
                duplicate analysis. SPEC-PERFOPT-001 optimization.

        Returns:
            CrossValidationResult with complete analysis.

        Raises:
            ValueError: If text is empty.
            FileNotFoundError: If audio file does not exist.
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        # Analyze text
        text_analysis = self.analyze_text(text)

        # Analyze voice (use precomputed SER result if available)
        voice_analysis = self.analyze_voice(audio_path, precomputed_emotion_result)

        # Detect discrepancies
        discrepancies = self.detect_discrepancies(text_analysis, voice_analysis)

        # Calculate scores
        consistency_score = self._calculate_consistency_score(discrepancies)
        deception_probability = self.calculate_deception_probability(discrepancies)
        risk_level = self._determine_risk_level(discrepancies)

        # Generate analysis notes
        analysis_notes = self._generate_analysis_notes(text_analysis, voice_analysis, discrepancies)

        return CrossValidationResult(
            text_analysis=text_analysis,
            voice_analysis=voice_analysis,
            discrepancies=discrepancies,
            overall_consistency_score=consistency_score,
            deception_probability=deception_probability,
            risk_level=risk_level,
            analysis_notes=analysis_notes,
        )

    def _generate_analysis_notes(
        self,
        text_analysis: TextAnalysisResult,
        voice_analysis: VoiceAnalysisResult,
        discrepancies: List[Discrepancy],
    ) -> List[str]:
        """Generate analysis notes.

        Args:
            text_analysis: Text analysis result.
            voice_analysis: Voice analysis result.
            discrepancies: List of discrepancies.

        Returns:
            List of analysis notes.
        """
        notes: List[str] = []

        if not discrepancies:
            notes.append("Text and voice analysis are consistent")
        else:
            notes.append(f"Detected {len(discrepancies)} discrepancy(ies)")

            # Note specific issues
            for d in discrepancies:
                if d.severity in [
                    DiscrepancySeverity.CRITICAL.value,
                    DiscrepancySeverity.HIGH.value,
                ]:
                    notes.append(f"High-priority: {d.description}")

        # Note crime patterns
        if text_analysis.crime_patterns_found:
            patterns = ", ".join(text_analysis.crime_patterns_found)
            notes.append(f"Crime patterns detected: {patterns}")

        # Note high stress
        if voice_analysis.stress_level > 0.7:
            notes.append("High stress level detected in voice")

        return notes
