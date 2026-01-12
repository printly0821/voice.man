"""
STT Accuracy Metrics Service (확장版)

음성 인식 정확도를 측정하고 분석하는 서비스입니다.

제공 기능:
- 신뢰도 점수 분석 (Confidence Scores)
- 화자별 인식 정확도 (Speaker Diarization Accuracy)
- 타임스탬프 정확도 (Timestamp Accuracy)
- 문장 단위 분석 (Sentence-level Analysis)
- 음성-텍스트 일치도 (Voice-Text Consistency)
- 단어/문자 오류율 계산 (WER/CER)
- 전사 통계 (Transcript Statistics)
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceMetrics:
    """신뢰도 메트릭 데이터 클래스"""

    # 단어 수준 신뢰도
    avg_word_confidence: float = 0.0
    min_word_confidence: float = 1.0
    max_word_confidence: float = 0.0
    low_confidence_words: int = 0
    low_confidence_ratio: float = 0.0
    std_word_confidence: float = 0.0

    # 세그먼트 수준 신뢰도
    avg_segment_confidence: float = 0.0
    min_segment_confidence: float = 1.0
    max_segment_confidence: float = 0.0

    # 전체 신뢰도 등급
    overall_confidence_grade: str = "N/A"
    confidence_distribution: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "avg_word_confidence": self.avg_word_confidence,
            "min_word_confidence": self.min_word_confidence,
            "max_word_confidence": self.max_word_confidence,
            "low_confidence_words": self.low_confidence_words,
            "low_confidence_ratio": self.low_confidence_ratio,
            "std_word_confidence": self.std_word_confidence,
            "avg_segment_confidence": self.avg_segment_confidence,
            "min_segment_confidence": self.min_segment_confidence,
            "max_segment_confidence": self.max_segment_confidence,
            "overall_confidence_grade": self.overall_confidence_grade,
            "confidence_distribution": self.confidence_distribution,
        }


@dataclass
class SpeakerAccuracyMetrics:
    """화자별 인식 정확도 메트릭"""

    # 화자별 통계
    speaker_count: int = 0
    speaker_words: Dict[str, int] = field(default_factory=dict)
    speaker_duration: Dict[str, float] = field(default_factory=dict)
    speaker_confidence: Dict[str, float] = field(default_factory=dict)

    # 화자 전환 정확도
    speaker_switches: int = 0
    speaker_switch_accuracy: float = 0.0

    # 화자별 밀도
    speaker_uniformity: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "speaker_count": self.speaker_count,
            "speaker_words": self.speaker_words,
            "speaker_duration": self.speaker_duration,
            "speaker_confidence": self.speaker_confidence,
            "speaker_switches": self.speaker_switches,
            "speaker_switch_accuracy": self.speaker_switch_accuracy,
            "speaker_uniformity": self.speaker_uniformity,
        }


@dataclass
class TimestampAccuracyMetrics:
    """타임스탬프 정확도 메트릭"""

    # 단어 타임스탬프 통계
    avg_word_duration: float = 0.0
    min_word_duration: float = 0.0
    max_word_duration: float = 0.0

    # 타임스탬프 갭 분석
    timestamp_gaps: List[float] = field(default_factory=list)
    avg_timestamp_gap: float = 0.0
    max_timestamp_gap: float = 0.0

    # 타임스탬프 정밀도 (100ms 기준)
    timestamps_within_100ms: int = 0
    timestamps_within_200ms: int = 0
    timestamp_precision_ratio: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "avg_word_duration": self.avg_word_duration,
            "min_word_duration": self.min_word_duration,
            "max_word_duration": self.max_word_duration,
            "avg_timestamp_gap": self.avg_timestamp_gap,
            "max_timestamp_gap": self.max_timestamp_gap,
            "timestamps_within_100ms": self.timestamps_within_100ms,
            "timestamps_within_200ms": self.timestamps_within_200ms,
            "timestamp_precision_ratio": self.timestamp_precision_ratio,
        }


@dataclass
class SentenceMetrics:
    """문장 단위 분석 메트릭"""

    # 문장 통계
    total_sentences: int = 0
    avg_sentence_length: float = 0.0
    avg_words_per_sentence: float = 0.0

    # 문장 완결성
    complete_sentences: int = 0
    sentence_completion_ratio: float = 0.0

    # 문장 부호 정확도
    punctuation_accuracy: float = 0.0
    missing_punctuation: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "total_sentences": self.total_sentences,
            "avg_sentence_length": self.avg_sentence_length,
            "avg_words_per_sentence": self.avg_words_per_sentence,
            "complete_sentences": self.complete_sentences,
            "sentence_completion_ratio": self.sentence_completion_ratio,
            "punctuation_accuracy": self.punctuation_accuracy,
            "missing_punctuation": self.missing_punctuation,
        }


@dataclass
class VoiceTextConsistencyMetrics:
    """음성-텍스트 일치도 메트릭"""

    # 음성 특성 vs 텍스트 일치
    voice_text_correlation: float = 0.0
    acoustic_feature_match: float = 0.0

    # 리듬 일치도
    rhythm_consistency: float = 0.0
    pause_distribution_match: float = 0.0

    # 전체 일치도 점수
    overall_consistency_score: float = 0.0
    consistency_grade: str = "N/A"

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "voice_text_correlation": self.voice_text_correlation,
            "acoustic_feature_match": self.acoustic_feature_match,
            "rhythm_consistency": self.rhythm_consistency,
            "pause_distribution_match": self.pause_distribution_match,
            "overall_consistency_score": self.overall_consistency_score,
            "consistency_grade": self.consistency_grade,
        }


@dataclass
class ErrorMetrics:
    """오류율 메트릭 데이터 클래스"""

    # 단어 오류율 (WER)
    wer: float = 0.0
    substitutions: int = 0
    insertions: int = 0
    deletions: int = 0
    word_count: int = 0

    # 문자 오류율 (CER)
    cer: float = 0.0
    char_substitutions: int = 0
    char_insertions: int = 0
    char_deletions: int = 0
    char_count: int = 0

    # 정확도 등급
    accuracy_grade: str = "N/A"

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "wer": self.wer,
            "substitutions": self.substitutions,
            "insertions": self.insertions,
            "deletions": self.deletions,
            "word_count": self.word_count,
            "cer": self.cer,
            "char_substitutions": self.char_substitutions,
            "char_insertions": self.char_insertions,
            "char_deletions": self.char_deletions,
            "char_count": self.char_count,
            "accuracy_grade": self.accuracy_grade,
        }


@dataclass
class TranscriptStats:
    """전사 통계 데이터 클래스"""

    # 기본 통계
    total_words: int = 0
    unique_words: int = 0
    total_characters: int = 0
    total_segments: int = 0

    # 시간 관련
    duration_seconds: float = 0.0
    words_per_minute: float = 0.0
    avg_segment_duration: float = 0.0

    # 화자 관련
    num_speakers: int = 0
    speaker_turns: Dict[str, int] = field(default_factory=dict)

    # 언어 관련 (한국어)
    korean_chars: int = 0
    korean_ratio: float = 0.0

    # 음소 속도 (Speaking Rate)
    speaking_rate_ratio: float = 0.0
    avg_syllables_per_word: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "total_words": self.total_words,
            "unique_words": self.unique_words,
            "total_characters": self.total_characters,
            "total_segments": self.total_segments,
            "duration_seconds": self.duration_seconds,
            "words_per_minute": self.words_per_minute,
            "avg_segment_duration": self.avg_segment_duration,
            "num_speakers": self.num_speakers,
            "speaker_turns": self.speaker_turns,
            "korean_chars": self.korean_chars,
            "korean_ratio": self.korean_ratio,
            "speaking_rate_ratio": self.speaking_rate_ratio,
            "avg_syllables_per_word": self.avg_syllables_per_word,
        }


@dataclass
class STTAccuracyResult:
    """STT 정확도 측정 결과 (확장版)"""

    file_name: str
    file_path: str

    # 메트릭
    confidence: ConfidenceMetrics
    speaker_accuracy: SpeakerAccuracyMetrics
    timestamp_accuracy: TimestampAccuracyMetrics
    sentence_metrics: SentenceMetrics
    voice_text_consistency: VoiceTextConsistencyMetrics
    errors: Optional[ErrorMetrics] = None
    stats: TranscriptStats = field(default_factory=TranscriptStats)

    # 전사 텍스트
    transcript_text: str = ""

    # 처리 상태
    has_reference: bool = False
    reference_source: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "file_name": self.file_name,
            "file_path": self.file_path,
            "confidence": self.confidence.to_dict(),
            "speaker_accuracy": self.speaker_accuracy.to_dict(),
            "timestamp_accuracy": self.timestamp_accuracy.to_dict(),
            "sentence_metrics": self.sentence_metrics.to_dict(),
            "voice_text_consistency": self.voice_text_consistency.to_dict(),
            "errors": self.errors.to_dict() if self.errors else None,
            "stats": self.stats.to_dict(),
            "transcript_text": self.transcript_text,
            "has_reference": self.has_reference,
            "reference_source": self.reference_source,
        }


class STTAccuracyService:
    """
    STT 정확도 분석 서비스 (확장版)

    WhisperX 결과에서 다양한 정확도 메트릭을 추출하고 계산합니다.
    """

    def __init__(self):
        """서비스 초기화"""
        self._korean_char_pattern = re.compile(r"[가-힣]")
        self._sentence_end_pattern = re.compile(r"[.!?。！？]+")

    def analyze_from_pipeline_result(
        self,
        file_path: str,
        pipeline_result: Any,
        reference_text: Optional[str] = None,
        duration_seconds: float = 0.0,
        audio_features: Optional[Dict[str, Any]] = None,
    ) -> STTAccuracyResult:
        """
        WhisperX 파이프라인 결과에서 STT 정확도를 종합 분석합니다.

        Args:
            file_path: 오디오 파일 경로
            pipeline_result: WhisperX PipelineResult 객체
            reference_text: 참조 텍스트 (WER/CER 계산용)
            duration_seconds: 오디오 길이 (초)
            audio_features: 오디오 특성 정보 (선택적)

        Returns:
            STTAccuracyResult 분석 결과
        """
        file_name = Path(file_path).name

        # 1. 신뢰도 분석
        confidence = self._analyze_confidence(pipeline_result)

        # 2. 화자별 인식 정확도
        speaker_accuracy = self._analyze_speaker_accuracy(pipeline_result)

        # 3. 타임스탬프 정확도
        timestamp_accuracy = self._analyze_timestamp_accuracy(pipeline_result)

        # 4. 문장 단위 분석
        sentence_metrics = self._analyze_sentences(pipeline_result)

        # 5. 음성-텍스트 일치도
        voice_text_consistency = self._analyze_voice_text_consistency(
            pipeline_result, audio_features
        )

        # 6. 전사 통계
        stats = self._calculate_stats(pipeline_result, duration_seconds)

        # 7. 오류율 분석 (참조 텍스트가 있는 경우)
        errors = None
        has_reference = reference_text is not None
        reference_source = ""

        if has_reference and reference_text:
            errors = self._calculate_error_rates(pipeline_result.text, reference_text)
            reference_source = "provided"

        return STTAccuracyResult(
            file_name=file_name,
            file_path=file_path,
            confidence=confidence,
            speaker_accuracy=speaker_accuracy,
            timestamp_accuracy=timestamp_accuracy,
            sentence_metrics=sentence_metrics,
            voice_text_consistency=voice_text_consistency,
            errors=errors,
            stats=stats,
            transcript_text=pipeline_result.text,
            has_reference=has_reference,
            reference_source=reference_source,
        )

    def _analyze_confidence(self, pipeline_result: Any) -> ConfidenceMetrics:
        """파이프라인 결과에서 신뢰도 점수를 추출합니다."""
        word_confidences = []
        segment_confidences = []

        # 단어 수준 신뢰도 추출
        for word in pipeline_result.word_segments:
            if "score" in word:
                word_confidences.append(word["score"])

        # 세그먼트 수준 신뢰도 추출
        for segment in pipeline_result.segments:
            if "score" in segment:
                segment_confidences.append(segment["score"])

        # 단어 수준 신뢰도 계산
        if word_confidences:
            avg_word_conf = float(np.mean(word_confidences))
            min_word_conf = float(np.min(word_confidences))
            max_word_conf = float(np.max(word_confidences))
            std_word_conf = float(np.std(word_confidences))
            low_conf_words = sum(1 for c in word_confidences if c < 0.5)
            low_conf_ratio = low_conf_words / len(word_confidences)

            # 신뢰도 분포 계산
            confidence_distribution = {
                "very_low": 0,
                "low": 0,
                "medium": 0,
                "high": 0,
                "very_high": 0,
            }
            for c in word_confidences:
                if c < 0.3:
                    confidence_distribution["very_low"] += 1
                elif c < 0.5:
                    confidence_distribution["low"] += 1
                elif c < 0.7:
                    confidence_distribution["medium"] += 1
                elif c < 0.9:
                    confidence_distribution["high"] += 1
                else:
                    confidence_distribution["very_high"] += 1
        else:
            avg_word_conf = 0.0
            min_word_conf = 0.0
            max_word_conf = 0.0
            std_word_conf = 0.0
            low_conf_words = 0
            low_conf_ratio = 0.0
            confidence_distribution = {}

        # 세그먼트 수준 신뢰도 계산
        if segment_confidences:
            avg_seg_conf = float(np.mean(segment_confidences))
            min_seg_conf = float(np.min(segment_confidences))
            max_seg_conf = float(np.max(segment_confidences))
        else:
            avg_seg_conf = 0.0
            min_seg_conf = 0.0
            max_seg_conf = 0.0

        # 전체 신뢰도 등급 결정
        if avg_word_conf >= 0.95:
            grade = "A"
        elif avg_word_conf >= 0.90:
            grade = "B"
        elif avg_word_conf >= 0.80:
            grade = "C"
        elif avg_word_conf >= 0.70:
            grade = "D"
        else:
            grade = "F"

        return ConfidenceMetrics(
            avg_word_confidence=avg_word_conf,
            min_word_confidence=min_word_conf,
            max_word_confidence=max_word_conf,
            low_confidence_words=low_conf_words,
            low_confidence_ratio=low_conf_ratio,
            std_word_confidence=std_word_conf,
            avg_segment_confidence=avg_seg_conf,
            min_segment_confidence=min_seg_conf,
            max_segment_confidence=max_seg_conf,
            overall_confidence_grade=grade,
            confidence_distribution=confidence_distribution,
        )

    def _analyze_speaker_accuracy(self, pipeline_result: Any) -> SpeakerAccuracyMetrics:
        """화자별 인식 정확도를 분석합니다."""
        segments = pipeline_result.segments
        word_segments = pipeline_result.word_segments

        speakers = pipeline_result.speakers
        speaker_count = len(speakers)

        # 화자별 단어 수 집계
        speaker_words = {s: 0 for s in speakers}
        speaker_duration = {s: 0.0 for s in speakers}
        speaker_confidence = {s: [] for s in speakers}

        # 단어별 화자 할당
        for word in word_segments:
            if "speaker" in word and "score" in word:
                spk = word["speaker"]
                if spk in speaker_words:
                    speaker_words[spk] += 1
                    speaker_confidence[spk].append(word["score"])

        # 세그먼트별 화자 지속시간
        for seg in segments:
            if "speaker" in seg:
                spk = seg["speaker"]
                duration = seg.get("end", 0) - seg.get("start", 0)
                if spk in speaker_duration:
                    speaker_duration[spk] += duration

        # 화자별 평균 신뢰도 계산
        for spk in speaker_confidence:
            if speaker_confidence[spk]:
                speaker_confidence[spk] = float(np.mean(speaker_confidence[spk]))
            else:
                speaker_confidence[spk] = 0.0

        # 화자 전환 횟수 및 정확도
        speaker_switches = 0
        prev_speaker = None
        for seg in segments:
            curr_speaker = seg.get("speaker")
            if prev_speaker is not None and curr_speaker != prev_speaker:
                speaker_switches += 1
            prev_speaker = curr_speaker

        # 화자 전환 정확도 (세그먼트 수 대비 전환 수)
        total_segments = len(segments)
        speaker_switch_accuracy = 1.0
        if total_segments > 1:
            # 적절한 화자 전환 비율 계산
            expected_switches = max(1, total_segments // 10)  # 대략적인 기대치
            if speaker_switches > 0:
                speaker_switch_accuracy = min(1.0, expected_switches / speaker_switches)

        # 화자 균일성 (duration 표준편차 기반)
        speaker_uniformity = 0.0
        durations = list(speaker_duration.values())
        if len(durations) > 1:
            mean_duration = np.mean(durations)
            if mean_duration > 0:
                std_duration = np.std(durations)
                speaker_uniformity = max(0.0, 1.0 - (std_duration / mean_duration))

        return SpeakerAccuracyMetrics(
            speaker_count=speaker_count,
            speaker_words=speaker_words,
            speaker_duration=speaker_duration,
            speaker_confidence=speaker_confidence,
            speaker_switches=speaker_switches,
            speaker_switch_accuracy=speaker_switch_accuracy,
            speaker_uniformity=speaker_uniformity,
        )

    def _analyze_timestamp_accuracy(self, pipeline_result: Any) -> TimestampAccuracyMetrics:
        """타임스탬프 정확도를 분석합니다."""
        word_segments = pipeline_result.word_segments

        if not word_segments:
            return TimestampAccuracyMetrics()

        # 단어 지속시간 분석
        word_durations = []
        for word in word_segments:
            if "start" in word and "end" in word:
                duration = word["end"] - word["start"]
                word_durations.append(duration)

        if word_durations:
            avg_word_duration = float(np.mean(word_durations))
            min_word_duration = float(np.min(word_durations))
            max_word_duration = float(np.max(word_durations))
        else:
            avg_word_duration = 0.0
            min_word_duration = 0.0
            max_word_duration = 0.0

        # 타임스탬프 갭 분석
        timestamp_gaps = []
        prev_end = None
        for word in word_segments:
            if "start" in word and prev_end is not None:
                gap = word["start"] - prev_end
                if gap > 0:
                    timestamp_gaps.append(gap)
            if "end" in word:
                prev_end = word["end"]

        if timestamp_gaps:
            avg_timestamp_gap = float(np.mean(timestamp_gaps))
            max_timestamp_gap = float(np.max(timestamp_gaps))
        else:
            avg_timestamp_gap = 0.0
            max_timestamp_gap = 0.0

        # 타임스탬프 정밀도 (100ms/200ms 기준)
        timestamps_within_100ms = sum(1 for d in word_durations if 0.05 <= d <= 0.2)
        timestamps_within_200ms = sum(1 for d in word_durations if 0.05 <= d <= 0.4)
        timestamp_precision_ratio = (
            timestamps_within_100ms / len(word_durations) if word_durations else 0.0
        )

        return TimestampAccuracyMetrics(
            avg_word_duration=avg_word_duration,
            min_word_duration=min_word_duration,
            max_word_duration=max_word_duration,
            timestamp_gaps=timestamp_gaps,
            avg_timestamp_gap=avg_timestamp_gap,
            max_timestamp_gap=max_timestamp_gap,
            timestamps_within_100ms=timestamps_within_100ms,
            timestamps_within_200ms=timestamps_within_200ms,
            timestamp_precision_ratio=timestamp_precision_ratio,
        )

    def _analyze_sentences(self, pipeline_result: Any) -> SentenceMetrics:
        """문장 단위 분석을 수행합니다."""
        text = pipeline_result.text
        segments = pipeline_result.segments

        # 문장 분할
        sentences = self._sentence_end_pattern.split(text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]

        total_sentences = len(sentences)

        if total_sentences > 0:
            # 평균 문장 길이
            avg_sentence_length = float(np.mean([len(s) for s in sentences]))
            avg_words_per_sentence = float(np.mean([len(s.split()) for s in sentences]))
        else:
            avg_sentence_length = 0.0
            avg_words_per_sentence = 0.0

        # 문장 완결성 (끝나는 문장 부호로 판단)
        complete_sentences = 0
        missing_punctuation = 0
        for seg in segments:
            seg_text = seg.get("text", "").strip()
            if seg_text:
                if self._sentence_end_pattern.search(seg_text[-1] if seg_text else ""):
                    complete_sentences += 1
                else:
                    missing_punctuation += 1

        # 문장 완결률
        total_segments = len(segments)
        sentence_completion_ratio = (
            complete_sentences / total_segments if total_segments > 0 else 0.0
        )

        # 문장 부호 정확도
        punctuation_accuracy = sentence_completion_ratio

        return SentenceMetrics(
            total_sentences=total_sentences,
            avg_sentence_length=avg_sentence_length,
            avg_words_per_sentence=avg_words_per_sentence,
            complete_sentences=complete_sentences,
            sentence_completion_ratio=sentence_completion_ratio,
            punctuation_accuracy=punctuation_accuracy,
            missing_punctuation=missing_punctuation,
        )

    def _analyze_voice_text_consistency(
        self, pipeline_result: Any, audio_features: Optional[Dict[str, Any]]
    ) -> VoiceTextConsistencyMetrics:
        """음성-텍스트 일치도를 분석합니다."""
        segments = pipeline_result.segments
        word_segments = pipeline_result.word_segments

        # 음성-텍스트 상관관수 (단어 속도 vs 음성 길이)
        if word_segments and len(word_segments) > 1:
            word_durations = []
            word_gap = []
            prev_end = None
            for word in word_segments:
                if "start" in word and "end" in word:
                    word_durations.append(word["end"] - word["start"])
                    if prev_end is not None:
                        word_gap.append(word["start"] - prev_end)
                    prev_end = word["end"]

            if word_durations and word_gap:
                avg_duration = np.mean(word_durations)
                avg_gap = np.mean(word_gap)

                # 리듬 일치도 (지속시간과 갭의 균형)
                if avg_gap > 0:
                    rhythm_consistency = min(1.0, avg_duration / (avg_gap * 2))
                else:
                    rhythm_consistency = 0.5

                # 일시정지 분포 일치
                gap_std = np.std(word_gap)
                pause_distribution_match = max(0.0, 1.0 - (gap_std / avg_gap if avg_gap > 0 else 1))
            else:
                rhythm_consistency = 0.0
                pause_distribution_match = 0.0

            # 음성-텍스트 상관관수 (간단 계산)
            voice_text_correlation = (rhythm_consistency + pause_distribution_match) / 2
        else:
            voice_text_correlation = 0.0
            rhythm_consistency = 0.0
            pause_distribution_match = 0.0

        # 오디오 특성 일치도 (제공된 경우)
        acoustic_feature_match = 0.0
        if audio_features:
            # 오디오 특성이 제공되면 이를 활용한 일치도 계산
            mfcc_consistency = audio_features.get("mfcc_consistency", 0.0)
            pitch_variance = audio_features.get("pitch_variance", 0.0)
            acoustic_feature_match = (mfcc_consistency + pitch_variance) / 2
        else:
            # 세그먼트 길이 변동성을 대리 지표로 사용
            if segments:
                seg_lengths = [s.get("end", 0) - s.get("start", 0) for s in segments]
                if seg_lengths:
                    seg_std = np.std(seg_lengths)
                    seg_mean = np.mean(seg_lengths)
                    acoustic_feature_match = max(
                        0.0, 1.0 - (seg_std / seg_mean if seg_mean > 0 else 1)
                    )

        # 전체 일치도 점수
        overall_consistency_score = (
            voice_text_correlation * 0.4
            + acoustic_feature_match * 0.3
            + rhythm_consistency * 0.15
            + pause_distribution_match * 0.15
        )

        # 등급 결정
        if overall_consistency_score >= 0.8:
            grade = "A"
        elif overall_consistency_score >= 0.6:
            grade = "B"
        elif overall_consistency_score >= 0.4:
            grade = "C"
        elif overall_consistency_score >= 0.2:
            grade = "D"
        else:
            grade = "F"

        return VoiceTextConsistencyMetrics(
            voice_text_correlation=voice_text_correlation,
            acoustic_feature_match=acoustic_feature_match,
            rhythm_consistency=rhythm_consistency,
            pause_distribution_match=pause_distribution_match,
            overall_consistency_score=overall_consistency_score,
            consistency_grade=grade,
        )

    def _calculate_stats(self, pipeline_result: Any, duration_seconds: float) -> TranscriptStats:
        """전사 통계를 계산합니다."""
        text = pipeline_result.text
        segments = pipeline_result.segments

        # 기본 통계
        words = self._tokenize_korean(text)
        total_words = len(words)
        unique_words = len(set(words))
        total_characters = len(text.replace(" ", ""))
        total_segments = len(segments)

        # 시간 관련
        words_per_minute = 0.0
        if duration_seconds > 0:
            words_per_minute = (total_words / duration_seconds) * 60

        avg_segment_duration = 0.0
        if segments:
            total_seg_duration = sum(seg.get("end", 0) - seg.get("start", 0) for seg in segments)
            avg_segment_duration = total_seg_duration / len(segments)

        # 화자 관련
        num_speakers = len(pipeline_result.speakers)
        speaker_turns = {}
        for speaker, stats in pipeline_result.speaker_stats.items():
            speaker_turns[speaker] = stats.get("turn_count", 0)

        # 한국어 비율
        korean_chars = len(self._korean_char_pattern.findall(text))
        korean_ratio = korean_chars / total_characters if total_characters > 0 else 0.0

        # 음소 속도 (Speaking Rate)
        speaking_rate_ratio = 0.0
        if duration_seconds > 0 and total_words > 0:
            # 대략적인 음소 속도 (문자/초)
            speaking_rate = total_characters / duration_seconds
            # 정상 범위: 10-20 char/sec (한국어 기준)
            if 10 <= speaking_rate <= 20:
                speaking_rate_ratio = 1.0
            elif speaking_rate < 10:
                speaking_rate_ratio = speaking_rate / 10
            else:
                speaking_rate_ratio = max(0.0, 1.0 - (speaking_rate - 20) / 30)

        # 평균 음절 수 (한국어 복합어 고려)
        avg_syllables_per_word = 0.0
        if total_words > 0:
            # 한글 음절 수 추정 (간단 계산)
            korean_syllables = sum(
                len([c for c in w if ord("가") <= ord(c) <= ord("힣")]) for w in words
            )
            avg_syllables_per_word = korean_syllables / total_words

        return TranscriptStats(
            total_words=total_words,
            unique_words=unique_words,
            total_characters=total_characters,
            total_segments=total_segments,
            duration_seconds=duration_seconds,
            words_per_minute=words_per_minute,
            avg_segment_duration=avg_segment_duration,
            num_speakers=num_speakers,
            speaker_turns=speaker_turns,
            korean_chars=korean_chars,
            korean_ratio=korean_ratio,
            speaking_rate_ratio=speaking_rate_ratio,
            avg_syllables_per_word=avg_syllables_per_word,
        )

    def _calculate_error_rates(self, hypothesis: str, reference: str) -> ErrorMetrics:
        """WER(단어 오류율)과 CER(문자 오류율)을 계산합니다."""
        # 텍스트 정제
        hyp_words = self._tokenize_korean(hypothesis)
        ref_words = self._tokenize_korean(reference)

        # WER 계산
        sub_w, ins_w, del_w = self._levenshtein_stats(hyp_words, ref_words)
        total_words = len(ref_words) if ref_words else 1
        wer = (sub_w + ins_w + del_w) / total_words if total_words > 0 else 0.0

        # CER 계산
        hyp_chars = list(hypothesis.replace(" ", ""))
        ref_chars = list(reference.replace(" ", ""))
        sub_c, ins_c, del_c = self._levenshtein_stats(hyp_chars, ref_chars)
        total_chars = len(ref_chars) if ref_chars else 1
        cer = (sub_c + ins_c + del_c) / total_chars if total_chars > 0 else 0.0

        # 정확도 등급
        if wer < 0.05:
            grade = "A"
        elif wer < 0.10:
            grade = "B"
        elif wer < 0.15:
            grade = "C"
        elif wer < 0.20:
            grade = "D"
        else:
            grade = "F"

        return ErrorMetrics(
            wer=wer,
            substitutions=sub_w,
            insertions=ins_w,
            deletions=del_w,
            word_count=total_words,
            cer=cer,
            char_substitutions=sub_c,
            char_insertions=ins_c,
            char_deletions=del_c,
            char_count=total_chars,
            accuracy_grade=grade,
        )

    def _tokenize_korean(self, text: str) -> List[str]:
        """한국어 텍스트를 단어 단위로 분할합니다."""
        words = text.split()
        return [w.strip() for w in words if w.strip()]

    def _levenshtein_stats(self, source: List[str], target: List[str]) -> Tuple[int, int, int]:
        """Levenshtein 거리를 계산하고 대체/삽입/삭제 수를 반환합니다."""
        m, n = len(source), len(target)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if source[i - 1] == target[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i - 1][j - 1],  # 대체
                        dp[i][j - 1],  # 삽입
                        dp[i - 1][j],  # 삭제
                    )

        # 백트래킹
        substitutions = insertions = deletions = 0
        i, j = m, n

        while i > 0 or j > 0:
            if i > 0 and j > 0 and source[i - 1] == target[j - 1]:
                i -= 1
                j -= 1
            elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
                substitutions += 1
                i -= 1
                j -= 1
            elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
                insertions += 1
                j -= 1
            elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
                deletions += 1
                i -= 1
            else:
                break

        return substitutions, insertions, deletions


class STTAccuracyCalculator:
    """여러 파일의 STT 정확도를 계산하고 집계하는 유틸리티 클래스"""

    def __init__(self):
        self.service = STTAccuracyService()

    def calculate_batch_summary(self, results: List[STTAccuracyResult]) -> Dict[str, Any]:
        """여러 STT 정확도 결과의 집계를 계산합니다."""
        if not results:
            return {}

        # 신뢰도 집계
        avg_confidence = np.mean([r.confidence.avg_word_confidence for r in results])
        min_confidence = np.min([r.confidence.min_word_confidence for r in results])

        # 화자 인식 집계
        avg_speaker_count = np.mean([r.speaker_accuracy.speaker_count for r in results])
        avg_speaker_uniformity = np.mean([r.speaker_accuracy.speaker_uniformity for r in results])

        # 타임스탬프 정밀도 집계
        avg_timestamp_precision = np.mean(
            [r.timestamp_accuracy.timestamp_precision_ratio for r in results]
        )

        # 문장 완결성 집계
        avg_sentence_completion = np.mean(
            [r.sentence_metrics.sentence_completion_ratio for r in results]
        )

        # 음성-텍스트 일치도 집계
        avg_consistency = np.mean(
            [r.voice_text_consistency.overall_consistency_score for r in results]
        )

        # WER/CER 집계
        results_with_ref = [r for r in results if r.errors is not None]
        if results_with_ref:
            avg_wer = np.mean([r.errors.wer for r in results_with_ref])
            avg_cer = np.mean([r.errors.cer for r in results_with_ref])
            ref_count = len(results_with_ref)
        else:
            avg_wer = avg_cer = 0.0
            ref_count = 0

        # 등급 분포
        grade_distribution = {}
        for r in results:
            grade = r.confidence.overall_confidence_grade
            grade_distribution[grade] = grade_distribution.get(grade, 0) + 1

        # 전사 통계 집계
        total_words = sum(r.stats.total_words for r in results)
        total_duration = sum(r.stats.duration_seconds for r in results)
        avg_wpm = np.mean(
            [r.stats.words_per_minute for r in results if r.stats.words_per_minute > 0]
        )

        return {
            "total_files": len(results),
            "files_with_reference": ref_count,
            "avg_confidence": float(avg_confidence),
            "min_confidence": float(min_confidence),
            "avg_speaker_count": float(avg_speaker_count),
            "avg_speaker_uniformity": float(avg_speaker_uniformity),
            "avg_timestamp_precision": float(avg_timestamp_precision),
            "avg_sentence_completion": float(avg_sentence_completion),
            "avg_voice_text_consistency": float(avg_consistency),
            "avg_wer": float(avg_wer),
            "avg_cer": float(avg_cer),
            "grade_distribution": grade_distribution,
            "total_words": total_words,
            "total_duration_minutes": total_duration / 60,
            "avg_words_per_minute": float(avg_wpm),
        }
