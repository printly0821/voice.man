"""
타임라인 시각화 서비스

가스라이팅 패턴, 감정 변화, 음성 특성의 시계열 시각화 제공
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass


@dataclass
class TimelineEvent:
    """타임라인 이벤트"""

    timestamp: datetime
    event_type: str
    speaker: str
    description: str
    metadata: Dict[str, Any]


@dataclass
class TimelineSegment:
    """타임라인 세그먼트"""

    start_time: datetime
    end_time: datetime
    speaker: str
    text: str
    emotion: Optional[str] = None
    stress_level: Optional[float] = None
    gaslighting_pattern: Optional[str] = None


class TimelineService:
    """
    타임라인 시각화 서비스

    FEATURES:
    - 시계열 데이터 처리
    - 이벤트 기반 타임라인 생성
    - 세그먼트 그룹화
    - 필터링 및 페이지네이션

    EXAMPLE:
        ```python
        service = TimelineService()

        # 세그먼트 추가
        service.add_segment(
            start_time=datetime(2024, 1, 1, 10, 0),
            end_time=datetime(2024, 1, 1, 10, 5),
            speaker="SPEAKER_00",
            text="안녕하세요",
            emotion="neutral"
        )

        # 타임라인 생성
        timeline = service.generate_timeline(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31)
        )
        ```
    """

    def __init__(self) -> None:
        """타임라인 서비스 초기화"""
        self.segments: List[TimelineSegment] = []
        self.events: List[TimelineEvent] = []

    def add_segment(
        self,
        start_time: datetime,
        end_time: datetime,
        speaker: str,
        text: str,
        emotion: Optional[str] = None,
        stress_level: Optional[float] = None,
        gaslighting_pattern: Optional[str] = None,
    ) -> None:
        """
        타임라인 세그먼트 추가

        Args:
            start_time: 시작 시간
            end_time: 종료 시간
            speaker: 화자 ID
            text: 전사 텍스트
            emotion: 감정 (선택 사항)
            stress_level: 스트레스 레벨 (선택 사항)
            gaslighting_pattern: 가스라이팅 패턴 (선택 사항)
        """
        segment = TimelineSegment(
            start_time=start_time,
            end_time=end_time,
            speaker=speaker,
            text=text,
            emotion=emotion,
            stress_level=stress_level,
            gaslighting_pattern=gaslighting_pattern,
        )
        self.segments.append(segment)

    def add_event(
        self,
        timestamp: datetime,
        event_type: str,
        speaker: str,
        description: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        타임라인 이벤트 추가

        Args:
            timestamp: 이벤트 시간
            event_type: 이벤트 유형
            speaker: 화자 ID
            description: 설명
            metadata: 추가 메타데이터
        """
        event = TimelineEvent(
            timestamp=timestamp,
            event_type=event_type,
            speaker=speaker,
            description=description,
            metadata=metadata or {},
        )
        self.events.append(event)

    def generate_timeline(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        speaker: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        타임라인 생성

        Args:
            start_date: 시작 날짜 (선택 사항)
            end_date: 종료 날짜 (선택 사항)
            speaker: 화자 필터 (선택 사항)

        Returns:
            타임라인 데이터
            ```python
            {
                "segments": [...],
                "events": [...],
                "statistics": {...}
            }
            ```
        """
        # 세그먼트 필터링
        filtered_segments = self._filter_segments(
            start_date=start_date, end_date=end_date, speaker=speaker
        )

        # 이벤트 필터링
        filtered_events = self._filter_events(
            start_date=start_date, end_date=end_date, speaker=speaker
        )

        # 통계 계산
        statistics = self._calculate_statistics(filtered_segments, filtered_events)

        return {
            "segments": [self._segment_to_dict(s) for s in filtered_segments],
            "events": [self._event_to_dict(e) for e in filtered_events],
            "statistics": statistics,
        }

    def get_emotion_timeline(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        speaker: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        감정 타임라인 조회

        Args:
            start_date: 시작 날짜 (선택 사항)
            end_date: 종료 날짜 (선택 사항)
            speaker: 화자 필터 (선택 사항)

        Returns:
            감정 타임라인 데이터
        """
        filtered_segments = self._filter_segments(
            start_date=start_date, end_date=end_date, speaker=speaker
        )

        timeline = []
        for segment in filtered_segments:
            if segment.emotion:
                timeline.append(
                    {
                        "timestamp": segment.start_time.isoformat(),
                        "emotion": segment.emotion,
                        "speaker": segment.speaker,
                    }
                )

        return timeline

    def get_stress_timeline(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        speaker: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        스트레스 타임라인 조회

        Args:
            start_date: 시작 날짜 (선택 사항)
            end_date: 종료 날짜 (선택 사항)
            speaker: 화자 필터 (선택 사항)

        Returns:
            스트레스 타임라인 데이터
        """
        filtered_segments = self._filter_segments(
            start_date=start_date, end_date=end_date, speaker=speaker
        )

        timeline = []
        for segment in filtered_segments:
            if segment.stress_level is not None:
                timeline.append(
                    {
                        "timestamp": segment.start_time.isoformat(),
                        "stress_level": segment.stress_level,
                        "speaker": segment.speaker,
                    }
                )

        return timeline

    def get_gaslighting_timeline(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        pattern_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        가스라이팅 패턴 타임라인 조회

        Args:
            start_date: 시작 날짜 (선택 사항)
            end_date: 종료 날짜 (선택 사항)
            pattern_type: 패턴 유형 필터 (선택 사항)

        Returns:
            가스라이팅 패턴 타임라인 데이터
        """
        filtered_segments = self._filter_segments(start_date=start_date, end_date=end_date)

        timeline = []
        for segment in filtered_segments:
            if segment.gaslighting_pattern:
                if pattern_type is None or segment.gaslighting_pattern == pattern_type:
                    timeline.append(
                        {
                            "timestamp": segment.start_time.isoformat(),
                            "pattern": segment.gaslighting_pattern,
                            "speaker": segment.speaker,
                            "text": segment.text,
                        }
                    )

        return timeline

    def get_speaker_statistics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        화자별 통계 조회

        Args:
            start_date: 시작 날짜 (선택 사항)
            end_date: 종료 날짜 (선택 사항)

        Returns:
            화자별 통계
        """
        filtered_segments = self._filter_segments(start_date=start_date, end_date=end_date)

        speaker_stats: Dict[str, Dict[str, Any]] = {}

        for segment in filtered_segments:
            if segment.speaker not in speaker_stats:
                speaker_stats[segment.speaker] = {
                    "segment_count": 0,
                    "total_duration": 0.0,
                    "emotions": {},
                    "avg_stress": 0.0,
                }

            stats = speaker_stats[segment.speaker]
            stats["segment_count"] += 1
            stats["total_duration"] += (segment.end_time - segment.start_time).total_seconds()

            if segment.emotion:
                stats["emotions"][segment.emotion] = stats["emotions"].get(segment.emotion, 0) + 1

            if segment.stress_level is not None:
                stats["avg_stress"] = (
                    stats["avg_stress"] * (stats["segment_count"] - 1) + segment.stress_level
                ) / stats["segment_count"]

        return speaker_stats

    def _filter_segments(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        speaker: Optional[str] = None,
    ) -> List[TimelineSegment]:
        """세그먼트 필터링"""
        filtered = self.segments

        if start_date:
            filtered = [s for s in filtered if s.start_time >= start_date]

        if end_date:
            filtered = [s for s in filtered if s.end_time <= end_date]

        if speaker:
            filtered = [s for s in filtered if s.speaker == speaker]

        return filtered

    def _filter_events(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        speaker: Optional[str] = None,
    ) -> List[TimelineEvent]:
        """이벤트 필터링"""
        filtered = self.events

        if start_date:
            filtered = [e for e in filtered if e.timestamp >= start_date]

        if end_date:
            filtered = [e for e in filtered if e.timestamp <= end_date]

        if speaker:
            filtered = [e for e in filtered if e.speaker == speaker]

        return filtered

    def _calculate_statistics(
        self,
        segments: List[TimelineSegment],
        events: List[TimelineEvent],
    ) -> Dict[str, Any]:
        """통계 계산"""
        total_duration = sum((s.end_time - s.start_time).total_seconds() for s in segments)

        speakers = set(s.speaker for s in segments)

        emotion_counts: Dict[str, int] = {}
        for segment in segments:
            if segment.emotion:
                emotion_counts[segment.emotion] = emotion_counts.get(segment.emotion, 0) + 1

        return {
            "total_segments": len(segments),
            "total_events": len(events),
            "total_duration_seconds": total_duration,
            "unique_speakers": list(speakers),
            "emotion_distribution": emotion_counts,
        }

    def _segment_to_dict(self, segment: TimelineSegment) -> Dict[str, Any]:
        """세그먼트를 딕셔너리로 변환"""
        return {
            "start_time": segment.start_time.isoformat(),
            "end_time": segment.end_time.isoformat(),
            "speaker": segment.speaker,
            "text": segment.text,
            "emotion": segment.emotion,
            "stress_level": segment.stress_level,
            "gaslighting_pattern": segment.gaslighting_pattern,
        }

    def _event_to_dict(self, event: TimelineEvent) -> Dict[str, Any]:
        """이벤트를 딕셔너리로 변환"""
        return {
            "timestamp": event.timestamp.isoformat(),
            "event_type": event.event_type,
            "speaker": event.speaker,
            "description": event.description,
            "metadata": event.metadata,
        }

    def clear(self) -> None:
        """모든 데이터 삭제"""
        self.segments.clear()
        self.events.clear()
