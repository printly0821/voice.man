"""
포렌식 차트 시각화 서비스

가스라이팅 패턴, 감정 분석, 스트레스 레벨 등의 시각화 데이터 생성
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum


class ChartType(str, Enum):
    """차트 유형"""

    TIMELINE = "timeline"
    BAR = "bar"
    PIE = "pie"
    LINE = "line"
    HEATMAP = "heatmap"
    RADAR = "radar"
    SANKEY = "sankey"


@dataclass
class ChartData:
    """차트 데이터"""

    chart_type: ChartType
    title: str
    data: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None


class ForensicChartService:
    """
    포렌식 차트 시각화 서비스

    FEATURES:
    - 가스라이팅 패턴 차트
    - 감정 분석 차트
    - 스트레스 레벨 차트
    - 화자 비교 차트
    - 시계열 히트맵

    EXAMPLE:
        ```python
        service = ForensicChartService()

        # 가스라이팅 패턴 바 차트 생성
        chart = service.create_gaslighting_bar_chart({
            "부정": 15,
            "전가": 10,
            "축소": 8,
            "혼란": 5
        })
        ```
    """

    def __init__(self) -> None:
        """포렌식 차트 서비스 초기화"""
        self.data_store: Dict[str, Any] = {}

    def create_gaslighting_bar_chart(
        self,
        pattern_counts: Dict[str, int],
        title: str = "가스라이팅 패턴 분포",
    ) -> ChartData:
        """
        가스라이팅 패턴 바 차트 생성

        Args:
            pattern_counts: 패턴별 빈도수
            title: 차트 제목

        Returns:
            바 차트 데이터
        """
        return ChartData(
            chart_type=ChartType.BAR,
            title=title,
            data={
                "labels": list(pattern_counts.keys()),
                "values": list(pattern_counts.values()),
                "colors": self._get_pattern_colors(list(pattern_counts.keys())),
            },
        )

    def create_emotion_pie_chart(
        self,
        emotion_counts: Dict[str, int],
        title: str = "감정 분포",
    ) -> ChartData:
        """
        감정 파이 차트 생성

        Args:
            emotion_counts: 감정별 빈도수
            title: 차트 제목

        Returns:
            파이 차트 데이터
        """
        return ChartData(
            chart_type=ChartType.PIE,
            title=title,
            data={
                "labels": list(emotion_counts.keys()),
                "values": list(emotion_counts.values()),
                "colors": self._get_emotion_colors(list(emotion_counts.keys())),
            },
        )

    def create_stress_line_chart(
        self,
        stress_timeline: List[Dict[str, Any]],
        title: str = "스트레스 레벨 변화",
    ) -> ChartData:
        """
        스트레스 레벨 라인 차트 생성

        Args:
            stress_timeline: 시간별 스트레스 레벨 데이터
            title: 차트 제목

        Returns:
            라인 차트 데이터
        """
        return ChartData(
            chart_type=ChartType.LINE,
            title=title,
            data={
                "timestamps": [d["timestamp"] for d in stress_timeline],
                "values": [d["stress_level"] for d in stress_timeline],
                "speakers": [d.get("speaker", "") for d in stress_timeline],
            },
        )

    def create_speaker_comparison_radar(
        self,
        speaker_data: Dict[str, Dict[str, float]],
        title: str = "화자별 특성 비교",
    ) -> ChartData:
        """
        화자별 특성 비교 레이더 차트 생성

        Args:
            speaker_data: 화자별 특성 데이터
            title: 차트 제목

        Returns:
            레이더 차트 데이터
        """
        # 모든 화자의 공통 키 찾기
        all_keys = set()
        for data in speaker_data.values():
            all_keys.update(data.keys())

        labels = list(all_keys)
        datasets = []

        for speaker, data in speaker_data.items():
            values = [data.get(k, 0) for k in labels]
            datasets.append({"label": speaker, "values": values})

        return ChartData(
            chart_type=ChartType.RADAR,
            title=title,
            data={
                "labels": labels,
                "datasets": datasets,
            },
        )

    def create_emotion_sankey(
        self,
        emotion_transitions: Dict[str, Dict[str, int]],
        title: str = "감정 전이 흐름",
    ) -> ChartData:
        """
        감정 전이 샌키 다이어그램 생성

        Args:
            emotion_transitions: 감정별 전이 횟수
            title: 차트 제목

        Returns:
            샌키 다이어그램 데이터
        """
        nodes = list(
            set(
                list(emotion_transitions.keys())
                + [k for d in emotion_transitions.values() for k in d.keys()]
            )
        )

        links = []
        for source, targets in emotion_transitions.items():
            for target, count in targets.items():
                if count > 0:
                    links.append(
                        {
                            "source": nodes.index(source),
                            "target": nodes.index(target),
                            "value": count,
                        }
                    )

        return ChartData(
            chart_type=ChartType.SANKEY,
            title=title,
            data={"nodes": nodes, "links": links},
        )

    def create_timeline_heatmap(
        self,
        timeline_data: List[Dict[str, Any]],
        title: str = "타임라인 히트맵",
    ) -> ChartData:
        """
        타임라인 히트맵 생성

        Args:
            timeline_data: 시계열 데이터
            title: 차트 제목

        Returns:
            히트맵 데이터
        """
        # 시간대별, 날짜별 집계
        time_slots = {}
        for data in timeline_data:
            timestamp = datetime.fromisoformat(data["timestamp"])
            date_key = timestamp.strftime("%Y-%m-%d")
            hour_key = timestamp.hour

            if date_key not in time_slots:
                time_slots[date_key] = {}

            if hour_key not in time_slots[date_key]:
                time_slots[date_key][hour_key] = 0

            time_slots[date_key][hour_key] += 1

        return ChartData(
            chart_type=ChartType.HEATMAP,
            title=title,
            data={"time_slots": time_slots},
        )

    def create_comprehensive_dashboard(
        self,
        forensic_data: Dict[str, Any],
    ) -> List[ChartData]:
        """
        종합 포렌식 대시보드 생성

        Args:
            forensic_data: 포렌식 분석 데이터

        Returns:
            차트 데이터 리스트
        """
        charts = []

        # 가스라이팅 패턴 바 차트
        if "gaslighting_patterns" in forensic_data:
            charts.append(self.create_gaslighting_bar_chart(forensic_data["gaslighting_patterns"]))

        # 감정 파이 차트
        if "emotion_counts" in forensic_data:
            charts.append(self.create_emotion_pie_chart(forensic_data["emotion_counts"]))

        # 스트레스 라인 차트
        if "stress_timeline" in forensic_data:
            charts.append(self.create_stress_line_chart(forensic_data["stress_timeline"]))

        # 화자 비교 레이더 차트
        if "speaker_comparison" in forensic_data:
            charts.append(self.create_speaker_comparison_radar(forensic_data["speaker_comparison"]))

        return charts

    def _get_pattern_colors(self, patterns: List[str]) -> List[str]:
        """패턴별 색상 반환"""
        color_map = {
            "부정": "#FF4444",
            "전가": "#FF8800",
            "축소": "#FFCC00",
            "혼란": "#9966FF",
        }
        return [color_map.get(p, "#CCCCCC") for p in patterns]

    def _get_emotion_colors(self, emotions: List[str]) -> List[str]:
        """감정별 색상 반환"""
        color_map = {
            "기쁨": "#FFD700",
            "슬픔": "#4169E1",
            "분노": "#DC143C",
            "공포": "#8B008B",
            "혐오": "#228B22",
            "놀람": "#FF6347",
            "중립": "#A9A9A9",
        }
        return [color_map.get(e, "#CCCCCC") for e in emotions]

    def export_chart_config(self, chart: ChartData, format: str = "json") -> str:
        """
        차트 설정 내보내기

        Args:
            chart: 차트 데이터
            format: 내보내기 형식 ('json', 'dict')

        Returns:
            내보낸 설정 문자열 또는 딕셔너리
        """
        config = {
            "type": chart.chart_type.value,
            "title": chart.title,
            "data": chart.data,
            "metadata": chart.metadata or {},
        }

        if format == "json":
            import json

            return json.dumps(config, ensure_ascii=False, indent=2)
        elif format == "dict":
            return config
        else:
            raise ValueError(f"Unsupported format: {format}")
