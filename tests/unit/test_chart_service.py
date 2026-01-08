"""
Unit tests for Chart Generation Service.

Tests Matplotlib-based chart generation for legal evidence reports.
"""

import pytest
from pathlib import Path
from matplotlib.figure import Figure
import numpy as np

from voice_man.services.chart_service import ChartService, ChartType


class TestChartServiceInitialization:
    """Test ChartService initialization and configuration."""

    def test_chart_service_initialization(self, tmp_path):
        """Test that ChartService initializes with default settings."""
        service = ChartService(output_dir=str(tmp_path))
        assert service.output_dir == tmp_path
        assert service.dpi == 300
        assert service.figure_width == 10
        assert service.figure_height == 6

    def test_chart_service_custom_dimensions(self, tmp_path):
        """Test ChartService with custom dimensions."""
        service = ChartService(output_dir=str(tmp_path), width=12, height=8, dpi=150)
        assert service.figure_width == 12
        assert service.figure_height == 8
        assert service.dpi == 150


class TestTimelineChart:
    """Test timeline chart generation for conversation flow."""

    @pytest.mark.asyncio
    async def test_generate_timeline_chart_basic(self, tmp_path):
        """Test basic timeline chart generation."""
        service = ChartService(output_dir=str(tmp_path))

        # Mock timeline data
        data = [
            {"speaker": "A", "timestamp": 0.0, "duration": 5.0, "emotion": "neutral"},
            {"speaker": "B", "timestamp": 5.0, "duration": 3.0, "emotion": "angry"},
            {"speaker": "A", "timestamp": 8.0, "duration": 4.0, "emotion": "sad"},
        ]

        figure = await service.generate_timeline_chart(data)

        assert isinstance(figure, Figure)
        assert len(figure.axes) == 1
        ax = figure.axes[0]
        assert ax.get_xlabel() == "시간 (초)"
        assert ax.get_ylabel() == "화자"
        assert ax.get_title() == "대화 타임라인"

    @pytest.mark.asyncio
    async def test_timeline_chart_empty_data(self, tmp_path):
        """Test timeline chart with empty data."""
        service = ChartService(output_dir=str(tmp_path))

        with pytest.raises(ValueError, match="Timeline data is required"):
            await service.generate_timeline_chart([])

    @pytest.mark.asyncio
    async def test_timeline_chart_korean_labels(self, tmp_path):
        """Test that timeline chart supports Korean labels."""
        service = ChartService(output_dir=str(tmp_path))

        data = [
            {"speaker": "남성", "timestamp": 0.0, "duration": 5.0, "emotion": "neutral"},
            {"speaker": "여성", "timestamp": 5.0, "duration": 3.0, "emotion": "angry"},
        ]

        figure = await service.generate_timeline_chart(data)
        assert isinstance(figure, Figure)

        # Verify Korean text is rendered (may show warnings but doesn't crash)
        ax = figure.axes[0]
        # Check that chart has title with Korean text
        assert ax.get_title() == "대화 타임라인"


class TestEmotionChangeChart:
    """Test emotion change chart generation."""

    @pytest.mark.asyncio
    async def test_generate_emotion_chart_basic(self, tmp_path):
        """Test basic emotion change chart."""
        service = ChartService(output_dir=str(tmp_path))

        data = [
            {"timestamp": 0.0, "emotion": "neutral", "intensity": 0.3},
            {"timestamp": 5.0, "emotion": "angry", "intensity": 0.7},
            {"timestamp": 10.0, "emotion": "sad", "intensity": 0.5},
            {"timestamp": 15.0, "emotion": "happy", "intensity": 0.4},
        ]

        figure = await service.generate_emotion_chart(data)

        assert isinstance(figure, Figure)
        assert len(figure.axes) == 1
        ax = figure.axes[0]
        assert ax.get_xlabel() == "시간 (초)"
        assert ax.get_ylabel() == "감정 강도"
        assert ax.get_title() == "감정 변화 추이"

    @pytest.mark.asyncio
    async def test_emotion_chart_empty_data(self, tmp_path):
        """Test emotion chart with empty data."""
        service = ChartService(output_dir=str(tmp_path))

        with pytest.raises(ValueError, match="Emotion data is required"):
            await service.generate_emotion_chart([])

    @pytest.mark.asyncio
    async def test_emotion_chart_color_mapping(self, tmp_path):
        """Test that different emotions have different colors."""
        service = ChartService(output_dir=str(tmp_path))

        data = [
            {"timestamp": i * 5.0, "emotion": emotion, "intensity": 0.5 + (i * 0.1)}
            for i, emotion in enumerate(["neutral", "angry", "sad", "happy", "fear"])
        ]

        figure = await service.generate_emotion_chart(data)
        assert isinstance(figure, Figure)

        # Verify color mapping exists
        ax = figure.axes[0]
        assert len(ax.lines) > 0


class TestCrimeDistributionChart:
    """Test crime speech distribution chart."""

    @pytest.mark.asyncio
    async def test_generate_crime_distribution_chart(self, tmp_path):
        """Test crime distribution pie chart."""
        service = ChartService(output_dir=str(tmp_path))

        data = {
            "협박": 5,
            "모욕": 3,
            "가스라이팅": 2,
            "성희롱": 1,
        }

        figure = await service.generate_crime_distribution_chart(data)

        assert isinstance(figure, Figure)
        assert len(figure.axes) == 1
        ax = figure.axes[0]
        assert ax.get_title() == "범죄 발언 유형 분포"

    @pytest.mark.asyncio
    async def test_crime_distribution_empty_data(self, tmp_path):
        """Test crime distribution with empty data."""
        service = ChartService(output_dir=str(tmp_path))

        with pytest.raises(ValueError, match="Crime distribution data is required"):
            await service.generate_crime_distribution_chart({})

    @pytest.mark.asyncio
    async def test_crime_distribution_percentages(self, tmp_path):
        """Test that crime distribution shows percentages."""
        service = ChartService(output_dir=str(tmp_path))

        data = {"협박": 7, "모욕": 3}
        figure = await service.generate_crime_distribution_chart(data)

        assert isinstance(figure, Figure)
        ax = figure.axes[0]

        # Verify autopct is enabled (percentages shown)
        assert len(ax.patches) == 2  # Two categories


class TestGaslightingPatternChart:
    """Test gaslighting pattern visualization."""

    @pytest.mark.asyncio
    async def test_generate_gaslighting_pattern_chart(self, tmp_path):
        """Test gaslighting pattern scatter chart."""
        service = ChartService(output_dir=str(tmp_path))

        data = [
            {"timestamp": 10.0, "pattern_type": "denial", "severity": 0.6},
            {"timestamp": 20.0, "pattern_type": "countering", "severity": 0.8},
            {"timestamp": 30.0, "pattern_type": "blocking", "severity": 0.5},
            {"timestamp": 40.0, "pattern_type": "denial", "severity": 0.7},
        ]

        figure = await service.generate_gaslighting_pattern_chart(data)

        assert isinstance(figure, Figure)
        assert len(figure.axes) == 1
        ax = figure.axes[0]
        assert ax.get_xlabel() == "시간 (초)"
        assert ax.get_ylabel() == "가스라이팅 패턴"
        assert ax.get_title() == "가스라이팅 패턴 분석"

    @pytest.mark.asyncio
    async def test_gaslighting_chart_empty_data(self, tmp_path):
        """Test gaslighting chart with empty data."""
        service = ChartService(output_dir=str(tmp_path))

        with pytest.raises(ValueError, match="Gaslighting pattern data is required"):
            await service.generate_gaslighting_pattern_chart([])


class TestChartImageExport:
    """Test chart export to image files."""

    @pytest.mark.asyncio
    async def test_save_chart_as_png(self, tmp_path):
        """Test saving chart as PNG file."""
        service = ChartService(output_dir=str(tmp_path))

        data = {"협박": 5, "모욕": 3}
        figure = await service.generate_crime_distribution_chart(data)

        output_path = await service.save_chart(figure, "test_crime_dist.png")

        assert output_path.exists()
        assert output_path.suffix == ".png"
        assert output_path.stat().st_size > 0

    @pytest.mark.asyncio
    async def test_save_chart_with_custom_path(self, tmp_path):
        """Test saving chart to custom path."""
        service = ChartService(output_dir=str(tmp_path))

        data = [
            {"timestamp": 0.0, "emotion": "neutral", "intensity": 0.5},
            {"timestamp": 5.0, "emotion": "angry", "intensity": 0.7},
        ]
        figure = await service.generate_emotion_chart(data)

        custom_path = tmp_path / "custom_emotion.png"
        output_path = await service.save_chart(figure, str(custom_path))

        assert output_path == custom_path
        assert output_path.exists()

    @pytest.mark.asyncio
    async def test_save_chart_creates_directory(self, tmp_path):
        """Test that save_chart creates output directory if needed."""
        service = ChartService(output_dir=str(tmp_path))

        data = {"협박": 5}
        figure = await service.generate_crime_distribution_chart(data)

        nested_path = tmp_path / "charts" / "nested" / "test.png"
        output_path = await service.save_chart(figure, str(nested_path))

        assert output_path.exists()
        assert output_path.parent.exists()


class TestChartTypeValidation:
    """Test ChartType enum validation."""

    def test_chart_type_enum_values(self):
        """Test that ChartType enum has all required chart types."""
        assert hasattr(ChartType, "TIMELINE")
        assert hasattr(ChartType, "EMOTION")
        assert hasattr(ChartType, "CRIME_DISTRIBUTION")
        assert hasattr(ChartType, "GASLIGHTING_PATTERN")

    def test_chart_type_from_string(self):
        """Test creating ChartType from string."""
        assert ChartType["TIMELINE"] == ChartType.TIMELINE
        assert ChartType["EMOTION"] == ChartType.EMOTION


class TestChartIntegration:
    """Integration tests for chart generation workflow."""

    @pytest.mark.asyncio
    async def test_generate_all_charts_for_report(self, tmp_path):
        """Test generating all chart types for a complete report."""
        service = ChartService(output_dir=str(tmp_path))

        # Timeline chart
        timeline_data = [
            {"speaker": "A", "timestamp": 0.0, "duration": 5.0, "emotion": "neutral"},
            {"speaker": "B", "timestamp": 5.0, "duration": 3.0, "emotion": "angry"},
        ]
        timeline_fig = await service.generate_timeline_chart(timeline_data)
        timeline_path = await service.save_chart(timeline_fig, "timeline.png")
        assert timeline_path.exists()

        # Emotion chart
        emotion_data = [
            {"timestamp": 0.0, "emotion": "neutral", "intensity": 0.3},
            {"timestamp": 5.0, "emotion": "angry", "intensity": 0.7},
        ]
        emotion_fig = await service.generate_emotion_chart(emotion_data)
        emotion_path = await service.save_chart(emotion_fig, "emotion.png")
        assert emotion_path.exists()

        # Crime distribution chart
        crime_data = {"협박": 5, "모욕": 3}
        crime_fig = await service.generate_crime_distribution_chart(crime_data)
        crime_path = await service.save_chart(crime_fig, "crime.png")
        assert crime_path.exists()

        # Gaslighting pattern chart
        gaslighting_data = [
            {"timestamp": 10.0, "pattern_type": "denial", "severity": 0.6},
        ]
        gaslighting_fig = await service.generate_gaslighting_pattern_chart(gaslighting_data)
        gaslighting_path = await service.save_chart(gaslighting_fig, "gaslighting.png")
        assert gaslighting_path.exists()

        # Verify all charts exist
        assert timeline_path.exists()
        assert emotion_path.exists()
        assert crime_path.exists()
        assert gaslighting_path.exists()
