"""
Chart Generation Service for Legal Evidence Reports.

Provides Matplotlib-based visualization for:
- Timeline charts (conversation flow)
- Emotion change charts (sentiment analysis)
- Crime distribution charts (legal evidence)
- Gaslighting pattern charts (abuse detection)
"""

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from pathlib import Path
from enum import Enum
from typing import Any
import logging

logger = logging.getLogger(__name__)


class ChartType(Enum):
    """Supported chart types for legal evidence reports."""

    TIMELINE = "timeline"
    EMOTION = "emotion"
    CRIME_DISTRIBUTION = "crime_distribution"
    GASLIGHTING_PATTERN = "gaslighting_pattern"


class ChartService:
    """
    Service for generating charts for legal evidence reports.

    Supports Korean text rendering and produces high-quality visualizations
    suitable for legal documentation.
    """

    # Emotion color mapping
    EMOTION_COLORS = {
        "neutral": "#808080",  # Gray
        "happy": "#FFD700",  # Gold
        "sad": "#4169E1",  # Royal Blue
        "angry": "#DC143C",  # Crimson
        "fear": "#8B008B",  # Dark Magenta
        "disgust": "#228B22",  # Forest Green
        "surprise": "#FF8C00",  # Dark Orange
    }

    # Gaslighting pattern colors
    PATTERN_COLORS = {
        "denial": "#DC143C",  # Crimson
        "countering": "#FF8C00",  # Dark Orange
        "blocking": "#4169E1",  # Royal Blue
        "trivializing": "#808080",  # Gray
        "forgetting": "#9370DB",  # Medium Purple
    }

    def __init__(
        self,
        output_dir: str = "charts",
        width: int = 10,
        height: int = 6,
        dpi: int = 300,
    ):
        """
        Initialize ChartService.

        Args:
            output_dir: Directory for saving chart images
            width: Figure width in inches
            height: Figure height in inches
            dpi: Resolution for saved images
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figure_width = width
        self.figure_height = height
        self.dpi = dpi

        # Configure matplotlib for Korean support
        self._setup_korean_font()

    def _setup_korean_font(self) -> None:
        """Configure matplotlib to support Korean text rendering."""
        try:
            # Try common Korean fonts
            korean_fonts = [
                "NanumGothic",
                "Malgun Gothic",
                "AppleGothic",
                "UnDotum",
                "Baekmuk Dotum",
            ]

            font_found = False
            for font_name in korean_fonts:
                try:
                    matplotlib.rcParams["font.sans-serif"] = [font_name] + matplotlib.rcParams[
                        "font.sans-serif"
                    ]
                    font_found = True
                    logger.info(f"Using Korean font: {font_name}")
                    break
                except Exception:
                    continue

            if not font_found:
                logger.warning("No Korean font found, text may not display correctly")

            # Configure negative sign
            matplotlib.rcParams["axes.unicode_minus"] = False
        except Exception as e:
            logger.warning(f"Failed to setup Korean font: {e}")

    async def generate_timeline_chart(self, data: list[dict[str, Any]]) -> Figure:
        """
        Generate timeline chart showing conversation flow.

        Args:
            data: List of timeline entries with speaker, timestamp, duration, emotion

        Returns:
            Figure: Matplotlib figure object

        Raises:
            ValueError: If data is empty or missing required fields
        """
        if not data:
            raise ValueError("Timeline data is required")

        # Validate data structure
        required_fields = {"speaker", "timestamp", "duration"}
        for entry in data:
            if not required_fields.issubset(entry.keys()):
                raise ValueError(f"Timeline entry missing required fields: {entry}")

        fig, ax = plt.subplots(figsize=(self.figure_width, self.figure_height))

        # Group by speaker
        speakers = {}
        for entry in data:
            speaker = entry["speaker"]
            if speaker not in speakers:
                speakers[speaker] = []
            speakers[speaker].append(entry)

        # Plot timeline for each speaker
        y_offset = 0
        for speaker, entries in speakers.items():
            for entry in entries:
                start_time = entry["timestamp"]
                duration = entry["duration"]
                emotion = entry.get("emotion", "neutral")
                color = self.EMOTION_COLORS.get(emotion, "#808080")

                ax.barh(
                    y_offset,
                    duration,
                    left=start_time,
                    height=0.5,
                    color=color,
                    edgecolor="black",
                    linewidth=0.5,
                )

            # Set speaker label
            ax.text(
                -max(entry["duration"] for entry in entries) * 0.1,
                y_offset,
                speaker,
                ha="right",
                va="center",
                fontsize=10,
            )
            y_offset += 1

        ax.set_xlabel("시간 (초)", fontsize=12)
        ax.set_ylabel("화자", fontsize=12)
        ax.set_title("대화 타임라인", fontsize=14, fontweight="bold")
        ax.grid(True, axis="x", alpha=0.3)

        # Remove y-axis ticks
        ax.set_yticks([])

        plt.tight_layout()
        return fig

    async def generate_emotion_chart(self, data: list[dict[str, Any]]) -> Figure:
        """
        Generate emotion change chart over time.

        Args:
            data: List of emotion entries with timestamp, emotion, intensity

        Returns:
            Figure: Matplotlib figure object

        Raises:
            ValueError: If data is empty or missing required fields
        """
        if not data:
            raise ValueError("Emotion data is required")

        # Validate data structure
        required_fields = {"timestamp", "emotion", "intensity"}
        for entry in data:
            if not required_fields.issubset(entry.keys()):
                raise ValueError(f"Emotion entry missing required fields: {entry}")

        fig, ax = plt.subplots(figsize=(self.figure_width, self.figure_height))

        # Group by emotion
        emotion_lines: dict[str, list[tuple[float, float]]] = {}
        for entry in data:
            emotion = entry["emotion"]
            if emotion not in emotion_lines:
                emotion_lines[emotion] = []
            emotion_lines[emotion].append((entry["timestamp"], entry["intensity"]))

        # Plot each emotion line
        for emotion, points in sorted(emotion_lines.items()):
            points.sort()  # Sort by timestamp
            timestamps = [p[0] for p in points]
            intensities = [p[1] for p in points]
            color = self.EMOTION_COLORS.get(emotion, "#808080")

            ax.plot(
                timestamps,
                intensities,
                marker="o",
                label=emotion,
                color=color,
                linewidth=2,
                markersize=6,
            )

        ax.set_xlabel("시간 (초)", fontsize=12)
        ax.set_ylabel("감정 강도", fontsize=12)
        ax.set_title("감정 변화 추이", fontsize=14, fontweight="bold")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.0)

        plt.tight_layout()
        return fig

    async def generate_crime_distribution_chart(self, data: dict[str, int]) -> Figure:
        """
        Generate crime type distribution pie chart.

        Args:
            data: Dictionary mapping crime types to counts

        Returns:
            Figure: Matplotlib figure object

        Raises:
            ValueError: If data is empty
        """
        if not data:
            raise ValueError("Crime distribution data is required")

        fig, ax = plt.subplots(figsize=(self.figure_width, self.figure_height))

        # Crime type colors (distinct colors for legal categories)
        colors = [
            "#DC143C",  # Crimson
            "#FF8C00",  # Dark Orange
            "#4169E1",  # Royal Blue
            "#228B22",  # Forest Green
            "#8B008B",  # Dark Magenta
            "#DAA520",  # Goldenrod
        ]

        labels = list(data.keys())
        sizes = list(data.values())

        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            autopct="%1.1f%%",
            startangle=90,
            colors=colors[: len(labels)],
            textprops={"fontsize": 11},
        )

        # Enhance percentage text
        for autotext in autotexts:
            autotext.set_fontsize(10)
            autotext.set_fontweight("bold")

        ax.set_title("범죄 발언 유형 분포", fontsize=14, fontweight="bold")

        plt.tight_layout()
        return fig

    async def generate_gaslighting_pattern_chart(self, data: list[dict[str, Any]]) -> Figure:
        """
        Generate gaslighting pattern scatter chart.

        Args:
            data: List of pattern entries with timestamp, pattern_type, severity

        Returns:
            Figure: Matplotlib figure object

        Raises:
            ValueError: If data is empty or missing required fields
        """
        if not data:
            raise ValueError("Gaslighting pattern data is required")

        # Validate data structure
        required_fields = {"timestamp", "pattern_type", "severity"}
        for entry in data:
            if not required_fields.issubset(entry.keys()):
                raise ValueError(f"Gaslighting entry missing required fields: {entry}")

        fig, ax = plt.subplots(figsize=(self.figure_width, self.figure_height))

        # Group by pattern type
        pattern_types = {}
        for entry in data:
            pattern = entry["pattern_type"]
            if pattern not in pattern_types:
                pattern_types[pattern] = {"x": [], "y": [], "sizes": []}
            pattern_types[pattern]["x"].append(entry["timestamp"])
            pattern_types[pattern]["y"].append(pattern)
            pattern_types[pattern]["sizes"].append(entry["severity"] * 500)

        # Plot scatter for each pattern type
        for pattern, coords in pattern_types.items():
            color = self.PATTERN_COLORS.get(pattern, "#808080")
            ax.scatter(
                coords["x"],
                coords["y"],
                s=coords["sizes"],
                alpha=0.6,
                color=color,
                edgecolors="black",
                linewidth=1,
                label=pattern,
            )

        ax.set_xlabel("시간 (초)", fontsize=12)
        ax.set_ylabel("가스라이팅 패턴", fontsize=12)
        ax.set_title("가스라이팅 패턴 분석", fontsize=14, fontweight="bold")
        ax.legend(loc="upper right")
        ax.grid(True, axis="x", alpha=0.3)

        plt.tight_layout()
        return fig

    async def save_chart(self, figure: Figure, filename: str) -> Path:
        """
        Save chart to file.

        Args:
            figure: Matplotlib figure object
            filename: Output filename

        Returns:
            Path: Path to saved file
        """
        output_path = self.output_dir / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        figure.savefig(
            output_path,
            dpi=self.dpi,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )

        plt.close(figure)
        logger.info(f"Chart saved to: {output_path}")

        return output_path
