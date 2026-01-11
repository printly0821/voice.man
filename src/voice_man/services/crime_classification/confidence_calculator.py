"""
Confidence Calculator Service
SPEC-CRIME-CLASS-001 Phase 6: Confidence Interval Calculation (95% CI, Bootstrap)
"""

import random
from typing import Dict, List


class ConfidenceCalculator:
    """
    신뢰 구간 계산기

    Bootstrap method for 95% confidence interval calculation
    """

    def __init__(self, n_iterations: int = 1000, confidence: float = 0.95) -> None:
        """
        Initialize confidence calculator

        Args:
            n_iterations: Number of bootstrap iterations
            confidence: Confidence level (default: 0.95 for 95% CI)
        """
        self.n_iterations = n_iterations
        self.confidence = confidence

    def calculate_confidence_interval(
        self,
        text_score: float,
        audio_score: float,
        psych_score: float,
        weights: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Calculate 95% confidence interval using bootstrap method

        Args:
            text_score: Text modality score
            audio_score: Audio modality score
            psych_score: Psychological modality score
            weights: Modality weights {"text": 0.4, "audio": 0.35, "psych": 0.25}

        Returns:
            Dictionary with {"lower_95": lower_bound, "upper_95": upper_bound}
        """
        # Create score array for bootstrap
        scores = [
            text_score * weights["text"],
            audio_score * weights["audio"],
            psych_score * weights["psych"],
        ]

        # Bootstrap sampling
        bootstrap_means = []
        for _ in range(self.n_iterations):
            # Resample with replacement
            sample = random.choices(scores, k=len(scores))
            bootstrap_means.append(sum(sample))

        # Calculate confidence interval
        alpha = 1 - self.confidence
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        sorted_means = sorted(bootstrap_means)
        lower_bound = sorted_means[int(lower_percentile / 100 * len(sorted_means))]
        upper_bound = sorted_means[int(upper_percentile / 100 * len(sorted_means))]

        point_estimate = sum(scores)

        return {
            "lower_95": lower_bound,
            "upper_95": upper_bound,
            "point_estimate": point_estimate,
        }

    def calculate_standard_error(
        self,
        text_score: float,
        audio_score: float,
        psych_score: float,
        weights: Dict[str, float],
    ) -> float:
        """
        Calculate standard error of the weighted score

        Args:
            text_score: Text modality score
            audio_score: Audio modality score
            psych_score: Psychological modality score
            weights: Modality weights

        Returns:
            Standard error value
        """
        scores = [
            text_score * weights["text"],
            audio_score * weights["audio"],
            psych_score * weights["psych"],
        ]

        if len(scores) < 2:
            return 0.0

        # Calculate sample standard deviation
        mean = sum(scores) / len(scores)
        variance = sum((s - mean) ** 2 for s in scores) / (len(scores) - 1)
        std_dev = variance**0.5

        # Standard error = std_dev / sqrt(n)
        import math

        return std_dev / math.sqrt(len(scores))
