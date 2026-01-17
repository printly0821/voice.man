"""
Bootstrap Confidence Interval Calculation for Forensic Score Uncertainty.

Implements bootstrap resampling with percentile and BCa methods
for statistical significance assessment in forensic analysis.

TAG: [FORENSIC-EVIDENCE-001]
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional, Literal


class BootstrapConfidenceInterval:
    """
    Bootstrap confidence interval calculator for forensic scores.

    Provides 95% confidence interval computation using bootstrap resampling
    for uncertainty quantification in legal evidence analysis.
    """

    def __init__(
        self,
        n_iterations: int = 10000,
        confidence_level: float = 0.95,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize bootstrap confidence interval calculator.

        Args:
            n_iterations: Number of bootstrap iterations (default: 10000)
            confidence_level: Confidence level (default: 0.95 for 95% CI)
            random_seed: Random seed for reproducibility (optional)
        """
        self.n_iterations = n_iterations
        self.confidence_level = confidence_level
        self.random_seed = random_seed

        if random_seed is not None:
            np.random.seed(random_seed)

    def calculate_ci(
        self,
        data: np.ndarray,
        method: Literal["percentile", "bca"] = "percentile",
    ) -> Tuple[float, float]:
        """
        Calculate bootstrap confidence interval.

        Args:
            data: Array of forensic scores
            method: Method to use ('percentile' or 'bca')

        Returns:
            Tuple of (lower_bound, upper_bound) for confidence interval
        """
        if method == "percentile":
            return self._calculate_percentile_ci(data)
        elif method == "bca":
            return self._calculate_bca_ci(data)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _calculate_percentile_ci(self, data: np.ndarray) -> Tuple[float, float]:
        """
        Calculate confidence interval using percentile method.

        Args:
            data: Array of forensic scores

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        # Reset seed for this calculation if specified
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        bootstrap_means = []

        n = len(data)

        for _ in range(self.n_iterations):
            # Resample with replacement
            resample = np.random.choice(data, size=n, replace=True)
            bootstrap_means.append(np.mean(resample))

        bootstrap_means = np.array(bootstrap_means)

        # Calculate percentiles
        alpha = 1 - self.confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        ci_lower = np.percentile(bootstrap_means, lower_percentile)
        ci_upper = np.percentile(bootstrap_means, upper_percentile)

        return ci_lower, ci_upper

    def _calculate_bca_ci(self, data: np.ndarray) -> Tuple[float, float]:
        """
        Calculate confidence interval using BCa (Bias-Corrected and Accelerated) method.

        Args:
            data: Array of forensic scores

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        # Reset seed for this calculation if specified
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        n = len(data)
        bootstrap_means = []

        for _ in range(self.n_iterations):
            resample = np.random.choice(data, size=n, replace=True)
            bootstrap_means.append(np.mean(resample))

        bootstrap_means = np.array(bootstrap_means)

        # Calculate bias correction factor (z0)
        original_mean = np.mean(data)
        z0 = self._calculate_z0(bootstrap_means, original_mean)

        # Calculate acceleration factor (a)
        a = self._calculate_acceleration(data)

        # Calculate adjusted percentiles
        alpha = 1 - self.confidence_level
        z_alpha = self._get_z_score(alpha / 2)
        z_1_alpha = self._get_z_score(1 - alpha / 2)

        # BCa adjusted percentiles
        lower_percentile = self._norm_cdf(z0 + (z0 + z_alpha) / (1 - a * (z0 + z_alpha)))
        upper_percentile = self._norm_cdf(z0 + (z0 + z_1_alpha) / (1 - a * (z0 + z_1_alpha)))

        ci_lower = np.percentile(bootstrap_means, lower_percentile * 100)
        ci_upper = np.percentile(bootstrap_means, upper_percentile * 100)

        return ci_lower, ci_upper

    def _calculate_z0(self, bootstrap_means: np.ndarray, original_mean: float) -> float:
        """Calculate bias correction factor z0."""
        proportion = np.sum(bootstrap_means < original_mean) / len(bootstrap_means)
        # Avoid division by zero or invalid values
        proportion = np.clip(proportion, 1e-10, 1 - 1e-10)
        z0 = self._norm_ppf(proportion)
        return z0

    def _calculate_acceleration(self, data: np.ndarray) -> float:
        """Calculate acceleration factor a using jackknife."""
        n = len(data)
        jackknife_means = []

        for i in range(n):
            # Leave-one-out mean
            jackknife_sample = np.delete(data, i)
            jackknife_means.append(np.mean(jackknife_sample))

        jackknife_means = np.array(jackknife_means)
        mean_of_means = np.mean(jackknife_means)

        numerator = np.sum((mean_of_means - jackknife_means) ** 3)
        denominator = 6 * (np.sum((mean_of_means - jackknife_means) ** 2) ** 1.5)

        # Avoid division by zero
        if denominator == 0:
            return 0.0

        a = numerator / denominator
        return a

    def _norm_ppf(self, p: float) -> float:
        """Inverse of normal CDF (percent point function)."""
        from scipy import stats

        return stats.norm.ppf(p)

    def _norm_cdf(self, x: float) -> float:
        """Normal cumulative distribution function."""
        from scipy import stats

        return stats.norm.cdf(x)

    def _get_z_score(self, p: float) -> float:
        """Get z-score for given probability."""
        from scipy import stats

        return stats.norm.ppf(p)

    def get_bootstrap_statistics(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Get comprehensive bootstrap statistics.

        Args:
            data: Array of forensic scores

        Returns:
            Dictionary containing mean, std, and confidence interval
        """
        ci_lower, ci_upper = self.calculate_ci(data, method="percentile")

        return {
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "confidence_level": self.confidence_level,
            "n_iterations": self.n_iterations,
        }
