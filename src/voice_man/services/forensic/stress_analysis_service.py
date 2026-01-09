"""
Stress Analysis Service
SPEC-FORENSIC-001 TASK-011~014: Voice stress analysis service

This service provides voice stress analysis features including
shimmer, HNR, formant stability, and stress index calculation.
"""

from typing import Tuple
import numpy as np

from voice_man.models.forensic.audio_features import StressFeatures


class StressAnalysisService:
    """
    Service for analyzing voice stress indicators.

    This service provides methods for:
    - Shimmer calculation (amplitude perturbation)
    - HNR (Harmonics-to-Noise Ratio) calculation
    - Formant stability analysis
    - Stress index calculation with risk level classification
    """

    # Stress index thresholds
    LOW_STRESS_THRESHOLD = 33.0
    HIGH_STRESS_THRESHOLD = 66.0

    # Weight factors for stress index calculation
    SHIMMER_WEIGHT = 0.25
    HNR_WEIGHT = 0.25
    FORMANT_WEIGHT = 0.25
    JITTER_WEIGHT = 0.25

    def __init__(self):
        """Initialize the StressAnalysisService."""
        self._parselmouth = None
        self._librosa = None

    @property
    def parselmouth(self):
        """Lazy load parselmouth."""
        if self._parselmouth is None:
            try:
                import parselmouth

                self._parselmouth = parselmouth
            except ImportError:
                self._parselmouth = None
        return self._parselmouth

    @property
    def librosa(self):
        """Lazy load librosa."""
        if self._librosa is None:
            import librosa

            self._librosa = librosa
        return self._librosa

    def calculate_shimmer(self, audio: np.ndarray, sr: int) -> float:
        """
        Calculate shimmer (amplitude perturbation) as percentage.

        TASK-011: Shimmer calculation using parselmouth or fallback method

        Shimmer measures the cycle-to-cycle variation in amplitude,
        which increases with voice stress and pathology.

        Args:
            audio: Audio signal as numpy array
            sr: Sample rate in Hz

        Returns:
            Shimmer as percentage (0-15%)
        """
        try:
            if self.parselmouth is not None:
                return self._calculate_shimmer_parselmouth(audio, sr)
            else:
                return self._calculate_shimmer_fallback(audio, sr)
        except Exception:
            return self._calculate_shimmer_fallback(audio, sr)

    def _calculate_shimmer_parselmouth(self, audio: np.ndarray, sr: int) -> float:
        """Calculate shimmer using parselmouth/Praat."""
        # Convert to parselmouth Sound object
        sound = self.parselmouth.Sound(audio, sampling_frequency=sr)

        # Create PointProcess for pitch analysis
        point_process = sound.to_pitch().to_point_process()

        # Calculate shimmer (local)
        shimmer = sound.get_shimmer_local(
            point_process,
            0.0,  # start time
            0.0,  # end time (0 = entire sound)
            0.0001,  # shortest period
            0.02,  # longest period
            1.3,  # maximum period factor
            1.6,  # maximum amplitude factor
        )

        if np.isnan(shimmer):
            return self._calculate_shimmer_fallback(audio, sr)

        # Convert to percentage and clamp to valid range
        shimmer_percent = shimmer * 100
        return float(min(15.0, max(0.0, shimmer_percent)))

    def _calculate_shimmer_fallback(self, audio: np.ndarray, sr: int) -> float:
        """Calculate shimmer using a simple amplitude variation method."""
        # Simple amplitude variation calculation
        # Find peaks in the audio signal
        frame_length = int(sr * 0.025)  # 25ms frames
        hop_length = int(sr * 0.010)  # 10ms hop

        amplitudes = []
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i : i + frame_length]
            amp = np.max(np.abs(frame))
            if amp > 0.01:  # Ignore very quiet frames
                amplitudes.append(amp)

        if len(amplitudes) < 2:
            return 0.0

        amplitudes = np.array(amplitudes)

        # Calculate shimmer as mean absolute difference between consecutive amplitudes
        amp_diffs = np.abs(np.diff(amplitudes))
        mean_amp = np.mean(amplitudes)

        if mean_amp > 0:
            shimmer_percent = (np.mean(amp_diffs) / mean_amp) * 100
        else:
            shimmer_percent = 0.0

        return float(min(15.0, max(0.0, shimmer_percent)))

    def calculate_hnr(self, audio: np.ndarray, sr: int) -> float:
        """
        Calculate Harmonics-to-Noise Ratio (HNR) in decibels.

        TASK-012: HNR calculation using parselmouth or fallback method

        HNR measures the ratio of periodic (harmonic) energy to aperiodic
        (noise) energy in the voice signal. Lower HNR indicates more
        breathiness or hoarseness, often associated with stress.

        Args:
            audio: Audio signal as numpy array
            sr: Sample rate in Hz

        Returns:
            HNR in decibels (0-40dB)
        """
        try:
            if self.parselmouth is not None:
                return self._calculate_hnr_parselmouth(audio, sr)
            else:
                return self._calculate_hnr_fallback(audio, sr)
        except Exception:
            return self._calculate_hnr_fallback(audio, sr)

    def _calculate_hnr_parselmouth(self, audio: np.ndarray, sr: int) -> float:
        """Calculate HNR using parselmouth/Praat."""
        # Convert to parselmouth Sound object
        sound = self.parselmouth.Sound(audio, sampling_frequency=sr)

        # Calculate harmonicity (HNR)
        harmonicity = sound.to_harmonicity()

        # Get mean HNR
        hnr_values = harmonicity.values[0]
        valid_hnr = hnr_values[~np.isnan(hnr_values)]

        if len(valid_hnr) == 0:
            return self._calculate_hnr_fallback(audio, sr)

        hnr_mean = float(np.mean(valid_hnr))

        # Clamp to valid range
        return float(min(40.0, max(0.0, hnr_mean)))

    def _calculate_hnr_fallback(self, audio: np.ndarray, sr: int) -> float:
        """Calculate HNR using autocorrelation method."""
        # Use autocorrelation to estimate HNR
        # Frame-based analysis
        frame_length = int(sr * 0.040)  # 40ms frames
        hop_length = int(sr * 0.010)  # 10ms hop

        hnr_values = []

        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i : i + frame_length]

            # Skip silent frames
            if np.max(np.abs(frame)) < 0.01:
                continue

            # Calculate autocorrelation
            autocorr = np.correlate(frame, frame, mode="full")
            autocorr = autocorr[len(autocorr) // 2 :]

            # Normalize
            autocorr = autocorr / (autocorr[0] + 1e-10)

            # Find first peak after zero (fundamental period)
            min_period = int(sr / 600)  # Max F0 = 600 Hz
            max_period = int(sr / 75)  # Min F0 = 75 Hz

            if max_period > len(autocorr):
                max_period = len(autocorr) - 1

            # Find maximum in valid range
            search_range = autocorr[min_period:max_period]
            if len(search_range) > 0:
                max_autocorr = np.max(search_range)

                # HNR approximation from autocorrelation
                if max_autocorr > 0 and max_autocorr < 1:
                    hnr = 10 * np.log10(max_autocorr / (1 - max_autocorr + 1e-10))
                    if not np.isnan(hnr) and not np.isinf(hnr):
                        hnr_values.append(hnr)

        if len(hnr_values) == 0:
            return 15.0  # Default moderate HNR

        hnr_mean = np.mean(hnr_values)
        return float(min(40.0, max(0.0, hnr_mean)))

    def calculate_formant_stability(self, audio: np.ndarray, sr: int) -> float:
        """
        Calculate formant stability score (0-100).

        TASK-013: Formant stability analysis

        Measures the stability of F1 and F2 formants over time.
        Higher stability indicates more controlled speech,
        while lower stability may indicate stress or emotional state.

        Args:
            audio: Audio signal as numpy array
            sr: Sample rate in Hz

        Returns:
            Formant stability score (0-100)
        """
        try:
            if self.parselmouth is not None:
                return self._calculate_formant_stability_parselmouth(audio, sr)
            else:
                return self._calculate_formant_stability_fallback(audio, sr)
        except Exception:
            return self._calculate_formant_stability_fallback(audio, sr)

    def _calculate_formant_stability_parselmouth(self, audio: np.ndarray, sr: int) -> float:
        """Calculate formant stability using parselmouth/Praat."""
        # Convert to parselmouth Sound object
        sound = self.parselmouth.Sound(audio, sampling_frequency=sr)

        # Extract formants
        formant = sound.to_formant_burg(
            time_step=0.01,
            max_number_of_formants=5,
            maximum_formant=5500.0,
            window_length=0.025,
            pre_emphasis_from=50.0,
        )

        # Get F1 and F2 values over time
        times = formant.xs()
        f1_values = []
        f2_values = []

        for t in times:
            f1 = formant.get_value_at_time(1, t)
            f2 = formant.get_value_at_time(2, t)
            if not np.isnan(f1) and not np.isnan(f2):
                f1_values.append(f1)
                f2_values.append(f2)

        if len(f1_values) < 2:
            return 75.0  # Default moderate stability

        f1_values = np.array(f1_values)
        f2_values = np.array(f2_values)

        # Calculate coefficient of variation (CV) for each formant
        f1_cv = np.std(f1_values) / (np.mean(f1_values) + 1e-10)
        f2_cv = np.std(f2_values) / (np.mean(f2_values) + 1e-10)

        # Combined stability score
        # Lower CV = higher stability
        # CV of 0 = 100% stable, CV of 0.5 = 0% stable
        avg_cv = (f1_cv + f2_cv) / 2
        stability = 100.0 * (1.0 - min(1.0, avg_cv * 2))

        return float(min(100.0, max(0.0, stability)))

    def _calculate_formant_stability_fallback(self, audio: np.ndarray, sr: int) -> float:
        """Calculate formant stability using spectral centroid variation."""
        # Use spectral centroid as a proxy for formant tracking
        spectral_centroid = self.librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=512)[
            0
        ]

        if len(spectral_centroid) < 2:
            return 75.0

        # Calculate coefficient of variation
        cv = np.std(spectral_centroid) / (np.mean(spectral_centroid) + 1e-10)

        # Convert to stability score (0-100)
        # Lower CV = higher stability
        stability = 100.0 * (1.0 - min(1.0, cv))

        return float(min(100.0, max(0.0, stability)))

    def calculate_stress_index(
        self, shimmer_percent: float, hnr_db: float, formant_stability: float, jitter_percent: float
    ) -> Tuple[float, str]:
        """
        Calculate composite stress index and risk level.

        TASK-014: Stress index calculation using weighted combination

        The stress index combines multiple voice quality measures
        to produce an overall stress assessment.

        Args:
            shimmer_percent: Shimmer value (0-15%)
            hnr_db: HNR value (0-40dB)
            formant_stability: Formant stability score (0-100)
            jitter_percent: Jitter value (0-10%)

        Returns:
            Tuple of (stress_index, risk_level)
            - stress_index: 0-100 (higher = more stress)
            - risk_level: "low", "medium", or "high"
        """
        # Normalize each component to 0-100 stress contribution
        # Higher values indicate MORE stress

        # Shimmer: 0-15% -> 0-100 stress
        # Higher shimmer = more stress
        shimmer_normalized = (shimmer_percent / 15.0) * 100.0

        # HNR: 0-40dB -> 100-0 stress (inverted - lower HNR = more stress)
        # Lower HNR = more stress
        hnr_normalized = (1.0 - (hnr_db / 40.0)) * 100.0

        # Formant stability: 0-100 -> 100-0 stress (inverted)
        # Lower stability = more stress
        formant_normalized = 100.0 - formant_stability

        # Jitter: 0-10% -> 0-100 stress
        # Higher jitter = more stress
        jitter_normalized = (jitter_percent / 10.0) * 100.0

        # Clamp normalized values
        shimmer_normalized = min(100.0, max(0.0, shimmer_normalized))
        hnr_normalized = min(100.0, max(0.0, hnr_normalized))
        formant_normalized = min(100.0, max(0.0, formant_normalized))
        jitter_normalized = min(100.0, max(0.0, jitter_normalized))

        # Calculate weighted stress index
        stress_index = (
            self.SHIMMER_WEIGHT * shimmer_normalized
            + self.HNR_WEIGHT * hnr_normalized
            + self.FORMANT_WEIGHT * formant_normalized
            + self.JITTER_WEIGHT * jitter_normalized
        )

        # Clamp to valid range
        stress_index = min(100.0, max(0.0, stress_index))

        # Determine risk level
        if stress_index < self.LOW_STRESS_THRESHOLD:
            risk_level = "low"
        elif stress_index > self.HIGH_STRESS_THRESHOLD:
            risk_level = "high"
        else:
            risk_level = "medium"

        return float(stress_index), risk_level

    def analyze_stress(self, audio: np.ndarray, sr: int, jitter_percent: float) -> StressFeatures:
        """
        Perform complete stress analysis.

        Args:
            audio: Audio signal as numpy array
            sr: Sample rate in Hz
            jitter_percent: Pre-calculated jitter value

        Returns:
            StressFeatures with all stress indicators
        """
        # Calculate individual stress indicators
        shimmer = self.calculate_shimmer(audio, sr)
        hnr = self.calculate_hnr(audio, sr)
        formant_stability = self.calculate_formant_stability(audio, sr)

        # Calculate composite stress index
        stress_index, risk_level = self.calculate_stress_index(
            shimmer_percent=shimmer,
            hnr_db=hnr,
            formant_stability=formant_stability,
            jitter_percent=jitter_percent,
        )

        return StressFeatures(
            shimmer_percent=shimmer,
            hnr_db=hnr,
            formant_stability_score=formant_stability,
            stress_index=stress_index,
            risk_level=risk_level,
        )
