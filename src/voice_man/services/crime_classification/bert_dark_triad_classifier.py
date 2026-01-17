"""
BERT Dark Triad Classifier - Hybrid ML-based Approach
SPEC-CRIME-CLASS-001 Enhancement: BERT + Random Forest Hybrid

Integrates KoBERT embeddings with Random Forest for Dark Triad trait classification.
Maintains backward compatibility with existing keyword-based PsychologicalProfiler.
"""

from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np

from voice_man.services.crime_classification.bert_feature_extractor import (
    BERTFeatureExtractor,
)
from voice_man.services.crime_classification.bert_rf_classifier import (
    BERTRandomForestClassifier,
)
from voice_man.services.crime_classification.training_data_generator import (
    generate_training_data,
    DarkTriadTrainingDataGenerator,
)


class BERTDarkTriadClassifier:
    """
    BERT + Random Forest hybrid classifier for Dark Triad traits

    Combines deep learning embeddings with traditional ML for efficient
    personality trait classification from Korean text.

    Features:
    - Automatic model loading from disk or training from scratch
    - Fallback to keyword-based classifier if ML model unavailable
    - Cached embeddings for efficient repeated predictions
    - Version tracking for model updates

    Attributes:
        model_dir: Directory for model persistence
        use_ml_model: Whether to use ML model (True) or keyword fallback (False)
        feature_extractor: BERT feature extractor instance
        classifier: Random Forest classifier instance
        is_loaded: Whether ML model is loaded
    """

    DEFAULT_MODEL_DIR = Path("models/crime_classification/bert_rf")

    def __init__(
        self,
        model_dir: Optional[Path] = None,
        auto_load: bool = True,
        auto_train: bool = False,
        n_training_samples: int = 1000,
    ) -> None:
        """
        Initialize BERT Dark Triad classifier

        Args:
            model_dir: Directory for model persistence
            auto_load: Automatically load trained model if available
            auto_train: Automatically train if no model found
            n_training_samples: Number of samples for auto-training
        """
        self.model_dir = model_dir or self.DEFAULT_MODEL_DIR
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.feature_extractor = BERTFeatureExtractor()
        self.classifier = BERTRandomForestClassifier(feature_extractor=self.feature_extractor)

        self.is_loaded = False

        # Load existing model or train new one
        if auto_load:
            self._load_or_train(auto_train=auto_train, n_samples=n_training_samples)

    def _load_or_train(self, auto_train: bool, n_samples: int) -> None:
        """
        Load existing model or train new one

        Args:
            auto_train: Whether to train if no model found
            n_samples: Number of training samples
        """
        # Try to load existing model
        if self._try_load_model():
            return

        # Train new model if requested
        if auto_train:
            self.train(n_samples=n_samples)

    def _try_load_model(self) -> bool:
        """
        Attempt to load existing trained model

        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            # Check for model files
            metadata_file = self.model_dir / "metadata.json"
            if not metadata_file.exists():
                return False

            # Check for all trait models
            for trait in self.classifier.traits:
                model_file = self.model_dir / f"{trait}.pkl"
                if not model_file.exists():
                    return False

            # Load model
            self.classifier = BERTRandomForestClassifier.load(
                self.model_dir, feature_extractor=self.feature_extractor
            )
            self.is_loaded = True
            return True

        except Exception:
            # Load failed - model may be corrupted
            return False

    def train(
        self,
        texts: Optional[list] = None,
        labels: Optional[Dict[str, np.ndarray]] = None,
        n_samples: int = 1000,
        distribution: str = "balanced",
        test_size: float = 0.2,
        save_model: bool = True,
    ) -> Dict[str, float]:
        """
        Train the BERT + Random Forest classifier

        Args:
            texts: Training texts (None to generate synthetic data)
            labels: Training labels (None to generate synthetic data)
            n_samples: Number of synthetic samples to generate
            distribution: Label distribution for synthetic data
            test_size: Fraction of data for testing
            save_model: Whether to save trained model

        Returns:
            Dictionary with training metrics
        """
        # Generate training data if not provided
        if texts is None or labels is None:
            texts, labels = generate_training_data(n_samples=n_samples, distribution=distribution)

        # Train classifier
        metrics = self.classifier.train(
            texts=texts,
            labels=labels,
            test_size=test_size,
        )

        self.is_loaded = True

        # Save model if requested
        if save_model:
            self.save()

        return {
            "accuracy": float(metrics.accuracy),
            "f1_score": float(metrics.f1_score),
        }

    def predict(
        self,
        text: str,
        return_probabilities: bool = True,
        fallback_to_keywords: bool = True,
    ) -> Dict[str, float]:
        """
        Predict Dark Triad traits for text

        Args:
            text: Input text to analyze
            return_probabilities: Return probability scores (0-1)
            fallback_to_keywords: Use keyword matching if ML model unavailable

        Returns:
            Dictionary with trait scores (0-1 range)
        """
        # Use ML model if available
        if self.is_loaded:
            result = self.classifier.predict_single(
                text=text,
                return_probabilities=return_probabilities,
            )
            return result

        # Fallback to keyword-based classifier
        if fallback_to_keywords:
            return self._keyword_fallback(text)

        # Model not available and no fallback
        raise RuntimeError("ML model not trained. Call train() or set fallback_to_keywords=True")

    def predict_batch(
        self,
        texts: list,
        return_probabilities: bool = True,
        fallback_to_keywords: bool = True,
    ) -> list[Dict[str, float]]:
        """
        Predict Dark Triad traits for multiple texts

        Args:
            texts: List of input texts
            return_probabilities: Return probability scores (0-1)
            fallback_to_keywords: Use keyword matching if ML model unavailable

        Returns:
            List of dictionaries with trait scores
        """
        if self.is_loaded:
            results = self.classifier.predict(
                texts=texts,
                return_probabilities=return_probabilities,
            )

            # Convert to list of dicts
            return [
                {trait: float(scores[i]) for trait, scores in results.items()}
                for i in range(len(texts))
            ]

        # Fallback to keyword-based
        if fallback_to_keywords:
            return [self._keyword_fallback(text) for text in texts]

        raise RuntimeError("ML model not trained. Call train() or set fallback_to_keywords=True")

    def _keyword_fallback(self, text: str) -> Dict[str, float]:
        """
        Keyword-based classification fallback

        Uses simplified keyword matching for backward compatibility.

        Args:
            text: Input text

        Returns:
            Dictionary with trait scores (0-1 range)
        """
        # Load existing PsychologicalProfiler for keyword matching
        from voice_man.services.crime_classification.psychological_profiler import (
            PsychologicalProfiler,
        )

        profiler = PsychologicalProfiler()
        return profiler.analyze_text(text)

    def save(self, model_dir: Optional[Path] = None) -> None:
        """
        Save trained model to disk

        Args:
            model_dir: Directory to save model (default: instance model_dir)
        """
        if not self.is_loaded:
            raise RuntimeError("Cannot save untrained model")

        save_dir = model_dir or self.model_dir
        self.classifier.save(save_dir)

    def load(self, model_dir: Optional[Path] = None) -> None:
        """
        Load trained model from disk

        Args:
            model_dir: Directory to load model from (default: instance model_dir)
        """
        load_dir = model_dir or self.model_dir
        self.classifier = BERTRandomForestClassifier.load(
            load_dir, feature_extractor=self.feature_extractor
        )
        self.is_loaded = True

    def clear_cache(self) -> None:
        """Clear embedding cache"""
        self.feature_extractor.clear_cache()

    def get_cache_size(self) -> int:
        """
        Get number of cached embeddings

        Returns:
            Number of cached embeddings
        """
        return self.feature_extractor.get_cache_size()

    def get_model_info(self) -> Dict[str, Union[str, int, bool]]:
        """
        Get model information

        Returns:
            Dictionary with model metadata
        """
        info = {
            "is_loaded": self.is_loaded,
            "model_dir": str(self.model_dir),
            "cache_size": self.get_cache_size(),
        }

        if self.is_loaded and self.classifier.metadata:
            info.update(self.classifier.metadata.to_dict())

        return info


# Convenience function for quick usage
def create_classifier(
    model_dir: Optional[Path] = None,
    auto_train: bool = False,
    n_training_samples: int = 1000,
) -> BERTDarkTriadClassifier:
    """
    Create and optionally train a BERT Dark Triad classifier

    Args:
        model_dir: Directory for model persistence
        auto_train: Train if no model found
        n_training_samples: Samples for auto-training

    Returns:
        Initialized BERTDarkTriadClassifier
    """
    return BERTDarkTriadClassifier(
        model_dir=model_dir,
        auto_load=True,
        auto_train=auto_train,
        n_training_samples=n_training_samples,
    )
