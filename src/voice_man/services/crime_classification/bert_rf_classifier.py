"""
BERT + Random Forest Classifier for Dark Triad Traits
SPEC-CRIME-CLASS-001 Enhancement: BERT + Random Forest Hybrid

Combines KoBERT embeddings with Random Forest classification for personality trait prediction.
"""

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix

from voice_man.services.crime_classification.bert_feature_extractor import BERTFeatureExtractor


@dataclass
class TrainingMetrics:
    """
    Training metrics for model evaluation

    Attributes:
        accuracy: Model accuracy on test set
        precision: Precision score (macro average)
        recall: Recall score (macro average)
        f1_score: F1 score (macro average)
        cross_val_scores: Cross-validation scores
        feature_importance: Feature importance rankings
        confusion_matrix: Confusion matrix for multiclass
    """

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    cross_val_scores: List[float]
    feature_importance: Dict[str, float]
    confusion_matrix: List[List[int]]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


@dataclass
class ModelMetadata:
    """
    Model metadata for version tracking

    Attributes:
        version: Model version string
        created_at: Model creation timestamp
        model_type: Type of model (bert_rf)
        embedding_dim: Input embedding dimension
        n_estimators: Number of trees in Random Forest
        max_depth: Maximum depth of trees
        traits: List of Dark Triad traits classified
        training_samples: Number of training samples
        test_samples: Number of test samples
        metrics: Training metrics
    """

    version: str
    created_at: str
    model_type: str
    embedding_dim: int
    n_estimators: int
    max_depth: Optional[int]
    traits: List[str]
    training_samples: int
    test_samples: int
    metrics: TrainingMetrics

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data["metrics"] = self.metrics.to_dict()
        return data


class BERTRandomForestClassifier:
    """
    BERT + Random Forest hybrid classifier for Dark Triad traits

    Uses KoBERT embeddings as features and Random Forest for classification.
    Supports multi-label classification (each trait independently).

    Attributes:
        traits: List of Dark Triad trait names
        models: Dictionary of trait -> RandomForestClassifier
        feature_extractor: BERTFeatureExtractor instance
        metadata: Model metadata
        is_trained: Whether models have been trained
    """

    # Default model hyperparameters
    DEFAULT_N_ESTIMATORS = 100
    DEFAULT_MAX_DEPTH = 10
    DEFAULT_RANDOM_STATE = 42

    def __init__(
        self,
        traits: Optional[List[str]] = None,
        feature_extractor: Optional[BERTFeatureExtractor] = None,
        n_estimators: int = DEFAULT_N_ESTIMATORS,
        max_depth: Optional[int] = DEFAULT_MAX_DEPTH,
        random_state: int = DEFAULT_RANDOM_STATE,
    ) -> None:
        """
        Initialize BERT + RF classifier

        Args:
            traits: List of trait names (default: Dark Triad traits)
            feature_extractor: BERT feature extractor instance
            n_estimators: Number of trees in Random Forest
            max_depth: Maximum depth of trees (None for unlimited)
            random_state: Random seed for reproducibility
        """
        self.traits = traits or ["narcissism", "machiavellianism", "psychopathy"]
        self.feature_extractor = feature_extractor or BERTFeatureExtractor()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

        # Initialize models for each trait
        self.models: Dict[str, CalibratedClassifierCV] = {}
        self.metadata: Optional[ModelMetadata] = None
        self.is_trained = False

    def _create_model(self) -> CalibratedClassifierCV:
        """
        Create a new Random Forest model with calibration

        Returns:
            Calibrated Random Forest classifier
        """
        rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1,  # Use all CPU cores
            class_weight="balanced",  # Handle class imbalance
        )

        # Calibrate for better probability estimates
        calibrated = CalibratedClassifierCV(rf, cv=3, method="isotonic")
        return calibrated

    def train(
        self,
        texts: List[str],
        labels: Dict[str, np.ndarray],
        test_size: float = 0.2,
        cross_validation: int = 5,
    ) -> TrainingMetrics:
        """
        Train Random Forest models for each trait

        Args:
            texts: List of training texts
            labels: Dictionary of trait -> binary labels (0 or 1)
            test_size: Fraction of data for testing
            cross_validation: Number of CV folds

        Returns:
            TrainingMetrics with evaluation results
        """
        # Extract embeddings for all texts
        embeddings = self.feature_extractor.extract_embeddings_batch(texts)

        # Train separate model for each trait
        all_metrics = {}

        for trait in self.traits:
            if trait not in labels:
                continue

            trait_labels = labels[trait]

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                embeddings,
                trait_labels,
                test_size=test_size,
                random_state=self.random_state,
                stratify=trait_labels,
            )

            # Create and train model
            model = self._create_model()
            model.fit(X_train, y_train)
            self.models[trait] = model

            # Evaluate
            y_pred = model.predict(X_test)
            accuracy = np.mean(y_pred == y_test)

            # Cross-validation
            cv = StratifiedKFold(
                n_splits=cross_validation, shuffle=True, random_state=self.random_state
            )
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")

            # Feature importance (from base estimator)
            base_rf = model.calibrated_classifiers_[0].base_estimator
            feature_importance = base_rf.feature_importances_

            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred).tolist()

            all_metrics[trait] = {
                "accuracy": float(accuracy),
                "cv_scores": cv_scores.tolist(),
                "feature_importance": feature_importance.tolist(),
                "confusion_matrix": cm,
            }

        # Create metadata
        self.metadata = ModelMetadata(
            version=datetime.now().strftime("%Y%m%d_%H%M%S"),
            created_at=datetime.now().isoformat(),
            model_type="bert_random_forest",
            embedding_dim=self.feature_extractor.get_embedding_dimension(),
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            traits=self.traits,
            training_samples=len(texts),
            test_samples=int(len(texts) * test_size),
            metrics=TrainingMetrics(
                accuracy=np.mean([m["accuracy"] for m in all_metrics.values()]),
                precision=0.0,  # Computed from classification report
                recall=0.0,
                f1_score=float(np.mean([np.mean(m["cv_scores"]) for m in all_metrics.values()])),
                cross_val_scores=[],
                feature_importance={},
                confusion_matrix=[],
            ),
        )

        self.is_trained = True

        return self.metadata.metrics

    def predict(
        self, texts: List[str], return_probabilities: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Predict Dark Triad traits for texts

        Args:
            texts: List of texts to classify
            return_probabilities: Whether to return probability scores

        Returns:
            Dictionary of trait -> predictions (or probabilities)
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")

        # Extract embeddings
        embeddings = self.feature_extractor.extract_embeddings_batch(texts)

        results = {}
        for trait in self.raits:
            if trait not in self.models:
                continue

            model = self.models[trait]

            if return_probabilities:
                # Get probability of class 1 (presence of trait)
                results[trait] = model.predict_proba(embeddings)[:, 1]
            else:
                results[trait] = model.predict(embeddings)

        return results

    def predict_single(self, text: str, return_probabilities: bool = False) -> Dict[str, float]:
        """
        Predict Dark Triad traits for single text

        Args:
            text: Text to classify
            return_probabilities: Whether to return probability scores

        Returns:
            Dictionary of trait -> score (0-1)
        """
        result = self.predict([text], return_probabilities=return_probabilities)

        # Extract single values
        return {trait: float(values[0]) for trait, values in result.items()}

    def get_feature_importance(self, trait: str) -> np.ndarray:
        """
        Get feature importance for a trait

        Args:
            trait: Trait name

        Returns:
            Feature importance array (768-dimensional)
        """
        if trait not in self.models:
            raise ValueError(f"Trait {trait} not found in models")

        model = self.models[trait]
        base_rf = model.calibrated_classifiers_[0].base_estimator
        return base_rf.feature_importances_

    def save(self, model_dir: Path) -> None:
        """
        Save trained models and metadata

        Args:
            model_dir: Directory to save models
        """
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained model")

        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save models
        for trait, model in self.models.items():
            model_path = model_dir / f"{trait}.pkl"
            import joblib

            joblib.dump(model, model_path)

        # Save metadata
        if self.metadata:
            metadata_path = model_dir / "metadata.json"
            with metadata_path.open("w") as f:
                json.dump(self.metadata.to_dict(), f, indent=2)

    @classmethod
    def load(
        cls, model_dir: Path, feature_extractor: Optional[BERTFeatureExtractor] = None
    ) -> "BERTRandomForestClassifier":
        """
        Load trained models and metadata

        Args:
            model_dir: Directory containing saved models
            feature_extractor: BERT feature extractor (creates new if None)

        Returns:
            Loaded BERTRandomForestClassifier instance
        """
        import joblib

        model_dir = Path(model_dir)

        # Load metadata
        metadata_path = model_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found at {metadata_path}")

        with metadata_path.open("r") as f:
            metadata_dict = json.load(f)

        # Create instance
        feature_extractor = feature_extractor or BERTFeatureExtractor()
        instance = cls(
            traits=metadata_dict["traits"],
            feature_extractor=feature_extractor,
            n_estimators=metadata_dict["n_estimators"],
            max_depth=metadata_dict["max_depth"],
        )

        # Load models
        for trait in metadata_dict["traits"]:
            model_path = model_dir / f"{trait}.pkl"
            if model_path.exists():
                instance.models[trait] = joblib.load(model_path)

        instance.is_trained = True
        instance.metadata = ModelMetadata(**metadata_dict)

        return instance
