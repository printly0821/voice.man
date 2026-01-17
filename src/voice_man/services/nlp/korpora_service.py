"""
Korpora Integration Service
SPEC-NLP-KOBERT-001 TAG-002: Korean corpus integration using Korpora package

This service provides access to various Korean language datasets through the Korpora package,
enabling enhanced text analysis and training data augmentation.

Reference: https://github.com/ko-nlp/Korpora

Supported Datasets:
- NSMC (Naver Sentiment Movie Corpus): Movie review sentiment classification
- Korean hate speech: Hate speech detection
- Presidential petition data: Formal document analysis
- KcBERT training data: Comment-based text
"""

import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class KorpusDataset(str, Enum):
    """Supported Korpora datasets"""

    NSMC = "nsmc"  # Naver Sentiment Movie Corpus
    KOREAN_HATE_SPEECH = "korean_hate_speech"  # Hate speech comments
    PRESIDENTIAL_PETITION = "presidential_petition"  # Cheongwadae petitions
    KCBERT = "kcbert"  # KcBERT training data
    KOREAN_NICKNAME = "korean_nickname"  # Korean nicknames
    KOREAN_PAIR = "korean_pair"  # Parallel corpus
    QUESTION_PAIR = "question_pair"  # Question answering pairs
    NAVER_ENTITY = "naver_entity"  # Naver named entity recognition
    NAMU_WIKI = "namuwiki"  # Namuwiki dump


@dataclass
class DatasetInfo:
    """Dataset information"""

    name: str
    description: str
    size: Optional[int] = None
    num_classes: Optional[int] = None
    classes: Optional[List[str]] = None
    url: Optional[str] = None


class KorporaService:
    """
    Korpora integration service for Korean language datasets.

    Features:
        - Unified access to multiple Korean datasets
        - Lazy loading (datasets loaded on demand)
        - Training data augmentation support
        - Sentiment analysis benchmarking
        - Hate speech detection training

    Usage:
        service = KorporaService()

        # Load NSMC dataset
        nsmc = service.load_corpus(KorpusDataset.NSMC)

        # Get dataset info
        info = service.get_dataset_info(KorpusDataset.NSMC)

        # Get training samples
        train_texts = service.get_training_texts(KorpusDataset.NSMC, limit=1000)
    """

    # Dataset metadata
    DATASET_INFO = {
        KorpusDataset.NSMC: DatasetInfo(
            name="Naver Sentiment Movie Corpus",
            description="Korean movie review sentiment classification",
            size=200000,
            num_classes=2,
            classes=["negative", "positive"],
            url="https://github.com/e9t/nsmc",
        ),
        KorpusDataset.KOREAN_HATE_SPEECH: DatasetInfo(
            name="Korean Hate Speech",
            description="Korean hate speech comments",
            size=None,
            num_classes=None,
            classes=None,
            url="https://github.com/kocohub/korean-hate-speech",
        ),
        KorpusDataset.PRESIDENTIAL_PETITION: DatasetInfo(
            name="Presidential Petition",
            description="Cheongwadae presidential petition data",
            size=None,
            num_classes=None,
            classes=None,
            url="https://github.com/lovit/petitions",
        ),
        KorpusDataset.KCBERT: DatasetInfo(
            name="KcBERT Training Data",
            description="Korean comment-based text data",
            size=None,
            num_classes=None,
            classes=None,
            url="https://github.com/beomi/KcBERT",
        ),
    }

    def __init__(self):
        """Initialize Korpora service"""
        self._loaded_corpora: Dict[KorpusDataset, Any] = {}
        self._available_datasets: List[KorpusDataset] = []
        self._check_availability()

    def _check_availability(self):
        """Check which Korpora datasets are available"""
        try:
            from Korpora import Korpora

            # Check each dataset
            for dataset in KorpusDataset:
                try:
                    Korpora.load(dataset.value)
                    self._available_datasets.append(dataset)
                    logger.info(f"Korpora dataset available: {dataset.value}")
                except Exception as e:
                    logger.debug(f"Korpora dataset not available: {dataset.value} - {e}")
        except ImportError:
            logger.warning("Korpora package not installed. Install with: pip install Korpora")

    def is_available(self, dataset: KorpusDataset) -> bool:
        """
        Check if a specific dataset is available.

        Args:
            dataset: Dataset to check

        Returns:
            True if available, False otherwise
        """
        return dataset in self._available_datasets

    def get_available_datasets(self) -> List[KorpusDataset]:
        """
        Get list of available datasets.

        Returns:
            List of available dataset enums
        """
        return self._available_datasets.copy()

    def get_dataset_info(self, dataset: KorpusDataset) -> Optional[DatasetInfo]:
        """
        Get dataset information.

        Args:
            dataset: Dataset to query

        Returns:
            DatasetInfo or None if not found
        """
        return self.DATASET_INFO.get(dataset)

    def load_corpus(self, dataset: KorpusDataset, force_reload: bool = False):
        """
        Load a Korpora dataset.

        Args:
            dataset: Dataset to load
            force_reload: Force reload even if already loaded

        Returns:
            Loaded corpus object
        """
        if force_reload or dataset not in self._loaded_corpora:
            try:
                from Korpora import Korpora

                logger.info(f"Loading Korpora dataset: {dataset.value}")
                corpus = Korpora.load(dataset.value)
                self._loaded_corpora[dataset] = corpus
                return corpus
            except ImportError:
                raise RuntimeError(
                    "Korpora package not installed. Install with: pip install Korpora"
                )
            except Exception as e:
                logger.error(f"Failed to load dataset {dataset.value}: {e}")
                raise

        return self._loaded_corpora[dataset]

    def get_training_texts(
        self,
        dataset: KorpusDataset,
        split: str = "train",
        limit: Optional[int] = None,
        text_field: str = "text",
    ) -> List[str]:
        """
        Get training texts from a dataset.

        Args:
            dataset: Dataset to load
            split: Train/test split (default: "train")
            limit: Maximum number of texts to return
            text_field: Field name containing text data

        Returns:
            List of text strings
        """
        corpus = self.load_corpus(dataset)

        texts = []

        # NSMC specific
        if dataset == KorpusDataset.NSMC:
            if hasattr(corpus, "train"):
                texts = [getattr(item, text_field, item) for item in corpus.train]
            elif hasattr(corpus, "get_all_texts"):
                texts = corpus.get_all_texts()

        # Generic fallback
        if not texts and hasattr(corpus, "train"):
            texts = list(corpus.train)

        if limit:
            texts = texts[:limit]

        return texts

    def get_labeled_data(
        self,
        dataset: KorpusDataset,
        split: str = "train",
        limit: Optional[int] = None,
        text_field: str = "text",
        label_field: str = "label",
    ) -> List[tuple[str, Any]]:
        """
        Get labeled data (text, label) pairs.

        Args:
            dataset: Dataset to load
            split: Train/test split
            limit: Maximum number of samples
            text_field: Field name for text
            label_field: Field name for label

        Returns:
            List of (text, label) tuples
        """
        corpus = self.load_corpus(dataset)

        labeled_data = []

        # NSMC specific
        if dataset == KorpusDataset.NSMC:
            if hasattr(corpus, "train"):
                labeled_data = [
                    (getattr(item, text_field, item), getattr(item, label_field, item))
                    for item in corpus.train
                ]
            elif hasattr(corpus, "get_all_texts"):
                texts = corpus.get_all_texts()
                labels = corpus.get_all_labels()
                labeled_data = list(zip(texts, labels))

        # Generic fallback
        if not labeled_data and hasattr(corpus, "train"):
            labeled_data = [
                (getattr(item, text_field, item), getattr(item, label_field, item))
                for item in corpus.train
            ]

        if limit:
            labeled_data = labeled_data[:limit]

        return labeled_data

    def get_sentiment_distribution(self, dataset: KorpusDataset) -> Dict[str, int]:
        """
        Get sentiment label distribution for a dataset.

        Args:
            dataset: Dataset to analyze

        Returns:
            Dictionary mapping labels to counts
        """
        labeled_data = self.get_labeled_data(dataset)

        distribution = {}
        for _, label in labeled_data:
            label_str = str(label)
            distribution[label_str] = distribution.get(label_str, 0) + 1

        return distribution

    def augment_training_data(
        self,
        texts: List[str],
        dataset: KorpusDataset = KorpusDataset.NSMC,
        num_samples: int = 100,
    ) -> List[str]:
        """
        Augment training data with Korpora samples.

        Args:
            texts: Original training texts
            dataset: Korpora dataset to use for augmentation
            num_samples: Number of samples to add

        Returns:
            Augmented list of texts
        """
        additional_texts = self.get_training_texts(dataset, limit=num_samples)

        return texts + additional_texts

    def benchmark_model(
        self,
        model_predict_fn,
        dataset: KorpusDataset = KorpusDataset.NSMC,
        sample_size: int = 1000,
    ) -> Dict[str, float]:
        """
        Benchmark a model on a Korpora dataset.

        Args:
            model_predict_fn: Function that takes text and returns prediction
            dataset: Dataset to benchmark on
            sample_size: Number of samples to evaluate

        Returns:
            Dictionary with metrics (accuracy, etc.)
        """
        labeled_data = self.get_labeled_data(dataset, limit=sample_size)

        if not labeled_data:
            return {"error": "No labeled data available"}

        correct = 0
        total = 0

        for text, true_label in labeled_data:
            try:
                prediction = model_predict_fn(text)
                if prediction == true_label:
                    correct += 1
                total += 1
            except Exception as e:
                logger.warning(f"Prediction failed for text: {e}")

        if total == 0:
            return {"error": "No successful predictions"}

        accuracy = correct / total

        return {
            "accuracy": accuracy,
            "total_samples": total,
            "correct_predictions": correct,
            "dataset": dataset.value,
        }

    def export_for_training(
        self,
        dataset: KorpusDataset,
        output_path: str,
        split: str = "train",
        format: str = "jsonl",
    ):
        """
        Export dataset for external training.

        Args:
            dataset: Dataset to export
            output_path: Output file path
            split: Train/test split
            format: Output format (jsonl, csv, tsv)
        """
        labeled_data = self.get_labeled_data(dataset, split=split)

        import json

        with open(output_path, "w", encoding="utf-8") as f:
            if format == "jsonl":
                for text, label in labeled_data:
                    json.dump({"text": text, "label": label}, f, ensure_ascii=False)
                    f.write("\n")
            else:
                raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Exported {len(labeled_data)} samples to {output_path}")

    def get_service_status(self) -> Dict[str, Any]:
        """
        Get service status information.

        Returns:
            Dictionary with service status
        """
        return {
            "korpora_installed": len(self._available_datasets) > 0,
            "available_datasets": [d.value for d in self._available_datasets],
            "loaded_datasets": [d.value for d in self._loaded_corpora.keys()],
            "total_available": len(self._available_datasets),
        }


# Singleton instance
_korpora_service_instance: Optional[KorporaService] = None


def get_korpora_service() -> KorporaService:
    """
    Get singleton KorporaService instance.

    Returns:
        KorporaService instance
    """
    global _korpora_service_instance

    if _korpora_service_instance is None:
        _korpora_service_instance = KorporaService()

    return _korpora_service_instance
