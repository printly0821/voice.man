"""
BERT Feature Extractor for Dark Triad Classification
SPEC-CRIME-CLASS-001 Enhancement: BERT + Random Forest Hybrid

Extracts text embeddings using KoBERT (Korean BERT) for personality trait classification.
"""

import hashlib
import json
import pickle
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
import torch


class BERTFeatureExtractor:
    """
    BERT-based feature extractor for Korean text analysis

    Uses KoBERT to extract 768-dimensional text embeddings for personality classification.
    Supports caching and batch processing for efficiency.

    Attributes:
        model_name: Pre-trained KoBERT model name
        device: Torch device (cuda/cpu)
        max_length: Maximum token sequence length
        cache_dir: Directory for caching embeddings
        embedding_dim: Output embedding dimension (768 for BERT-base)
    """

    def __init__(
        self,
        model_name: str = "monologg/kobert",
        device: Optional[str] = None,
        max_length: int = 512,
        cache_dir: Optional[Path] = None,
    ) -> None:
        """
        Initialize BERT feature extractor

        Args:
            model_name: HuggingFace model identifier
            device: Torch device (auto-detect if None)
            max_length: Maximum sequence length for tokenization
            cache_dir: Cache directory for embeddings (disabled if None)
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.cache_dir = cache_dir or Path("models/crime_classification/bert_rf/cache")
        self.embedding_dim = 768  # BERT-base dimension

        # Create cache directory if needed
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Lazy loading: model loaded on first use
        self._tokenizer: Optional[Any] = None
        self._model: Optional[Any] = None
        self._is_loaded = False

    def _load_model(self) -> None:
        """
        Lazy load KoBERT model and tokenizer

        Model is loaded only when first needed to reduce startup time.
        """
        if self._is_loaded:
            return

        try:
            from transformers import BertModel, BertTokenizer

            self._tokenizer = BertTokenizer.from_pretrained(self.model_name)
            self._model = BertModel.from_pretrained(self.model_name)
            self._model.to(self.device)
            self._model.eval()  # Set to evaluation mode
            self._is_loaded = True

        except ImportError as e:
            raise ImportError(
                "transformers library is required. Install with: pip install transformers"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to load KoBERT model: {e}") from e

    def _get_cache_key(self, text: str) -> str:
        """
        Generate cache key from text

        Args:
            text: Input text

        Returns:
            MD5 hash of text for cache key
        """
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def _load_from_cache(self, cache_key: str) -> Optional[np.ndarray]:
        """
        Load embedding from cache

        Args:
            cache_key: Cache key for embedding

        Returns:
            Cached embedding if exists, None otherwise
        """
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        if cache_file.exists():
            try:
                with cache_file.open("rb") as f:
                    return pickle.load(f)
            except Exception:
                # Cache corruption - ignore and recompute
                pass

        return None

    def _save_to_cache(self, cache_key: str, embedding: np.ndarray) -> None:
        """
        Save embedding to cache

        Args:
            cache_key: Cache key for embedding
            embedding: Embedding to cache
        """
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        try:
            with cache_file.open("wb") as f:
                pickle.dump(embedding, f)
        except Exception:
            # Cache write failure - not critical
            pass

    def extract_embedding(self, text: str, use_cache: bool = True) -> np.ndarray:
        """
        Extract BERT embedding for single text

        Args:
            text: Input text to extract embedding from
            use_cache: Whether to use embedding cache

        Returns:
            768-dimensional numpy array (text embedding)
        """
        # Check cache first
        if use_cache:
            cache_key = self._get_cache_key(text)
            cached = self._load_from_cache(cache_key)
            if cached is not None:
                return cached

        # Ensure model is loaded
        if not self._is_loaded:
            self._load_model()

        # Tokenize text
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Extract embedding
        with torch.no_grad():
            outputs = self._model(**inputs)
            # Use [CLS] token embedding (first token)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        # Convert to 1D array
        embedding = embedding.flatten()

        # Cache result
        if use_cache:
            self._save_to_cache(cache_key, embedding)

        return embedding

    def extract_embeddings_batch(
        self, texts: List[str], use_cache: bool = True, batch_size: int = 8
    ) -> np.ndarray:
        """
        Extract BERT embeddings for multiple texts

        Args:
            texts: List of input texts
            use_cache: Whether to use embedding cache
            batch_size: Batch size for processing

        Returns:
            2D numpy array of shape (len(texts), 768)
        """
        embeddings = []

        # Process in batches for efficiency
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            # Try cache for each text
            batch_embeddings = []
            uncached_indices = []
            uncached_texts = []

            for j, text in enumerate(batch_texts):
                if use_cache:
                    cache_key = self._get_cache_key(text)
                    cached = self._load_from_cache(cache_key)
                    if cached is not None:
                        batch_embeddings.append((j, cached))
                        continue

                # Not in cache
                uncached_indices.append(j)
                uncached_texts.append(text)

            # Process uncached texts
            if uncached_texts:
                # Ensure model is loaded
                if not self._is_loaded:
                    self._load_model()

                # Tokenize batch
                inputs = self._tokenizer(
                    uncached_texts,
                    return_tensors="pt",
                    max_length=self.max_length,
                    truncation=True,
                    padding="max_length",
                )

                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Extract embeddings
                with torch.no_grad():
                    outputs = self._model(**inputs)
                    batch_uncached = (
                        outputs.last_hidden_state[:, 0, :].cpu().numpy()
                    )  # Shape: (batch, 768)

                # Add to batch results
                for idx, embedding in zip(uncached_indices, batch_uncached):
                    batch_embeddings.append((idx, embedding))

                    # Cache result
                    if use_cache:
                        cache_key = self._get_cache_key(uncached_texts[uncached_indices.index(idx)])
                        self._save_to_cache(cache_key, embedding)

            # Sort by original order and extract embeddings
            batch_embeddings.sort(key=lambda x: x[0])
            embeddings.extend([emb for _, emb in batch_embeddings])

        return np.array(embeddings)

    def get_embedding_dimension(self) -> int:
        """
        Get output embedding dimension

        Returns:
            Embedding dimension (768 for BERT-base)
        """
        return self.embedding_dim

    def clear_cache(self) -> None:
        """Clear all cached embeddings"""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()

    def get_cache_size(self) -> int:
        """
        Get number of cached embeddings

        Returns:
            Number of cached embeddings
        """
        return len(list(self.cache_dir.glob("*.pkl")))

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information

        Returns:
            Dictionary with model metadata
        """
        return {
            "model_name": self.model_name,
            "device": self.device,
            "max_length": self.max_length,
            "embedding_dim": self.embedding_dim,
            "cache_dir": str(self.cache_dir),
            "cache_size": self.get_cache_size(),
            "is_loaded": self._is_loaded,
        }
