"""
Unified BERT Emotion Classifier
SPEC-NLP-KOBERT-001 TASK-002: Emotion classification system

Supports both KoBERT and KLUE-BERT models with fine-tuned models for
Korean emotion classification with 6 or 7 emotion categories.

Features:
    - Model selection: KoBERT, KLUE-BERT, KLUE-RoBERTa
    - Auto-selection from configuration
    - Hot-swapping models
    - Backward compatibility with KoBERT
"""

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class EmotionResult:
    """감정 분류 결과"""

    primary_emotion: str  # 주요 감정
    confidence: float  # 신뢰도 (0.0-1.0)
    emotion_scores: Dict[str, float]  # 전체 감정 확률 분포
    is_uncertain: bool  # 불확실성 플래그
    key_tokens: Optional[List[str]] = None  # 핵심 토큰 (설명 가능성)


class EmotionClassificationHead(nn.Module):
    """Emotion classification head on top of KoBERT"""

    def __init__(self, hidden_size: int = 768, num_labels: int = 7, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, pooled_output: torch.Tensor) -> torch.Tensor:
        """Forward pass through classification head"""
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


class UnifiedBERTEmotionClassifier:
    """
    Unified BERT-based Korean emotion classifier supporting multiple models

    Features:
        - Model selection: KoBERT, KLUE-BERT, KLUE-RoBERTa
        - Auto-selection from configuration or environment
        - 6 or 7개 감정 카테고리 분류
        - 신뢰도 점수 계산
        - 불확실성 플래그
        - 배치 처리 지원
        - 성능 모니터링
        - Fine-tuned 모델 지원
        - Hot-swapping models

    Supported emotion categories:
        - 6 emotions: happy, sad, angry, fear, disgust, neutral
        - 7 emotions: happiness, sadness, anger, fear, disgust, surprise, neutral

    Supported models:
        - kobert: skt/kobert-base-v1 (default, backward compatible)
        - klue_bert: klue/bert-base
        - klue_roberta: klue/roberta-base
    """

    EMOTION_LABELS_7 = [
        "happiness",  # 행복
        "sadness",  # 슬픔
        "anger",  # 분노
        "fear",  # 공포
        "disgust",  # 혐오
        "surprise",  # 놀람
        "neutral",  # 중립
    ]

    EMOTION_LABELS_6 = [
        "happy",  # 행복
        "sad",  # 슬픔
        "angry",  # 분노
        "fear",  # 공포
        "disgust",  # 혐오
        "neutral",  # 중립
    ]

    # Default model configurations
    MODEL_CONFIGS = {
        "kobert": "skt/kobert-base-v1",
        "klue_bert": "klue/bert-base",
        "klue_roberta": "klue/roberta-base",
    }

    # Default to 7 for backward compatibility
    EMOTION_LABELS = EMOTION_LABELS_7

    def __init__(
        self,
        model_type: Literal["kobert", "klue_bert", "klue_roberta", "auto"] = "auto",
        model_name: Optional[str] = None,
        device: str = "auto",
        confidence_threshold: float = 0.7,
        max_length: int = 128,
        fine_tuned_path: Optional[Union[str, Path]] = None,
        num_labels: int = 7,
    ) -> None:
        """
        Initialize unified BERT emotion classifier

        Args:
            model_type: Model type ("kobert", "klue_bert", "klue_roberta", "auto")
                       "auto" selects from VOICE_MAN_BERT_MODEL env var or defaults to "kobert"
            model_name: Override model name (optional)
            device: Device type ("auto", "cuda", "cpu")
            confidence_threshold: Threshold for uncertainty flag
            max_length: Maximum sequence length
            fine_tuned_path: Path to fine-tuned model weights (optional)
            num_labels: Number of emotion labels (6 or 7)
        """
        # Determine model type
        if model_type == "auto":
            model_type = os.getenv("VOICE_MAN_BERT_MODEL", "kobert").lower()
            logger.info(f"Auto-selected model type: {model_type}")

        # Validate model type
        if model_type not in self.MODEL_CONFIGS and model_type != "kobert":
            logger.warning(f"Unknown model_type '{model_type}', falling back to 'kobert'")
            model_type = "kobert"

        self.model_type = model_type

        # Get model name
        if model_name is None:
            # Map legacy model names to model types
            if "kobert" in model_type or model_type == "kobert":
                model_name = self.MODEL_CONFIGS["kobert"]
            elif "klue" in model_type and "roberta" in model_type:
                model_name = self.MODEL_CONFIGS["klue_roberta"]
            elif "klue" in model_type:
                model_name = self.MODEL_CONFIGS["klue_bert"]
            else:
                model_name = self.MODEL_CONFIGS.get(model_type, self.MODEL_CONFIGS["kobert"])

        self.model_name = model_name
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.max_length = max_length
        self.fine_tuned_path = Path(fine_tuned_path) if fine_tuned_path else None

        # Set emotion labels based on num_labels
        if num_labels == 6:
            self.emotion_labels = self.EMOTION_LABELS_6
        elif num_labels == 7:
            self.emotion_labels = self.EMOTION_LABELS_7
        else:
            raise ValueError(f"num_labels must be 6 or 7, got {num_labels}")

        self.num_labels = num_labels

        # Load appropriate BERT model
        self.bert_model = self._load_bert_model()

        # Initialize classification head
        self.classification_head = EmotionClassificationHead(
            hidden_size=768, num_labels=self.num_labels
        )
        self.classification_head.to(self.bert_model._device_obj)

        # Load fine-tuned weights if available
        self._load_fine_tuned_weights()

        self.classification_head.eval()

        logger.info(
            f"UnifiedBERTEmotionClassifier initialized with {self.model_type} "
            f"({self.model_name}) on {self.bert_model.device.value} "
            f"with {self.num_labels} emotion labels"
        )

        # Track cleanup state for idempotency
        self._cleaned_up = False

    def _load_bert_model(self):
        """Load appropriate BERT model based on model_type"""
        if "klue" in self.model_type.lower() and "roberta" in self.model_name:
            from voice_man.services.nlp.klue_bert_model import KLUEBERTModel

            logger.info(f"Loading KLUE-RoBERTa model: {self.model_name}")
            return KLUEBERTModel(
                device=self.device, model_name=self.model_name, max_length=self.max_length
            )

        elif "klue" in self.model_type.lower():
            from voice_man.services.nlp.klue_bert_model import KLUEBERTModel

            logger.info(f"Loading KLUE-BERT model: {self.model_name}")
            return KLUEBERTModel(
                device=self.device, model_name=self.model_name, max_length=self.max_length
            )

        else:  # Default to KoBERT
            from voice_man.services.nlp.kobert_model import KoBERTModel

            logger.info(f"Loading KoBERT model: {self.model_name}")
            return KoBERTModel(
                device=self.device, model_name=self.model_name, max_length=self.max_length
            )

    def swap_model(
        self,
        model_type: Literal["kobert", "klue_bert", "klue_roberta"],
        model_name: Optional[str] = None,
    ) -> bool:
        """
        Hot-swap to a different model

        Args:
            model_type: New model type
            model_name: Override model name (optional)

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Swapping model from {self.model_type} to {model_type}")

            # Update model type
            self.model_type = model_type

            # Get model name
            if model_name is None:
                model_name = self.MODEL_CONFIGS.get(model_type, self.MODEL_CONFIGS["kobert"])

            self.model_name = model_name

            # Clear cache and load new model
            from voice_man.services.nlp.unified_bert import BERTModelFactory

            BERTModelFactory.clear_cache()

            # Load new model
            self.bert_model = self._load_bert_model()

            # Reinitialize classification head on new device
            self.classification_head = EmotionClassificationHead(
                hidden_size=768, num_labels=self.num_labels
            )
            self.classification_head.to(self.bert_model._device_obj)

            # Reload fine-tuned weights if available
            self._load_fine_tuned_weights()

            self.classification_head.eval()

            logger.info(f"Model swapped successfully to {model_type}")
            return True

        except Exception as e:
            logger.error(f"Failed to swap model: {e}")
            return False

    def get_current_model_info(self) -> Dict:
        """Get current model information"""
        return {
            "model_type": self.model_type,
            "model_name": self.model_name,
            "device": self.bert_model.device.value,
            "num_labels": self.num_labels,
            "emotion_labels": self.emotion_labels,
            "confidence_threshold": self.confidence_threshold,
            "max_length": self.max_length,
            "fine_tuned_path": str(self.fine_tuned_path) if self.fine_tuned_path else None,
        }

    def _load_fine_tuned_weights(self) -> bool:
        """
        Load fine-tuned weights if available.

        Returns:
            True if fine-tuned weights were loaded, False otherwise
        """
        if self.fine_tuned_path is None:
            logger.debug("No fine-tuned model path provided, using pre-trained model")
            return False

        model_path = Path(self.fine_tuned_path)

        if not model_path.exists():
            logger.warning(f"Fine-tuned model not found at {model_path}, using pre-trained model")
            return False

        try:
            # Try to load metadata first
            metadata_path = model_path.parent / f"{model_path.stem}_metadata.json"

            if metadata_path.exists():
                # Load metadata to get num_labels
                import json

                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)

                expected_num_labels = metadata.get("num_labels", self.num_labels)

                if expected_num_labels != self.num_labels:
                    logger.warning(
                        f"Fine-tuned model has {expected_num_labels} labels, "
                        f"but classifier initialized with {self.num_labels}. "
                        f"Updating classifier to match fine-tuned model."
                    )
                    self.num_labels = expected_num_labels
                    if expected_num_labels == 6:
                        self.emotion_labels = self.EMOTION_LABELS_6
                    elif expected_num_labels == 7:
                        self.emotion_labels = self.EMOTION_LABELS_7

                    # Reinitialize classification head with correct num_labels
                    self.classification_head = EmotionClassificationHead(
                        hidden_size=768, num_labels=self.num_labels
                    )
                    self.classification_head.to(self.bert_model._device_obj)

            # Load state dict
            state_dict = torch.load(model_path, map_location=self.bert_model._device_obj)
            self.classification_head.load_state_dict(state_dict)

            logger.info(f"Loaded fine-tuned model from {model_path}")
            logger.info(f"Emotion labels: {self.emotion_labels}")
            return True

        except Exception as e:
            logger.error(f"Failed to load fine-tuned model: {e}")
            logger.warning("Falling back to pre-trained model")
            return False

    def reload_finetuned_model(self, fine_tuned_path: Union[str, Path]) -> bool:
        """
        Reload a fine-tuned model.

        Args:
            fine_tuned_path: Path to fine-tuned model weights

        Returns:
            True if successful, False otherwise
        """
        self.fine_tuned_path = Path(fine_tuned_path)
        return self._load_fine_tuned_weights()

    def classify(self, text: str) -> EmotionResult:
        """
        Classify emotion in single text

        Args:
            text: Input text

        Returns:
            EmotionResult with classification results

        Raises:
            ValueError: If text is empty
            RuntimeError: If model is not loaded
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        if not self.bert_model.is_loaded():
            raise RuntimeError("Model is not loaded")

        start_time = time.time()

        # Get embeddings
        outputs = self.bert_model.encode(text)

        # Get pooled output (use [CLS] token representation)
        pooled_output = outputs.last_hidden_state[:, 0, :]

        # Classify
        with torch.no_grad():
            logits = self.classification_head(pooled_output)
            probs = torch.softmax(logits, dim=-1)

        # Get predictions
        confidence, primary_idx = torch.max(probs, dim=-1)
        primary_emotion = self.emotion_labels[primary_idx.item()]

        # Create emotion scores dictionary
        emotion_scores = {label: probs[0, i].item() for i, label in enumerate(self.emotion_labels)}

        # Determine uncertainty
        is_uncertain = confidence.item() < self.confidence_threshold

        # Extract key tokens (optional - attention-based)
        key_tokens = self._extract_key_tokens(text, outputs.last_hidden_state)

        inference_time = (time.time() - start_time) * 1000  # ms
        logger.debug(f"Emotion classification completed in {inference_time:.2f}ms")

        return EmotionResult(
            primary_emotion=primary_emotion,
            confidence=confidence.item(),
            emotion_scores=emotion_scores,
            is_uncertain=is_uncertain,
            key_tokens=key_tokens,
        )

    def classify_batch(self, texts: List[str], batch_size: int = 8) -> List[EmotionResult]:
        """
        Classify emotions in multiple texts

        Args:
            texts: List of input texts
            batch_size: Batch size for processing

        Returns:
            List of EmotionResult
        """
        if not texts:
            return []

        if not self.bert_model.is_loaded():
            raise RuntimeError("Model is not loaded")

        results = []
        start_time = time.time()

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            # Get embeddings for batch
            outputs_list = self.bert_model.encode_batch(batch_texts)

            # Process each output
            for outputs in outputs_list:
                pooled_output = outputs.last_hidden_state[:, 0, :]

                with torch.no_grad():
                    logits = self.classification_head(pooled_output)
                    probs = torch.softmax(logits, dim=-1)

                confidence, primary_idx = torch.max(probs, dim=-1)
                primary_emotion = self.emotion_labels[primary_idx.item()]

                emotion_scores = {
                    label: probs[0, i].item() for i, label in enumerate(self.emotion_labels)
                }

                is_uncertain = confidence.item() < self.confidence_threshold

                results.append(
                    EmotionResult(
                        primary_emotion=primary_emotion,
                        confidence=confidence.item(),
                        emotion_scores=emotion_scores,
                        is_uncertain=is_uncertain,
                        key_tokens=None,
                    )
                )

        inference_time = (time.time() - start_time) * 1000  # ms
        logger.debug(f"Batch emotion classification completed in {inference_time:.2f}ms")

        return results

    def _extract_key_tokens(
        self, text: str, hidden_states: torch.Tensor, top_k: int = 5
    ) -> Optional[List[str]]:
        """
        Extract key tokens using attention-based approach

        Args:
            text: Original text
            hidden_states: Hidden states from model
            top_k: Number of top tokens to return

        Returns:
            List of key tokens or None
        """
        try:
            # Simple implementation: use [CLS] token attention
            # In production, use actual attention weights
            tokens = self.bert_model._tokenizer.tokenize(text)
            return tokens[:top_k] if tokens else None
        except Exception as e:
            logger.warning(f"Failed to extract key tokens: {e}")
            return None

    def _compute_emotion_scores(self, text: str) -> tuple[float, Dict[str, float]]:
        """
        Compute emotion scores (for testing/mocking purposes)

        Args:
            text: Input text

        Returns:
            Tuple of (confidence, emotion_scores)
        """
        result = self.classify(text)
        return result.confidence, result.emotion_scores

    def cleanup(self) -> None:
        """
        Release all resources and clear memory.

        Implements ServiceCleanupProtocol for memory management.
        This method is idempotent and can be called multiple times safely.

        Actions performed:
        - Unloads BERT models from GPU/CPU memory
        - Clears classification head
        - Clears GPU cache
        - Clears model cache
        """
        if self._cleaned_up:
            logger.debug("UnifiedBERTEmotionClassifier already cleaned up, skipping")
            return

        try:
            logger.info("Starting UnifiedBERTEmotionClassifier cleanup...")

            # Cleanup BERT model
            if hasattr(self, "bert_model") and self.bert_model is not None:
                if hasattr(self.bert_model, "unload"):
                    self.bert_model.unload()
                    logger.debug("BERT model unloaded")
                elif hasattr(self.bert_model, "cleanup"):
                    self.bert_model.cleanup()
                    logger.debug("BERT model cleaned up")

            # Cleanup classification head
            if hasattr(self, "classification_head") and self.classification_head is not None:
                del self.classification_head
                self.classification_head = None
                logger.debug("Classification head cleared")

            # Clear BERT model cache
            try:
                from voice_man.services.nlp.unified_bert import BERTModelFactory

                BERTModelFactory.clear_cache()
                logger.debug("BERT model cache cleared")
            except ImportError:
                pass
            except Exception as e:
                logger.warning(f"Failed to clear BERT cache: {e}")

            # Clear GPU cache
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    if hasattr(torch.cuda, "ipc_collect"):
                        torch.cuda.ipc_collect()
                    logger.debug("GPU cache cleared")
            except ImportError:
                pass
            except Exception as e:
                logger.warning(f"Failed to clear GPU cache: {e}")

            # Force Python garbage collection
            import gc

            gc.collect()

            # Mark as cleaned up
            self._cleaned_up = True
            logger.info("UnifiedBERTEmotionClassifier cleanup completed")

        except Exception as e:
            logger.error(f"Error during UnifiedBERTEmotionClassifier cleanup: {e}")
            # Still mark as cleaned up to prevent repeated attempts
            self._cleaned_up = True

    def get_memory_usage(self) -> float:
        """
        Get current memory usage in MB.

        Implements ServiceCleanupProtocol for memory monitoring.

        Returns:
            Memory usage in megabytes
        """
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except Exception as e:
            logger.warning(f"Failed to get memory usage: {e}")
            return 0.0


# Backward compatibility alias
KoBERTEmotionClassifier = UnifiedBERTEmotionClassifier
