"""
KoBERT Emotion Classifier
SPEC-NLP-KOBERT-001 TASK-002: Emotion classification system
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

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


class KoBERTEmotionClassifier:
    """
    KoBERT 기반 한국어 감정 분류기

    Features:
        - 7개 감정 카테고리 분류
        - 신뢰도 점수 계산
        - 불확실성 플래그
        - 배치 처리 지원
        - 성능 모니터링
    """

    EMOTION_LABELS = [
        "happiness",  # 행복
        "sadness",  # 슬픔
        "anger",  # 분노
        "fear",  # 공포
        "disgust",  # 혐오
        "surprise",  # 놀람
        "neutral",  # 중립
    ]

    def __init__(
        self,
        model_name: str = "skt/kobert-base-v1",
        device: str = "auto",
        confidence_threshold: float = 0.7,
        max_length: int = 128,
    ) -> None:
        """
        Initialize emotion classifier

        Args:
            model_name: KoBERT model name or path
            device: Device type ("auto", "cuda", "cpu")
            confidence_threshold: Threshold for uncertainty flag
            max_length: Maximum sequence length
        """
        from voice_man.services.nlp.kobert_model import KoBERTModel

        self.model_name = model_name
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.max_length = max_length

        # Load KoBERT model
        self.kobert = KoBERTModel(device=device, model_name=model_name, max_length=max_length)

        # Initialize classification head
        self.classification_head = EmotionClassificationHead(
            hidden_size=768, num_labels=len(self.EMOTION_LABELS)
        )
        self.classification_head.to(self.kobert._device_obj)

        # Load fine-tuned weights if available
        self._load_fine_tuned_weights()

        self.classification_head.eval()

        logger.info(f"KoBERTEmotionClassifier initialized on {self.kobert.device.value}")

    def _load_fine_tuned_weights(self):
        """Load fine-tuned weights if available"""
        # TODO: Load from fine_tuned_path in config if available
        pass

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

        if not self.kobert.is_loaded():
            raise RuntimeError("Model is not loaded")

        start_time = time.time()

        # Get embeddings
        outputs = self.kobert.encode(text)

        # Get pooled output (use [CLS] token representation)
        pooled_output = outputs.last_hidden_state[:, 0, :]

        # Classify
        with torch.no_grad():
            logits = self.classification_head(pooled_output)
            probs = torch.softmax(logits, dim=-1)

        # Get predictions
        confidence, primary_idx = torch.max(probs, dim=-1)
        primary_emotion = self.EMOTION_LABELS[primary_idx.item()]

        # Create emotion scores dictionary
        emotion_scores = {label: probs[0, i].item() for i, label in enumerate(self.EMOTION_LABELS)}

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

        if not self.kobert.is_loaded():
            raise RuntimeError("Model is not loaded")

        results = []
        start_time = time.time()

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            # Get embeddings for batch
            outputs_list = self.kobert.encode_batch(batch_texts)

            # Process each output
            for outputs in outputs_list:
                pooled_output = outputs.last_hidden_state[:, 0, :]

                with torch.no_grad():
                    logits = self.classification_head(pooled_output)
                    probs = torch.softmax(logits, dim=-1)

                confidence, primary_idx = torch.max(probs, dim=-1)
                primary_emotion = self.EMOTION_LABELS[primary_idx.item()]

                emotion_scores = {
                    label: probs[0, i].item() for i, label in enumerate(self.EMOTION_LABELS)
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
            tokens = self.kobert._tokenizer.tokenize(text)
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

    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            "model_name": self.model_name,
            "device": self.kobert.device.value,
            "num_labels": len(self.EMOTION_LABELS),
            "emotion_labels": self.EMOTION_LABELS,
            "confidence_threshold": self.confidence_threshold,
            "max_length": self.max_length,
        }
