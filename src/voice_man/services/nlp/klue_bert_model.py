"""
KLUE-BERT Model Wrapper
KLUE-BERT integration with interface compatible to KoBERTModel

Supports:
    - klue/bert-base (110M params, 768 hidden size)
    - klue/roberta-base (125M params, 768 hidden size)

Features:
    - Automatic GPU/CPU detection
    - Singleton pattern for efficient resource usage
    - GPU memory monitoring
    - Graceful fallback to CPU
    - Performance optimization
"""

import logging
import time
import warnings
from enum import Enum
from threading import Lock
from typing import Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


class DeviceType(str, Enum):
    """Device types for model execution"""

    CUDA = "cuda"
    CPU = "cpu"


class KLUEModelVariant(str, Enum):
    """KLUE-BERT model variants"""

    BERT_BASE = "klue/bert-base"
    ROBERTA_BASE = "klue/roberta-base"


class KLUEBERTModel:
    """
    KLUE-BERT model wrapper with automatic GPU/CPU detection

    This implementation mirrors KoBERTModel interface for easy swapping.

    Features:
        - Automatic device detection (GPU/CPU)
        - Singleton pattern for efficient resource usage
        - GPU memory monitoring
        - Graceful fallback to CPU
        - Performance optimization

    Model Variants:
        - klue/bert-base: Standard BERT architecture (110M params)
        - klue/roberta-base: RoBERTa architecture (125M params)
    """

    # Class-level instance tracking for singleton pattern
    _instances: Dict[str, "KLUEBERTModel"] = {}
    _lock = Lock()

    def __new__(
        cls,
        device: str = "auto",
        model_name: str = "klue/bert-base",
        max_length: int = 128,
    ) -> "KLUEBERTModel":
        """
        Singleton pattern implementation with model-specific instances

        Args:
            device: Device type ("auto", "cuda", "cpu")
            model_name: Model name or path
            max_length: Maximum sequence length

        Returns:
            KLUEBERTModel instance
        """
        # Use model_name as part of the instance key
        instance_key = f"{model_name}_{device}_{max_length}"

        with cls._lock:
            if instance_key not in cls._instances:
                instance = super().__new__(cls)
                instance._initialized = False
                cls._instances[instance_key] = instance
            return cls._instances[instance_key]

    def __init__(
        self,
        device: str = "auto",
        model_name: str = "klue/bert-base",
        max_length: int = 128,
    ):
        """
        Initialize KLUE-BERT model

        Args:
            device: Device type ("auto", "cuda", "cpu")
            model_name: Model name or path (klue/bert-base or klue/roberta-base)
            max_length: Maximum sequence length
        """
        if self._initialized:
            return

        self._initialized = True
        self.model_name = model_name
        self.max_length = max_length

        # Device detection and setup
        self.device = self._detect_device(device)
        self._device_obj = self._get_device_object()

        # Model loading
        self._model = None
        self._tokenizer = None
        self._loading = False

        try:
            self._load_model()
        except Exception as e:
            logger.error(f"Failed to load KLUE-BERT model: {e}")
            raise RuntimeError(f"Failed to load KLUE-BERT model: {e}") from e

    def _detect_device(self, device: str) -> DeviceType:
        """
        Detect and validate device

        Args:
            device: Requested device

        Returns:
            Actual device to use
        """
        if device == "auto":
            if torch.cuda.is_available():
                logger.info("CUDA detected, using GPU")
                return DeviceType.CUDA
            else:
                warnings.warn(
                    "CUDA not available, falling back to CPU. GPU inference will not be available.",
                    UserWarning,
                    stacklevel=2,
                )
                logger.warning("CUDA not available, using CPU")
                return DeviceType.CPU

        elif device == "cuda":
            if torch.cuda.is_available():
                return DeviceType.CUDA
            else:
                warnings.warn(
                    "CUDA requested but not available, falling back to CPU",
                    UserWarning,
                    stacklevel=2,
                )
                return DeviceType.CPU

        elif device == "cpu":
            return DeviceType.CPU

        else:
            warnings.warn(
                f"Invalid device '{device}', falling back to CPU",
                UserWarning,
                stacklevel=2,
            )
            return DeviceType.CPU

    def _get_device_object(self) -> torch.device:
        """Get torch.device object"""
        return torch.device(self.device.value)

    def _load_model(self):
        """Load KLUE-BERT model and tokenizer"""
        self._loading = True

        try:
            from transformers import AutoModel, AutoTokenizer

            logger.info(f"Loading KLUE-BERT model: {self.model_name}")

            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # Load model
            self._model = AutoModel.from_pretrained(self.model_name)
            self._model.to(self._device_obj)
            self._model.eval()

            logger.info(f"KLUE-BERT model loaded successfully on {self.device.value}")

        except Exception as e:
            logger.error(f"Error loading KLUE-BERT model: {e}")
            raise
        finally:
            self._loading = False

    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self._model is not None and self._tokenizer is not None

    def is_loading(self) -> bool:
        """Check if model is currently loading"""
        return self._loading

    def encode(self, text: str):
        """
        Encode single text

        Args:
            text: Input text

        Returns:
            Model output with last_hidden_state
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        if not self.is_loaded():
            raise RuntimeError("Model is not loaded")

        # Tokenize
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
        )

        # Move to device
        inputs = {k: v.to(self._device_obj) for k, v in inputs.items()}

        # Encode
        with torch.no_grad():
            outputs = self._model(**inputs)

        return outputs

    def encode_batch(self, texts: List[str]):
        """
        Encode multiple texts

        Args:
            texts: List of input texts

        Returns:
            List of model outputs
        """
        if not texts:
            return []

        if not self.is_loaded():
            raise RuntimeError("Model is not loaded")

        # Tokenize batch
        inputs = self._tokenizer(
            texts,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
        )

        # Move to device
        inputs = {k: v.to(self._device_obj) for k, v in inputs.items()}

        # Encode
        with torch.no_grad():
            outputs = self._model(**inputs)

        # Split batch results
        results = []
        for i in range(len(texts)):
            result = type(outputs)(
                last_hidden_state=outputs.last_hidden_state[i : i + 1],
                pooler_output=outputs.pooler_output[i : i + 1]
                if hasattr(outputs, "pooler_output")
                else None,
            )
            results.append(result)

        return results

    def get_embeddings(self, text: str) -> torch.Tensor:
        """
        Get text embeddings

        Args:
            text: Input text

        Returns:
            Embedding tensor
        """
        outputs = self.encode(text)
        return outputs.last_hidden_state

    def tokenize(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Tokenize text

        Args:
            text: Input text

        Returns:
            Tokenized inputs
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        return self._tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
        )

    def get_gpu_memory_info(self) -> Optional[Dict[str, float]]:
        """
        Get GPU memory information

        Returns:
            Dict with 'total' and 'free' memory in MB, or None if CPU
        """
        if self.device != DeviceType.CUDA:
            return None

        try:
            total = torch.cuda.get_device_properties(0).total_memory / (1024**2)
            free = (
                torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
            ) / (1024**2)

            return {"total": total, "free": free}
        except Exception as e:
            logger.warning(f"Failed to get GPU memory info: {e}")
            return None

    def get_device_info(self) -> str:
        """Get device information string"""
        if self.device == DeviceType.CUDA:
            try:
                props = torch.cuda.get_device_properties(0)
                return f"CUDA: {props.name} ({props.total_memory / (1024**3):.1f}GB)"
            except Exception:
                return "CUDA (Unknown)"
        else:
            return "CPU"

    def get_model_info(self) -> Dict[str, any]:
        """Get model information"""
        # Determine hidden size based on model variant
        hidden_size = 768  # Default for both klue/bert-base and klue/roberta-base

        return {
            "model_type": "klue-bert",
            "model_name": self.model_name,
            "device": self.device.value,
            "hidden_size": hidden_size,
            "max_length": self.max_length,
            "is_loaded": self.is_loaded(),
        }

    def clear_cache(self):
        """Clear model cache"""
        if self.device == DeviceType.CUDA:
            torch.cuda.empty_cache()

    def warmup(self):
        """Perform warmup inference"""
        if self.is_loaded():
            self.encode("Warmup text")
