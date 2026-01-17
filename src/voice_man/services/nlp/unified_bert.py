"""
Unified BERT Interface
Factory pattern for BERT model selection and management

Provides:
    - Factory pattern for model selection (KoBERT vs KLUE-BERT)
    - Configuration-based model switching
    - A/B testing framework for model comparison
    - Performance benchmarking utilities
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch

logger = logging.getLogger(__name__)


class BERTModelType(str, Enum):
    """Supported BERT model types"""

    KOBERT = "kobert"
    KLUE_BERT = "klue_bert"
    KLUE_ROBERTA = "klue_roberta"


@dataclass
class BenchmarkResult:
    """Benchmark result for model comparison"""

    model_type: str
    model_name: str
    accuracy: Optional[float] = None
    avg_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    memory_mb: float = 0.0
    throughput: float = 0.0  # texts per second
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ModelConfig:
    """Configuration for BERT model"""

    model_type: BERTModelType
    model_name: str
    device: str = "auto"
    max_length: int = 128
    confidence_threshold: float = 0.7
    batch_size: int = 8


class BERTModelFactory:
    """
    Factory for creating BERT model instances

    Provides unified interface for creating different BERT models
    with configuration-based selection and A/B testing support.
    """

    # Default model configurations
    DEFAULT_CONFIGS: Dict[BERTModelType, ModelConfig] = {
        BERTModelType.KOBERT: ModelConfig(
            model_type=BERTModelType.KOBERT,
            model_name="skt/kobert-base-v1",
            device="auto",
            max_length=128,
            confidence_threshold=0.7,
            batch_size=8,
        ),
        BERTModelType.KLUE_BERT: ModelConfig(
            model_type=BERTModelType.KLUE_BERT,
            model_name="klue/bert-base",
            device="auto",
            max_length=128,
            confidence_threshold=0.7,
            batch_size=8,
        ),
        BERTModelType.KLUE_ROBERTA: ModelConfig(
            model_type=BERTModelType.KLUE_ROBERTA,
            model_name="klue/roberta-base",
            device="auto",
            max_length=128,
            confidence_threshold=0.7,
            batch_size=8,
        ),
    }

    _model_cache: Dict[str, Any] = {}

    @classmethod
    def get_model_type_from_env(cls) -> Optional[BERTModelType]:
        """
        Get model type from environment variable

        Returns:
            BERTModelType from VOICE_MAN_BERT_MODEL env var, or None
        """
        env_model = os.getenv("VOICE_MAN_BERT_MODEL", "").lower()
        if not env_model:
            return None

        try:
            return BERTModelType(env_model)
        except ValueError:
            logger.warning(f"Invalid BERT model type from env: {env_model}")
            return None

    @classmethod
    def get_model_from_config(cls, config: ModelConfig) -> Any:
        """
        Create or retrieve model from configuration

        Args:
            config: Model configuration

        Returns:
            Model instance (KoBERTModel or KLUEBERTModel)
        """
        # Create cache key
        cache_key = (
            f"{config.model_type.value}_{config.model_name}_{config.device}_{config.max_length}"
        )

        # Return cached model if available
        if cache_key in cls._model_cache:
            logger.debug(f"Using cached model: {cache_key}")
            return cls._model_cache[cache_key]

        # Create new model instance
        logger.info(f"Creating model: {config.model_name}")

        if config.model_type == BERTModelType.KOBERT:
            from voice_man.services.nlp.kobert_model import KoBERTModel

            model = KoBERTModel(
                device=config.device,
                model_name=config.model_name,
                max_length=config.max_length,
            )

        elif config.model_type in (BERTModelType.KLUE_BERT, BERTModelType.KLUE_ROBERTA):
            from voice_man.services.nlp.klue_bert_model import KLUEBERTModel

            model = KLUEBERTModel(
                device=config.device,
                model_name=config.model_name,
                max_length=config.max_length,
            )

        else:
            raise ValueError(f"Unsupported model type: {config.model_type}")

        # Cache the model
        cls._model_cache[cache_key] = model
        return model

    @classmethod
    def get_model(
        cls,
        model_type: Optional[Union[str, BERTModelType]] = None,
        model_name: Optional[str] = None,
        device: str = "auto",
        max_length: int = 128,
    ) -> Any:
        """
        Get model instance with flexible parameters

        Priority:
            1. Explicit model_type parameter
            2. VOICE_MAN_BERT_MODEL environment variable
            3. Default to KOBERT

        Args:
            model_type: Model type (kobert, klue_bert, klue_roberta)
            model_name: Override model name
            device: Device type (auto, cuda, cpu)
            max_length: Maximum sequence length

        Returns:
            Model instance
        """
        # Determine model type
        if model_type is None:
            model_type = cls.get_model_type_from_env()
            if model_type is None:
                model_type = BERTModelType.KOBERT
                logger.info("No model type specified, using default: KOBERT")
        elif isinstance(model_type, str):
            model_type = BERTModelType(model_type)

        # Get default config
        config = cls.DEFAULT_CONFIGS[model_type]

        # Override parameters if provided
        if model_name:
            config.model_name = model_name
        if device != "auto":
            config.device = device
        if max_length != 128:
            config.max_length = max_length

        return cls.get_model_from_config(config)

    @classmethod
    def clear_cache(cls):
        """Clear model cache"""
        cls._model_cache.clear()
        logger.info("Model cache cleared")


class ABTestFramework:
    """
    A/B testing framework for model comparison

    Provides utilities for running A/B tests between different
    BERT models and comparing their performance.
    """

    @staticmethod
    def run_ab_test(
        texts: List[str],
        model_a_type: BERTModelType,
        model_b_type: BERTModelType,
        device: str = "auto",
    ) -> Dict[str, BenchmarkResult]:
        """
        Run A/B test between two models

        Args:
            texts: Test texts
            model_a_type: First model type
            model_b_type: Second model type
            device: Device to use

        Returns:
            Dict with benchmark results for both models
        """
        logger.info(f"Running A/B test: {model_a_type.value} vs {model_b_type.value}")

        # Benchmark model A
        result_a = ABTestFramework.benchmark_model(
            texts=texts,
            model_type=model_a_type,
            device=device,
        )

        # Benchmark model B
        result_b = ABTestFramework.benchmark_model(
            texts=texts,
            model_type=model_b_type,
            device=device,
        )

        return {
            "model_a": result_a,
            "model_b": result_b,
        }

    @staticmethod
    def benchmark_model(
        texts: List[str],
        model_type: BERTModelType,
        device: str = "auto",
        warmup_runs: int = 3,
    ) -> BenchmarkResult:
        """
        Benchmark a single model

        Args:
            texts: Test texts
            model_type: Model type to benchmark
            device: Device to use
            warmup_runs: Number of warmup runs

        Returns:
            BenchmarkResult with performance metrics
        """
        logger.info(f"Benchmarking model: {model_type.value}")

        # Get model
        model = BERTModelFactory.get_model(model_type=model_type, device=device)

        # Warmup
        for _ in range(warmup_runs):
            model.encode(texts[0] if texts else "Warmup text")

        # Get initial memory
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated(0) / (1024**2)  # MB
        else:
            initial_memory = 0

        # Benchmark inference
        latencies = []
        start_time = time.time()

        for text in texts:
            if not text or not text.strip():
                continue

            inference_start = time.time()
            model.encode(text)
            inference_end = time.time()

            latencies.append((inference_end - inference_start) * 1000)  # ms

        end_time = time.time()

        # Get peak memory
        if device == "cuda" and torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated(0) / (1024**2)  # MB
            memory_mb = peak_memory - initial_memory
        else:
            memory_mb = 0

        # Calculate metrics
        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            max_latency = max(latencies)
            min_latency = min(latencies)
        else:
            avg_latency = max_latency = min_latency = 0.0

        total_time = end_time - start_time
        throughput = len(texts) / total_time if total_time > 0 else 0

        result = BenchmarkResult(
            model_type=model_type.value,
            model_name=model.model_name,
            avg_latency_ms=avg_latency,
            max_latency_ms=max_latency,
            min_latency_ms=min_latency,
            memory_mb=memory_mb,
            throughput=throughput,
        )

        logger.info(
            f"Benchmark complete: avg={avg_latency:.2f}ms, "
            f"throughput={throughput:.2f} texts/sec, "
            f"memory={memory_mb:.2f}MB"
        )

        return result

    @staticmethod
    def save_benchmark_results(
        results: Dict[str, BenchmarkResult],
        output_path: Union[str, Path],
    ):
        """
        Save benchmark results to JSON file

        Args:
            results: Benchmark results
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict
        results_dict = {
            key: {
                "model_type": r.model_type,
                "model_name": r.model_name,
                "accuracy": r.accuracy,
                "avg_latency_ms": r.avg_latency_ms,
                "max_latency_ms": r.max_latency_ms,
                "min_latency_ms": r.min_latency_ms,
                "memory_mb": r.memory_mb,
                "throughput": r.throughput,
                "timestamp": r.timestamp,
            }
            for key, r in results.items()
        }

        # Add metadata
        results_dict["_metadata"] = {
            "timestamp": datetime.now().isoformat(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)

        logger.info(f"Benchmark results saved to {output_path}")

    @staticmethod
    def load_benchmark_results(input_path: Union[str, Path]) -> Dict[str, BenchmarkResult]:
        """
        Load benchmark results from JSON file

        Args:
            input_path: Input file path

        Returns:
            Benchmark results
        """
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Remove metadata
        _metadata = data.pop("_metadata", {})

        # Convert to BenchmarkResult objects
        results = {}
        for key, value in data.items():
            results[key] = BenchmarkResult(
                model_type=value["model_type"],
                model_name=value["model_name"],
                accuracy=value.get("accuracy"),
                avg_latency_ms=value["avg_latency_ms"],
                max_latency_ms=value["max_latency_ms"],
                min_latency_ms=value["min_latency_ms"],
                memory_mb=value["memory_mb"],
                throughput=value["throughput"],
                timestamp=value["timestamp"],
            )

        return results


# Convenience functions
def get_bert_model(
    model_type: Optional[str] = None,
    model_name: Optional[str] = None,
    device: str = "auto",
    max_length: int = 128,
) -> Any:
    """
    Convenience function for getting BERT model

    Args:
        model_type: Model type (kobert, klue_bert, klue_roberta)
        model_name: Override model name
        device: Device type
        max_length: Maximum sequence length

    Returns:
        Model instance
    """
    return BERTModelFactory.get_model(
        model_type=model_type,
        model_name=model_name,
        device=device,
        max_length=max_length,
    )


def run_model_benchmark(
    texts: List[str],
    model_types: Optional[List[str]] = None,
    device: str = "auto",
    output_path: Optional[Union[str, Path]] = None,
) -> Dict[str, BenchmarkResult]:
    """
    Benchmark multiple models

    Args:
        texts: Test texts
        model_types: List of model types to benchmark
        device: Device to use
        output_path: Optional output path for results

    Returns:
        Dict of benchmark results
    """
    if model_types is None:
        model_types = ["kobert", "klue_bert", "klue_roberta"]

    results = {}
    for model_type_str in model_types:
        model_type = BERTModelType(model_type_str)
        results[model_type_str] = ABTestFramework.benchmark_model(
            texts=texts,
            model_type=model_type,
            device=device,
        )

    # Save results if path provided
    if output_path:
        ABTestFramework.save_benchmark_results(results, output_path)

    return results
