"""
BERT Model Benchmarking Utilities
Comprehensive benchmarking framework for comparing KoBERT and KLUE-BERT models

Provides:
    - Model performance comparison
    - Accuracy, latency, memory metrics
    - A/B testing framework
    - Result storage and analysis
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs"""

    warmup_runs: int = 3
    benchmark_runs: int = 10
    batch_sizes: List[int] = field(default_factory=lambda: [1, 4, 8, 16])
    text_lengths: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    device: str = "auto"


@dataclass
class ModelMetrics:
    """Performance metrics for a single model"""

    model_type: str
    model_name: str
    device: str

    # Latency metrics (ms)
    avg_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0

    # Throughput metrics
    throughput_texts_per_sec: float = 0.0

    # Memory metrics (MB)
    memory_allocated_mb: float = 0.0
    memory_reserved_mb: float = 0.0

    # Model info
    model_size_mb: Optional[float] = None
    hidden_size: Optional[int] = None

    # Timing metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ComparisonResult:
    """Result of comparing two models"""

    model_a_metrics: ModelMetrics
    model_b_metrics: ModelMetrics

    # Relative performance
    latency_improvement_percent: float = 0.0
    throughput_improvement_percent: float = 0.0
    memory_efficiency_percent: float = 0.0

    # Recommendation
    recommended_model: str = ""
    recommendation_reason: str = ""

    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class BERTBenchmark:
    """
    BERT model benchmarking framework

    Provides comprehensive benchmarking capabilities for comparing
    different BERT models (KoBERT, KLUE-BERT, KLUE-RoBERTa).
    """

    # Sample texts for benchmarking
    SAMPLE_TEXTS = [
        "안녕하세요, 오늘 날씨가 정말 좋네요.",
        "이번 회의에서 중요한 안건을 논의했습니다.",
        "감기 때문에 몸이 좀 불편합니다.",
        "새로운 프로젝트를 시작하게 되어 기대됩니다.",
        "어제 친구와 함께 맛있는 저녁을 먹었습니다.",
        "업무 관련하여 중요한 이메일을 보냈습니다.",
        "주말에 가족과 함께 시간을 보냈습니다.",
        "새로운 기술을 배우는 것은 언제나 흥미롭습니다.",
        "건강을 위해 매일 운동을 하고 있습니다.",
        "최근 읽은 책이 많은 영감을 주었습니다.",
    ]

    @staticmethod
    def benchmark_model(
        model: Any,
        config: Optional[BenchmarkConfig] = None,
        test_texts: Optional[List[str]] = None,
    ) -> ModelMetrics:
        """
        Benchmark a single model

        Args:
            model: BERT model instance
            config: Benchmark configuration
            test_texts: Optional custom test texts

        Returns:
            ModelMetrics with performance data
        """
        if config is None:
            config = BenchmarkConfig()

        if test_texts is None:
            test_texts = BERTBenchmark.SAMPLE_TEXTS

        logger.info(f"Benchmarking model: {model.model_name}")

        # Warmup
        logger.info(f"Running {config.warmup_runs} warmup runs...")
        for i in range(config.warmup_runs):
            text = test_texts[i % len(test_texts)]
            model.encode(text)

        # Get initial memory
        if model.device == torch.device("cuda"):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated(0) / (1024**2)  # MB
        else:
            initial_memory = 0

        # Benchmark runs
        logger.info(f"Running {config.benchmark_runs} benchmark runs...")
        latencies = []

        start_time = time.time()

        for i in range(config.benchmark_runs):
            text = test_texts[i % len(test_texts)]

            inference_start = time.time()
            model.encode(text)
            inference_end = time.time()

            latencies.append((inference_end - inference_start) * 1000)  # ms

        end_time = time.time()

        # Get peak memory
        if model.device == torch.device("cuda"):
            peak_memory = torch.cuda.max_memory_allocated(0) / (1024**2)  # MB
            memory_allocated = peak_memory
            memory_reserved = torch.cuda.memory_reserved(0) / (1024**2)
        else:
            memory_allocated = 0
            memory_reserved = 0

        # Calculate statistics
        latencies_sorted = sorted(latencies)
        n = len(latencies_sorted)

        avg_latency = sum(latencies) / n
        min_latency = latencies_sorted[0]
        max_latency = latencies_sorted[-1]
        p50_latency = latencies_sorted[int(n * 0.5)]
        p95_latency = latencies_sorted[int(n * 0.95)]
        p99_latency = latencies_sorted[int(n * 0.99)]

        total_time = end_time - start_time
        throughput = config.benchmark_runs / total_time if total_time > 0 else 0

        # Get model info
        model_info = model.get_model_info()

        metrics = ModelMetrics(
            model_type=model_info.get("model_type", "unknown"),
            model_name=model.model_name,
            device=model.device.value,
            avg_latency_ms=avg_latency,
            min_latency_ms=min_latency,
            max_latency_ms=max_latency,
            p50_latency_ms=p50_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            throughput_texts_per_sec=throughput,
            memory_allocated_mb=memory_allocated - initial_memory,
            memory_reserved_mb=memory_reserved,
            hidden_size=model_info.get("hidden_size"),
        )

        logger.info(
            f"Benchmark complete: avg={avg_latency:.2f}ms, "
            f"p95={p95_latency:.2f}ms, "
            f"throughput={throughput:.2f} texts/sec"
        )

        return metrics

    @staticmethod
    def compare_models(
        model_a: Any,
        model_b: Any,
        config: Optional[BenchmarkConfig] = None,
        test_texts: Optional[List[str]] = None,
    ) -> ComparisonResult:
        """
        Compare two models

        Args:
            model_a: First model
            model_b: Second model
            config: Benchmark configuration
            test_texts: Optional custom test texts

        Returns:
            ComparisonResult with comparison data
        """
        logger.info(f"Comparing models: {model_a.model_name} vs {model_b.model_name}")

        # Benchmark both models
        metrics_a = BERTBenchmark.benchmark_model(model_a, config, test_texts)
        metrics_b = BERTBenchmark.benchmark_model(model_b, config, test_texts)

        # Calculate improvements
        # Lower latency is better
        if metrics_a.avg_latency_ms > 0:
            latency_improvement = (
                (metrics_a.avg_latency_ms - metrics_b.avg_latency_ms) / metrics_a.avg_latency_ms
            ) * 100
        else:
            latency_improvement = 0.0

        # Higher throughput is better
        if metrics_a.throughput_texts_per_sec > 0:
            throughput_improvement = (
                (metrics_b.throughput_texts_per_sec - metrics_a.throughput_texts_per_sec)
                / metrics_a.throughput_texts_per_sec
            ) * 100
        else:
            throughput_improvement = 0.0

        # Lower memory is better
        if metrics_a.memory_allocated_mb > 0:
            memory_efficiency = (
                (metrics_a.memory_allocated_mb - metrics_b.memory_allocated_mb)
                / metrics_a.memory_allocated_mb
            ) * 100
        else:
            memory_efficiency = 0.0

        # Determine recommendation
        # Score: latency (40%) + throughput (40%) + memory (20%)
        score_a = (
            (1000 / metrics_a.avg_latency_ms if metrics_a.avg_latency_ms > 0 else 0) * 0.4
            + metrics_a.throughput_texts_per_sec * 0.4
            + (1000 / metrics_a.memory_allocated_mb if metrics_a.memory_allocated_mb > 0 else 0)
            * 0.2
        )

        score_b = (
            (1000 / metrics_b.avg_latency_ms if metrics_b.avg_latency_ms > 0 else 0) * 0.4
            + metrics_b.throughput_texts_per_sec * 0.4
            + (1000 / metrics_b.memory_allocated_mb if metrics_b.memory_allocated_mb > 0 else 0)
            * 0.2
        )

        if score_b > score_a:
            recommended = metrics_b.model_name
            reason = (
                f"Model B has better overall performance "
                f"({score_b:.2f} vs {score_a:.2f} composite score)"
            )
        else:
            recommended = metrics_a.model_name
            reason = (
                f"Model A has better overall performance "
                f"({score_a:.2f} vs {score_b:.2f} composite score)"
            )

        result = ComparisonResult(
            model_a_metrics=metrics_a,
            model_b_metrics=metrics_b,
            latency_improvement_percent=latency_improvement,
            throughput_improvement_percent=throughput_improvement,
            memory_efficiency_percent=memory_efficiency,
            recommended_model=recommended,
            recommendation_reason=reason,
        )

        logger.info(f"Comparison complete: {recommended} recommended")
        logger.info(f"Latency: {latency_improvement:+.2f}%")
        logger.info(f"Throughput: {throughput_improvement:+.2f}%")
        logger.info(f"Memory: {memory_efficiency:+.2f}%")

        return result

    @staticmethod
    def save_metrics(metrics: ModelMetrics, output_path: Path) -> None:
        """
        Save metrics to JSON file

        Args:
            metrics: Metrics to save
            output_path: Output file path
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "model_type": metrics.model_type,
            "model_name": metrics.model_name,
            "device": metrics.device,
            "avg_latency_ms": metrics.avg_latency_ms,
            "min_latency_ms": metrics.min_latency_ms,
            "max_latency_ms": metrics.max_latency_ms,
            "p50_latency_ms": metrics.p50_latency_ms,
            "p95_latency_ms": metrics.p95_latency_ms,
            "p99_latency_ms": metrics.p99_latency_ms,
            "throughput_texts_per_sec": metrics.throughput_texts_per_sec,
            "memory_allocated_mb": metrics.memory_allocated_mb,
            "memory_reserved_mb": metrics.memory_reserved_mb,
            "model_size_mb": metrics.model_size_mb,
            "hidden_size": metrics.hidden_size,
            "timestamp": metrics.timestamp,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Metrics saved to {output_path}")

    @staticmethod
    def save_comparison(result: ComparisonResult, output_path: Path) -> None:
        """
        Save comparison result to JSON file

        Args:
            result: Comparison result to save
            output_path: Output file path
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        def metrics_to_dict(metrics: ModelMetrics) -> Dict:
            return {
                "model_type": metrics.model_type,
                "model_name": metrics.model_name,
                "device": metrics.device,
                "avg_latency_ms": metrics.avg_latency_ms,
                "min_latency_ms": metrics.min_latency_ms,
                "max_latency_ms": metrics.max_latency_ms,
                "p50_latency_ms": metrics.p50_latency_ms,
                "p95_latency_ms": metrics.p95_latency_ms,
                "p99_latency_ms": metrics.p99_latency_ms,
                "throughput_texts_per_sec": metrics.throughput_texts_per_sec,
                "memory_allocated_mb": metrics.memory_allocated_mb,
                "memory_reserved_mb": metrics.memory_reserved_mb,
                "model_size_mb": metrics.model_size_mb,
                "hidden_size": metrics.hidden_size,
            }

        data = {
            "model_a": metrics_to_dict(result.model_a_metrics),
            "model_b": metrics_to_dict(result.model_b_metrics),
            "latency_improvement_percent": result.latency_improvement_percent,
            "throughput_improvement_percent": result.throughput_improvement_percent,
            "memory_efficiency_percent": result.memory_efficiency_percent,
            "recommended_model": result.recommended_model,
            "recommendation_reason": result.recommendation_reason,
            "timestamp": result.timestamp,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Comparison result saved to {output_path}")


def run_benchmark_comparison(
    model_types: List[str],
    output_dir: Optional[Path] = None,
    device: str = "auto",
) -> Dict[str, ComparisonResult]:
    """
    Run benchmark comparison between multiple models

    Args:
        model_types: List of model types to compare
        output_dir: Optional output directory for results
        device: Device to use for benchmarking

    Returns:
        Dict of comparison results
    """
    from voice_man.services.nlp.unified_bert import BERTModelFactory

    if output_dir is None:
        output_dir = Path("ref/benchmark_results")

    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    models = {}

    # Load all models
    for model_type in model_types:
        logger.info(f"Loading model: {model_type}")
        try:
            models[model_type] = BERTModelFactory.get_model(model_type=model_type, device=device)
        except Exception as e:
            logger.error(f"Failed to load model {model_type}: {e}")
            continue

    # Compare all pairs
    model_type_list = list(models.keys())
    for i in range(len(model_type_list)):
        for j in range(i + 1, len(model_type_list)):
            model_type_a = model_type_list[i]
            model_type_b = model_type_list[j]

            logger.info(f"Comparing {model_type_a} vs {model_type_b}")

            comparison = BERTBenchmark.compare_models(models[model_type_a], models[model_type_b])

            # Save results
            result_key = f"{model_type_a}_vs_{model_type_b}"
            results[result_key] = comparison

            output_path = output_dir / f"{result_key}.json"
            BERTBenchmark.save_comparison(comparison, output_path)

            # Save individual metrics
            metrics_path_a = output_dir / f"{model_type_a}_metrics.json"
            metrics_path_b = output_dir / f"{model_type_b}_metrics.json"
            BERTBenchmark.save_metrics(comparison.model_a_metrics, metrics_path_a)
            BERTBenchmark.save_metrics(comparison.model_b_metrics, metrics_path_b)

    return results


def generate_benchmark_report(results: Dict[str, ComparisonResult], output_path: Path) -> None:
    """
    Generate a human-readable benchmark report

    Args:
        results: Benchmark comparison results
        output_path: Output file path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# BERT Model Benchmark Report")
    lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("---\n")

    for key, result in results.items():
        lines.append(f"## Comparison: {key}")
        lines.append("")

        lines.append("### Model A")
        lines.append(f"- **Model**: {result.model_a_metrics.model_name}")
        lines.append(f"- **Avg Latency**: {result.model_a_metrics.avg_latency_ms:.2f} ms")
        lines.append(f"- **P95 Latency**: {result.model_a_metrics.p95_latency_ms:.2f} ms")
        lines.append(
            f"- **Throughput**: {result.model_a_metrics.throughput_texts_per_sec:.2f} texts/sec"
        )
        lines.append(f"- **Memory**: {result.model_a_metrics.memory_allocated_mb:.2f} MB")
        lines.append("")

        lines.append("### Model B")
        lines.append(f"- **Model**: {result.model_b_metrics.model_name}")
        lines.append(f"- **Avg Latency**: {result.model_b_metrics.avg_latency_ms:.2f} ms")
        lines.append(f"- **P95 Latency**: {result.model_b_metrics.p95_latency_ms:.2f} ms")
        lines.append(
            f"- **Throughput**: {result.model_b_metrics.throughput_texts_per_sec:.2f} texts/sec"
        )
        lines.append(f"- **Memory**: {result.model_b_metrics.memory_allocated_mb:.2f} MB")
        lines.append("")

        lines.append("### Performance Comparison")
        lines.append(f"- **Latency**: {result.latency_improvement_percent:+.2f}%")
        lines.append(f"- **Throughput**: {result.throughput_improvement_percent:+.2f}%")
        lines.append(f"- **Memory**: {result.memory_efficiency_percent:+.2f}%")
        lines.append("")

        lines.append("### Recommendation")
        lines.append(f"- **Recommended Model**: {result.recommended_model}")
        lines.append(f"- **Reason**: {result.recommendation_reason}")
        lines.append("")
        lines.append("---")
        lines.append("")

    report = "\n".join(lines)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    logger.info(f"Benchmark report saved to {output_path}")
