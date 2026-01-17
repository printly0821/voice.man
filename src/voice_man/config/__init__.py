"""
Voice Man configuration module.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from voice_man.config.whisperx_config import (
    WhisperXConfig,
    get_hf_token,
    HFTokenNotFoundError,
)

__all__ = [
    "WhisperXConfig",
    "get_hf_token",
    "HFTokenNotFoundError",
    "get_bert_config",
    "BERTConfig",
]


class BERTConfig:
    """
    BERT configuration manager

    Loads and manages BERT model configuration from YAML file
    with environment variable override support.
    """

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize BERT configuration

        Args:
            config_path: Path to bert_config.yaml (optional)
        """
        if config_path is None:
            # Default to src/voice_man/config/bert_config.yaml
            project_root = Path(__file__).parent.parent.parent.parent
            config_path = project_root / "src" / "voice_man" / "config" / "bert_config.yaml"

        self.config_path = config_path
        self._config: Optional[Dict[str, Any]] = None

    def load(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if self._config is not None:
            return self._config

        if not self.config_path.exists():
            # Return default configuration
            self._config = self._get_default_config()
            return self._config

        with open(self.config_path, "r", encoding="utf-8") as f:
            self._config = yaml.safe_load(f)

        return self._config

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "model": {
                "type": "kobert",
                "models": {
                    "kobert": "skt/kobert-base-v1",
                    "klue_bert": "klue/bert-base",
                    "klue_roberta": "klue/roberta-base",
                },
                "device": "auto",
                "max_length": 128,
            },
            "inference": {
                "batch_size": 8,
                "confidence_threshold": 0.7,
                "timeout_ms": 100,
            },
            "performance": {
                "max_gpu_memory_mb": 4096,
                "enable_cache": True,
                "cache_ttl_minutes": 30,
            },
            "ab_testing": {"enabled": False, "models": []},
            "benchmarking": {
                "warmup_runs": 3,
                "benchmark_runs": 10,
                "batch_sizes": [1, 4, 8, 16],
                "output_dir": "ref/benchmark_results",
            },
            "logging": {
                "log_model_loading": True,
                "log_inference_timing": True,
                "log_memory_usage": True,
            },
        }

    def get_model_type(self) -> str:
        """
        Get model type with environment variable override

        Priority:
            1. VOICE_MAN_BERT_MODEL environment variable
            2. Configuration file
            3. Default (kobert)

        Returns:
            Model type string
        """
        # Check environment variable first
        env_model = os.getenv("VOICE_MAN_BERT_MODEL")
        if env_model:
            return env_model.lower()

        # Fall back to config
        config = self.load()
        return config.get("model", {}).get("type", "kobert")

    def get_model_name(self, model_type: Optional[str] = None) -> str:
        """
        Get model name for given type

        Args:
            model_type: Model type (uses default if not specified)

        Returns:
            Model name string
        """
        if model_type is None:
            model_type = self.get_model_type()

        config = self.load()
        models = config.get("model", {}).get("models", {})

        return models.get(model_type, models.get("kobert", "skt/kobert-base-v1"))

    def get_device(self) -> str:
        """Get device setting"""
        config = self.load()
        return config.get("model", {}).get("device", "auto")

    def get_max_length(self) -> int:
        """Get maximum sequence length"""
        config = self.load()
        return config.get("model", {}).get("max_length", 128)

    def get_batch_size(self) -> int:
        """Get batch size"""
        config = self.load()
        return config.get("inference", {}).get("batch_size", 8)

    def get_confidence_threshold(self) -> float:
        """Get confidence threshold"""
        config = self.load()
        return config.get("inference", {}).get("confidence_threshold", 0.7)

    def is_cache_enabled(self) -> bool:
        """Check if caching is enabled"""
        config = self.load()
        return config.get("performance", {}).get("enable_cache", True)

    def get_benchmark_config(self) -> Dict[str, Any]:
        """Get benchmarking configuration"""
        config = self.load()
        return config.get("benchmarking", {})

    def is_ab_testing_enabled(self) -> bool:
        """Check if A/B testing is enabled"""
        config = self.load()
        return config.get("ab_testing", {}).get("enabled", False)

    def get_ab_test_models(self) -> list:
        """Get A/B test model configuration"""
        config = self.load()
        return config.get("ab_testing", {}).get("models", [])


# Global configuration instance
_bert_config: Optional[BERTConfig] = None


def get_bert_config(config_path: Optional[Path] = None) -> BERTConfig:
    """
    Get global BERT configuration instance

    Args:
        config_path: Optional path to configuration file

    Returns:
        BERTConfig instance
    """
    global _bert_config

    if _bert_config is None:
        _bert_config = BERTConfig(config_path)

    return _bert_config
