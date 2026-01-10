"""
BatchConfigManager - Dynamic batch size configuration for forensic pipeline.

SPEC-PERFOPT-001 Phase 2: Batch size management with memory-aware optimization.

Features:
    - StageBatchConfig dataclass with default/min/max batch sizes
    - get_optimal_batch_size(stage, available_memory_mb) method
    - Dynamic reduction when memory pressure detected
    - Memory-based batch calculation
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class StageBatchConfig:
    """
    Configuration for batch size per processing stage.

    Attributes:
        stage: Stage identifier (stt, alignment, diarization, ser, scoring)
        default_batch_size: Default batch size for normal operation
        min_batch_size: Minimum batch size (floor for reduction)
        max_batch_size: Maximum batch size (ceiling for scaling)
        memory_per_sample_mb: Memory required per sample in MB (0 if unknown)
    """

    stage: str
    default_batch_size: int
    min_batch_size: int
    max_batch_size: int
    memory_per_sample_mb: int = 0


# Default configurations for forensic pipeline stages
DEFAULT_STAGE_CONFIGS: Dict[str, StageBatchConfig] = {
    "stt": StageBatchConfig(
        stage="stt",
        default_batch_size=16,
        min_batch_size=1,
        max_batch_size=32,
        memory_per_sample_mb=1024,  # ~1GB per sample for STT
    ),
    "alignment": StageBatchConfig(
        stage="alignment",
        default_batch_size=32,
        min_batch_size=1,
        max_batch_size=64,
        memory_per_sample_mb=128,  # ~128MB per sample
    ),
    "diarization": StageBatchConfig(
        stage="diarization",
        default_batch_size=8,
        min_batch_size=1,
        max_batch_size=16,
        memory_per_sample_mb=1024,  # ~1GB per sample
    ),
    "ser": StageBatchConfig(
        stage="ser",
        default_batch_size=8,
        min_batch_size=1,
        max_batch_size=16,
        memory_per_sample_mb=1280,  # ~1.25GB per sample for SER
    ),
    "scoring": StageBatchConfig(
        stage="scoring",
        default_batch_size=64,
        min_batch_size=1,
        max_batch_size=128,
        memory_per_sample_mb=32,  # ~32MB per sample
    ),
}


class BatchConfigManager:
    """
    Manager for batch size configuration across forensic pipeline stages.

    Provides dynamic batch size optimization based on available memory
    and stage-specific constraints.

    Example:
        manager = BatchConfigManager()
        batch_size = manager.get_optimal_batch_size("stt", available_memory_mb=16000)
        # Use batch_size for processing

        # If memory pressure detected:
        reduced = manager.reduce_batch_size("stt", current_batch_size=16)
    """

    def __init__(self, configs: Optional[Dict[str, StageBatchConfig]] = None):
        """
        Initialize BatchConfigManager.

        Args:
            configs: Optional custom configurations. Uses defaults if None.
        """
        self._configs: Dict[str, StageBatchConfig] = {}

        # Load default configs
        for stage, config in DEFAULT_STAGE_CONFIGS.items():
            self._configs[stage] = StageBatchConfig(
                stage=config.stage,
                default_batch_size=config.default_batch_size,
                min_batch_size=config.min_batch_size,
                max_batch_size=config.max_batch_size,
                memory_per_sample_mb=config.memory_per_sample_mb,
            )

        # Override with custom configs if provided
        if configs:
            for stage, config in configs.items():
                self._configs[stage] = config

        logger.debug(f"BatchConfigManager initialized with {len(self._configs)} stage configs")

    def has_config(self, stage: str) -> bool:
        """
        Check if configuration exists for a stage.

        Args:
            stage: Stage identifier.

        Returns:
            True if configuration exists, False otherwise.
        """
        return stage in self._configs

    def get_config(self, stage: str) -> StageBatchConfig:
        """
        Get configuration for a stage.

        Args:
            stage: Stage identifier.

        Returns:
            StageBatchConfig for the stage.

        Raises:
            KeyError: If stage configuration not found.
        """
        if stage not in self._configs:
            raise KeyError(f"No configuration for stage: {stage}")
        return self._configs[stage]

    def add_config(self, config: StageBatchConfig) -> None:
        """
        Add or update a stage configuration.

        Args:
            config: StageBatchConfig to add/update.
        """
        self._configs[config.stage] = config
        logger.debug(f"Added/updated config for stage: {config.stage}")

    def get_optimal_batch_size(self, stage: str, available_memory_mb: int) -> int:
        """
        Calculate optimal batch size based on available memory.

        Args:
            stage: Stage identifier.
            available_memory_mb: Available GPU/system memory in MB.

        Returns:
            Optimal batch size within min/max bounds.

        Raises:
            KeyError: If stage configuration not found.
        """
        config = self.get_config(stage)

        # If no memory available, return minimum
        if available_memory_mb <= 0:
            return config.min_batch_size

        # Calculate based on memory per sample
        if config.memory_per_sample_mb > 0:
            max_possible = available_memory_mb // config.memory_per_sample_mb
            optimal = min(max_possible, config.max_batch_size)
            optimal = max(optimal, config.min_batch_size)
        else:
            # If memory per sample unknown, use defaults based on memory ratio
            # Assume default config is for ~16GB available
            ratio = available_memory_mb / 16384
            if ratio >= 1.0:
                optimal = config.max_batch_size
            elif ratio >= 0.5:
                optimal = config.default_batch_size
            else:
                optimal = max(int(config.default_batch_size * ratio), config.min_batch_size)

        logger.debug(
            f"Optimal batch size for {stage}: {optimal} "
            f"(available: {available_memory_mb}MB, per_sample: {config.memory_per_sample_mb}MB)"
        )

        return optimal

    def reduce_batch_size(self, stage: str, current_batch_size: int) -> int:
        """
        Reduce batch size by half (for memory pressure situations).

        Args:
            stage: Stage identifier.
            current_batch_size: Current batch size to reduce.

        Returns:
            Reduced batch size, respecting min_batch_size.

        Raises:
            KeyError: If stage configuration not found.
        """
        config = self.get_config(stage)

        reduced = current_batch_size // 2
        reduced = max(reduced, config.min_batch_size)

        logger.info(f"Reduced batch size for {stage}: {current_batch_size} -> {reduced}")
        return reduced

    def increase_batch_size(self, stage: str, current_batch_size: int) -> int:
        """
        Increase batch size by doubling (when memory allows).

        Args:
            stage: Stage identifier.
            current_batch_size: Current batch size to increase.

        Returns:
            Increased batch size, respecting max_batch_size.

        Raises:
            KeyError: If stage configuration not found.
        """
        config = self.get_config(stage)

        increased = current_batch_size * 2
        increased = min(increased, config.max_batch_size)

        logger.debug(f"Increased batch size for {stage}: {current_batch_size} -> {increased}")
        return increased

    def calculate_max_batch(self, stage: str, available_memory_mb: int) -> int:
        """
        Calculate maximum batch size that fits in available memory.

        Args:
            stage: Stage identifier.
            available_memory_mb: Available memory in MB.

        Returns:
            Maximum batch size that fits, respecting config limits.

        Raises:
            KeyError: If stage configuration not found.
        """
        config = self.get_config(stage)

        if config.memory_per_sample_mb <= 0:
            return config.max_batch_size

        max_from_memory = available_memory_mb // config.memory_per_sample_mb
        return min(max_from_memory, config.max_batch_size)

    def get_all_configs(self) -> Dict[str, StageBatchConfig]:
        """
        Get all stage configurations.

        Returns:
            Dictionary of all stage configurations.
        """
        return self._configs.copy()
