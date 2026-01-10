"""
BatchConfigManager Unit Tests

SPEC-PERFOPT-001 Phase 2: Dynamic batch size configuration for forensic pipeline.
TDD RED Phase - Tests written FIRST before implementation.

Features:
    - StageBatchConfig dataclass with default/min/max batch sizes
    - get_optimal_batch_size(stage, available_memory_mb) method
    - Dynamic reduction when memory pressure detected
"""

import pytest
from dataclasses import FrozenInstanceError


class TestBatchConfigManagerImport:
    """Test that BatchConfigManager can be imported."""

    def test_import_batch_config_manager(self):
        """
        GIVEN: SPEC-PERFOPT-001 Phase 2 implementation
        WHEN: Importing BatchConfigManager
        THEN: The import should succeed without errors
        """
        from voice_man.config.batch_config import BatchConfigManager

        assert BatchConfigManager is not None

    def test_import_stage_batch_config(self):
        """
        GIVEN: SPEC-PERFOPT-001 Phase 2 implementation
        WHEN: Importing StageBatchConfig dataclass
        THEN: The import should succeed without errors
        """
        from voice_man.config.batch_config import StageBatchConfig

        assert StageBatchConfig is not None


class TestStageBatchConfig:
    """Test StageBatchConfig dataclass."""

    def test_stage_batch_config_creation(self):
        """
        GIVEN: StageBatchConfig dataclass
        WHEN: Created with values
        THEN: Should have default, min, and max batch size fields
        """
        from voice_man.config.batch_config import StageBatchConfig

        config = StageBatchConfig(
            stage="stt",
            default_batch_size=16,
            min_batch_size=1,
            max_batch_size=32,
        )

        assert config.stage == "stt"
        assert config.default_batch_size == 16
        assert config.min_batch_size == 1
        assert config.max_batch_size == 32

    def test_stage_batch_config_memory_per_sample(self):
        """
        GIVEN: StageBatchConfig dataclass
        WHEN: Created with memory_per_sample_mb
        THEN: Should store memory requirement per sample
        """
        from voice_man.config.batch_config import StageBatchConfig

        config = StageBatchConfig(
            stage="ser",
            default_batch_size=8,
            min_batch_size=1,
            max_batch_size=16,
            memory_per_sample_mb=1280,  # ~1.25GB per sample
        )

        assert config.memory_per_sample_mb == 1280

    def test_stage_batch_config_default_memory(self):
        """
        GIVEN: StageBatchConfig without memory_per_sample_mb
        WHEN: Created
        THEN: Should have default memory_per_sample_mb of 0
        """
        from voice_man.config.batch_config import StageBatchConfig

        config = StageBatchConfig(
            stage="scoring",
            default_batch_size=32,
            min_batch_size=1,
            max_batch_size=64,
        )

        assert config.memory_per_sample_mb == 0


class TestBatchConfigManagerInitialization:
    """Test BatchConfigManager initialization."""

    def test_initialization_with_defaults(self):
        """
        GIVEN: BatchConfigManager class
        WHEN: Initialized without parameters
        THEN: Should have default configurations for all stages
        """
        from voice_man.config.batch_config import BatchConfigManager

        manager = BatchConfigManager()

        # Should have configs for all forensic stages
        assert manager.has_config("stt")
        assert manager.has_config("alignment")
        assert manager.has_config("diarization")
        assert manager.has_config("ser")
        assert manager.has_config("scoring")

    def test_get_stage_config(self):
        """
        GIVEN: BatchConfigManager instance
        WHEN: get_config() is called for a stage
        THEN: Should return StageBatchConfig for that stage
        """
        from voice_man.config.batch_config import BatchConfigManager, StageBatchConfig

        manager = BatchConfigManager()

        config = manager.get_config("stt")

        assert isinstance(config, StageBatchConfig)
        assert config.stage == "stt"

    def test_get_unknown_stage_raises_error(self):
        """
        GIVEN: BatchConfigManager instance
        WHEN: get_config() is called for unknown stage
        THEN: Should raise KeyError
        """
        from voice_man.config.batch_config import BatchConfigManager

        manager = BatchConfigManager()

        with pytest.raises(KeyError):
            manager.get_config("unknown_stage")


class TestOptimalBatchSize:
    """Test get_optimal_batch_size method."""

    def test_optimal_batch_size_with_plenty_memory(self):
        """
        GIVEN: BatchConfigManager with STT config (1024MB per sample, max=32)
        WHEN: get_optimal_batch_size() called with 40000 MB available
        THEN: Should return max batch size (32) since 40000/1024=39 > max
        """
        from voice_man.config.batch_config import BatchConfigManager

        manager = BatchConfigManager()

        # 40GB should allow 39 samples, but max is 32
        batch_size = manager.get_optimal_batch_size("stt", available_memory_mb=40000)

        # With plenty of memory, should use max batch size
        config = manager.get_config("stt")
        assert batch_size == config.max_batch_size

    def test_optimal_batch_size_with_limited_memory(self):
        """
        GIVEN: BatchConfigManager instance
        WHEN: get_optimal_batch_size() called with limited memory
        THEN: Should return reduced batch size based on memory
        """
        from voice_man.config.batch_config import BatchConfigManager

        manager = BatchConfigManager()

        # With limited memory, should reduce batch size
        batch_size = manager.get_optimal_batch_size("ser", available_memory_mb=5000)

        config = manager.get_config("ser")
        assert batch_size <= config.default_batch_size
        assert batch_size >= config.min_batch_size

    def test_optimal_batch_size_minimum_enforcement(self):
        """
        GIVEN: BatchConfigManager instance
        WHEN: get_optimal_batch_size() called with very low memory
        THEN: Should return at least min_batch_size
        """
        from voice_man.config.batch_config import BatchConfigManager

        manager = BatchConfigManager()

        batch_size = manager.get_optimal_batch_size("stt", available_memory_mb=100)

        config = manager.get_config("stt")
        assert batch_size >= config.min_batch_size

    def test_optimal_batch_size_unknown_stage(self):
        """
        GIVEN: BatchConfigManager instance
        WHEN: get_optimal_batch_size() called for unknown stage
        THEN: Should raise KeyError
        """
        from voice_man.config.batch_config import BatchConfigManager

        manager = BatchConfigManager()

        with pytest.raises(KeyError):
            manager.get_optimal_batch_size("unknown", available_memory_mb=10000)

    def test_optimal_batch_size_zero_memory(self):
        """
        GIVEN: BatchConfigManager instance
        WHEN: get_optimal_batch_size() called with 0 memory
        THEN: Should return min_batch_size
        """
        from voice_man.config.batch_config import BatchConfigManager

        manager = BatchConfigManager()

        batch_size = manager.get_optimal_batch_size("stt", available_memory_mb=0)

        config = manager.get_config("stt")
        assert batch_size == config.min_batch_size


class TestDynamicReduction:
    """Test dynamic batch size reduction under memory pressure."""

    def test_reduce_batch_size(self):
        """
        GIVEN: BatchConfigManager instance
        WHEN: reduce_batch_size() is called with current size
        THEN: Should return reduced size (half)
        """
        from voice_man.config.batch_config import BatchConfigManager

        manager = BatchConfigManager()

        reduced = manager.reduce_batch_size("stt", current_batch_size=16)

        assert reduced == 8

    def test_reduce_batch_size_respects_minimum(self):
        """
        GIVEN: BatchConfigManager instance with batch at minimum
        WHEN: reduce_batch_size() is called
        THEN: Should return min_batch_size (cannot go lower)
        """
        from voice_man.config.batch_config import BatchConfigManager

        manager = BatchConfigManager()
        config = manager.get_config("stt")

        reduced = manager.reduce_batch_size("stt", current_batch_size=config.min_batch_size)

        assert reduced == config.min_batch_size

    def test_increase_batch_size(self):
        """
        GIVEN: BatchConfigManager instance
        WHEN: increase_batch_size() is called
        THEN: Should return increased size (double, up to max)
        """
        from voice_man.config.batch_config import BatchConfigManager

        manager = BatchConfigManager()

        increased = manager.increase_batch_size("stt", current_batch_size=8)

        assert increased == 16

    def test_increase_batch_size_respects_maximum(self):
        """
        GIVEN: BatchConfigManager instance with batch near maximum
        WHEN: increase_batch_size() is called
        THEN: Should not exceed max_batch_size
        """
        from voice_man.config.batch_config import BatchConfigManager

        manager = BatchConfigManager()
        config = manager.get_config("stt")

        increased = manager.increase_batch_size("stt", current_batch_size=config.max_batch_size)

        assert increased == config.max_batch_size


class TestCustomConfiguration:
    """Test custom stage configuration."""

    def test_add_custom_stage_config(self):
        """
        GIVEN: BatchConfigManager instance
        WHEN: add_config() is called with custom StageBatchConfig
        THEN: Should register new stage configuration
        """
        from voice_man.config.batch_config import BatchConfigManager, StageBatchConfig

        manager = BatchConfigManager()

        custom_config = StageBatchConfig(
            stage="custom_stage",
            default_batch_size=10,
            min_batch_size=2,
            max_batch_size=20,
            memory_per_sample_mb=500,
        )

        manager.add_config(custom_config)

        assert manager.has_config("custom_stage")
        assert manager.get_config("custom_stage").default_batch_size == 10

    def test_update_existing_config(self):
        """
        GIVEN: BatchConfigManager with existing stt config
        WHEN: add_config() is called with new stt config
        THEN: Should update the existing configuration
        """
        from voice_man.config.batch_config import BatchConfigManager, StageBatchConfig

        manager = BatchConfigManager()

        new_config = StageBatchConfig(
            stage="stt",
            default_batch_size=32,
            min_batch_size=4,
            max_batch_size=64,
        )

        manager.add_config(new_config)

        assert manager.get_config("stt").default_batch_size == 32


class TestMemoryCalculation:
    """Test memory-based batch size calculation."""

    def test_calculate_max_batch_for_memory(self):
        """
        GIVEN: BatchConfigManager with known memory_per_sample_mb
        WHEN: calculate_max_batch() is called with available memory
        THEN: Should return max batch that fits in memory
        """
        from voice_man.config.batch_config import BatchConfigManager, StageBatchConfig

        manager = BatchConfigManager()

        # Add config with known memory per sample
        config = StageBatchConfig(
            stage="test_stage",
            default_batch_size=8,
            min_batch_size=1,
            max_batch_size=32,
            memory_per_sample_mb=1000,  # 1GB per sample
        )
        manager.add_config(config)

        # With 8GB available, should fit 8 samples
        max_batch = manager.calculate_max_batch("test_stage", available_memory_mb=8000)

        assert max_batch == 8

    def test_calculate_max_batch_respects_max_config(self):
        """
        GIVEN: BatchConfigManager with limited max_batch_size
        WHEN: calculate_max_batch() called with lots of memory
        THEN: Should not exceed max_batch_size from config
        """
        from voice_man.config.batch_config import BatchConfigManager, StageBatchConfig

        manager = BatchConfigManager()

        config = StageBatchConfig(
            stage="test_stage",
            default_batch_size=8,
            min_batch_size=1,
            max_batch_size=16,
            memory_per_sample_mb=1000,
        )
        manager.add_config(config)

        # With 100GB available, still should not exceed max_batch_size
        max_batch = manager.calculate_max_batch("test_stage", available_memory_mb=100000)

        assert max_batch == 16


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
