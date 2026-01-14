"""
Asset Tracking Package for Forensic Pipeline
SPEC-ASSET-001 through SPEC-ASSET-004

Provides comprehensive asset management with UUID v7 tracking:
- UUID v7 Manager: Time-ordered unique identifiers
- Staged Batch Processor: 10 → 50 → 183 file processing
- Core Asset Manager: Storage tier management and lifecycle
- Document Utilization: Report generation and usage tracking
"""

from .core_asset_manager import (
    DEFAULT_STORAGE_POLICIES,
    CoreAssetManager,
    StoragePolicy,
    StorageTier,
    get_core_asset_manager,
)
from .document_utilization import (
    DOCUMENT_TEMPLATES,
    DocumentDistributionConfig,
    DocumentDistributionManager,
    DocumentFormat,
    DocumentPurpose,
    DocumentTemplate,
    DocumentUsageEvent,
    DocumentUtilizationManager,
)
from .staged_batch_processor import (
    BatchCheckpoint,
    BatchStageConfig,
    StagePipeline,
    StageResult,
    StagedBatchConfig,
    StagedBatchProcessor,
    ProcessingStage,
)
from .uuidv7_manager import (
    AssetMetadata,
    AssetRegistry,
    AssetStatus,
    AssetType,
    UUIDv7Generator,
    get_asset_registry,
)

__all__ = [
    # UUID v7 Manager
    "UUIDv7Generator",
    "AssetRegistry",
    "AssetMetadata",
    "AssetType",
    "AssetStatus",
    "get_asset_registry",
    # Staged Batch Processor
    "ProcessingStage",
    "StagedBatchProcessor",
    "StagedBatchConfig",
    "BatchStageConfig",
    "StageResult",
    "BatchCheckpoint",
    "StagePipeline",
    # Core Asset Manager
    "CoreAssetManager",
    "StorageTier",
    "StoragePolicy",
    "DEFAULT_STORAGE_POLICIES",
    "get_core_asset_manager",
    # Document Utilization
    "DocumentPurpose",
    "DocumentFormat",
    "DocumentTemplate",
    "DOCUMENT_TEMPLATES",
    "DocumentUtilizationManager",
    "DocumentUsageEvent",
    "DocumentDistributionConfig",
    "DocumentDistributionManager",
]
