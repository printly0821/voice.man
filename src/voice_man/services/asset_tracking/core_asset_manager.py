"""
Core Asset Management System for Forensic Pipeline
SPEC-ASSET-003: Centralized Asset Lifecycle Management

Manages the lifecycle of core assets:
- Audio recordings (source)
- STT transcripts (derived)
- SER results (derived)
- Forensic scores (derived)
- Reports (final output)
"""

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .uuidv7_manager import (
    AssetMetadata,
    AssetRegistry,
    AssetStatus,
    AssetType,
    get_asset_registry,
)

logger = logging.getLogger(__name__)


class StorageTier(str, Enum):
    """Storage tiers for assets based on access frequency"""

    HOT = "hot"  # Frequently accessed (SSD, fast storage)
    WARM = "warm"  # Occasionally accessed (HDD, standard storage)
    COLD = "cold"  # Rarely accessed (archive, compressed)
    OFFSITE = "offsites"  # Disaster recovery (cloud, remote backup)


@dataclass
class StoragePolicy:
    """Storage policy for different asset types"""

    asset_type: AssetType
    primary_tier: StorageTier
    retention_days: int
    backup_enabled: bool
    compression_enabled: bool
    encryption_enabled: bool = True
    archive_after_days: Optional[int] = None


# Default storage policies
DEFAULT_STORAGE_POLICIES: Dict[AssetType, StoragePolicy] = {
    AssetType.AUDIO_FILE: StoragePolicy(
        asset_type=AssetType.AUDIO_FILE,
        primary_tier=StorageTier.WARM,
        retention_days=365 * 7,  # 7 years (legal requirement)
        backup_enabled=True,
        compression_enabled=False,  # Don't compress original audio
        archive_after_days=90,
    ),
    AssetType.TRANSCRIPT: StoragePolicy(
        asset_type=AssetType.TRANSCRIPT,
        primary_tier=StorageTier.HOT,
        retention_days=365 * 7,
        backup_enabled=True,
        compression_enabled=False,
        archive_after_days=30,
    ),
    AssetType.STT_RESULT: StoragePolicy(
        asset_type=AssetType.STT_RESULT,
        primary_tier=StorageTier.HOT,
        retention_days=365 * 3,
        backup_enabled=True,
        compression_enabled=True,
        archive_after_days=30,
    ),
    AssetType.SER_RESULT: StoragePolicy(
        asset_type=AssetType.SER_RESULT,
        primary_tier=StorageTier.WARM,
        retention_days=365 * 5,
        backup_enabled=True,
        compression_enabled=True,
        archive_after_days=60,
    ),
    AssetType.FORENSIC_SCORE: StoragePolicy(
        asset_type=AssetType.FORENSIC_SCORE,
        primary_tier=StorageTier.HOT,
        retention_days=365 * 10,  # 10 years (forensic evidence)
        backup_enabled=True,
        compression_enabled=False,
        archive_after_days=None,  # Never archive, keep in hot storage
    ),
    AssetType.GASLIGHTING_ANALYSIS: StoragePolicy(
        asset_type=AssetType.GASLIGHTING_ANALYSIS,
        primary_tier=StorageTier.HOT,
        retention_days=365 * 10,
        backup_enabled=True,
        compression_enabled=False,
        archive_after_days=None,
    ),
    AssetType.CRIME_LANGUAGE: StoragePolicy(
        asset_type=AssetType.CRIME_LANGUAGE,
        primary_tier=StorageTier.HOT,
        retention_days=365 * 10,
        backup_enabled=True,
        compression_enabled=False,
        archive_after_days=None,
    ),
    AssetType.CROSS_VALIDATION: StoragePolicy(
        asset_type=AssetType.CROSS_VALIDATION,
        primary_tier=StorageTier.WARM,
        retention_days=365 * 5,
        backup_enabled=True,
        compression_enabled=True,
        archive_after_days=60,
    ),
    AssetType.REPORT_HTML: StoragePolicy(
        asset_type=AssetType.REPORT_HTML,
        primary_tier=StorageTier.WARM,
        retention_days=365 * 10,
        backup_enabled=True,
        compression_enabled=False,
        archive_after_days=180,
    ),
    AssetType.REPORT_PDF: StoragePolicy(
        asset_type=AssetType.REPORT_PDF,
        primary_tier=StorageTier.WARM,
        retention_days=365 * 10,
        backup_enabled=True,
        compression_enabled=False,
        archive_after_days=180,
    ),
    AssetType.BATCH_JOB: StoragePolicy(
        asset_type=AssetType.BATCH_JOB,
        primary_tier=StorageTier.COLD,
        retention_days=365 * 2,
        backup_enabled=True,
        compression_enabled=True,
        archive_after_days=30,
    ),
    AssetType.BATCH_STAGE: StoragePolicy(
        asset_type=AssetType.BATCH_STAGE,
        primary_tier=StorageTier.COLD,
        retention_days=365,
        backup_enabled=True,
        compression_enabled=True,
        archive_after_days=30,
    ),
}


@dataclass
class AssetLocation:
    """Physical location of an asset"""

    tier: StorageTier
    path: str
    server: Optional[str] = None  # For remote storage
    url: Optional[str] = None  # For cloud storage
    checksum_sha256: Optional[str] = None
    size_bytes: Optional[int] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class CoreAssetManager:
    """
    Core asset lifecycle management

    Manages storage, backup, archival, and retrieval of forensic assets.
    """

    def __init__(
        self,
        base_path: Optional[Path] = None,
        registry: Optional[AssetRegistry] = None,
    ):
        """
        Initialize core asset manager

        Args:
            base_path: Base path for storage
            registry: Asset registry
        """
        self.base_path = base_path or Path("data/assets")
        self.registry = registry or get_asset_registry()

        # Create directory structure
        self._create_storage_structure()

        # Storage tracking
        self._asset_locations: Dict[str, AssetLocation] = {}

    def _create_storage_structure(self):
        """Create storage directory structure"""
        for tier in StorageTier:
            tier_path = self.base_path / tier.value
            tier_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created storage tier: {tier_path}")

        # Create subdirectories for each asset type
        for asset_type in AssetType:
            for tier in StorageTier:
                type_path = self.base_path / tier.value / asset_type.value
                type_path.mkdir(parents=True, exist_ok=True)

    def register_audio_file(
        self,
        file_path: str,
        copy_to_storage: bool = True,
        compute_hash: bool = True,
    ) -> AssetMetadata:
        """
        Register an audio file asset

        Args:
            file_path: Path to audio file
            copy_to_storage: Whether to copy file to managed storage
            compute_hash: Whether to compute SHA-256 hash

        Returns:
            AssetMetadata
        """
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        # Compute hash for deduplication
        file_hash = None
        if compute_hash:
            file_hash = self._compute_file_hash(file_path_obj)

            # Check for duplicate
            existing = self.registry.find_by_hash(file_hash)
            if existing:
                logger.info(f"Duplicate file detected: {file_path} -> {existing.asset_id}")
                return existing

        # Get file metadata
        file_size = file_path_obj.stat().st_size
        duration = self._get_audio_duration(file_path_obj)

        # Register asset
        asset = self.registry.register_asset(
            asset_type=AssetType.AUDIO_FILE,
            original_path=str(file_path_obj),
            file_hash=file_hash,
            file_size_bytes=file_size,
            duration_seconds=duration,
        )

        # Copy to storage if requested
        if copy_to_storage:
            storage_path = self._store_asset(asset, file_path_obj)
            asset.storage_path = str(storage_path)
            self.registry.update_asset(asset.asset_id, storage_path=str(storage_path))

        return asset

    def register_derived_asset(
        self,
        asset_type: AssetType,
        parent_id: str,
        content: Any,
        storage_format: str = "json",
        **metadata,
    ) -> AssetMetadata:
        """
        Register a derived asset (transcript, SER result, etc.)

        Args:
            asset_type: Type of derived asset
            parent_id: Parent audio file asset ID
            content: Content to store (dict for JSON, text for TXT, etc.)
            storage_format: Storage format (json, txt, etc.)
            **metadata: Additional metadata

        Returns:
            AssetMetadata
        """
        import json

        # Register asset
        asset = self.registry.register_asset(
            asset_type=asset_type,
            parent_id=parent_id,
            metadata=metadata,
        )

        # Determine storage path
        policy = DEFAULT_STORAGE_POLICIES[asset_type]
        tier_path = self.base_path / policy.primary_tier.value / asset_type.value
        tier_path.mkdir(parents=True, exist_ok=True)

        # Store content
        file_path = tier_path / f"{asset.asset_id}.{storage_format}"

        if storage_format == "json":
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(content, f, indent=2, ensure_ascii=False)
        elif storage_format == "txt":
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(str(content))
        else:
            raise ValueError(f"Unsupported storage format: {storage_format}")

        # Update asset with storage location
        self.registry.update_asset(
            asset.asset_id, storage_path=str(file_path), status=AssetStatus.COMPLETED
        )

        # Track location
        self._asset_locations[asset.asset_id] = AssetLocation(
            tier=policy.primary_tier,
            path=str(file_path),
            size_bytes=file_path.stat().st_size,
            checksum_sha256=self._compute_file_hash(file_path),
        )

        return asset

    def get_asset_lineage(self, asset_id: str) -> List[AssetMetadata]:
        """
        Get full lineage for an asset

        Args:
            asset_id: Asset ID

        Returns:
            List of assets from root to leaf
        """
        return self.registry.get_asset_lineage(asset_id)

    def get_storage_path(self, asset_id: str) -> Optional[str]:
        """
        Get storage path for an asset

        Args:
            asset_id: Asset ID

        Returns:
            Storage path or None if not stored
        """
        asset = self.registry.get_asset(asset_id)
        return asset.storage_path if asset else None

    def archive_asset(self, asset_id: str) -> bool:
        """
        Archive an asset to cold storage

        Args:
            asset_id: Asset ID to archive

        Returns:
            True if successful
        """
        asset = self.registry.get_asset(asset_id)
        if not asset:
            return False

        policy = DEFAULT_STORAGE_POLICIES.get(asset.asset_type)
        if not policy or not policy.archive_after_days:
            logger.info(f"Asset {asset_id} does not support archiving")
            return False

        # Check if archiving is needed
        days_since_creation = (datetime.now(timezone.utc) - asset.created_at).days
        if days_since_creation < policy.archive_after_days:
            logger.info(
                f"Asset {asset_id} not yet ready for archiving ({days_since_creation}/{policy.archive_after_days} days)"
            )
            return False

        # Move to cold storage
        # Implementation depends on storage backend
        # For now, just update status
        self.registry.update_asset(asset_id, status=AssetStatus.ARCHIVED)
        logger.info(f"Archived asset: {asset_id}")

        return True

    def get_asset_by_original_path(self, original_path: str) -> Optional[AssetMetadata]:
        """
        Find asset by original file path

        Args:
            original_path: Original file path

        Returns:
            AssetMetadata or None
        """
        for asset in self.registry._assets.values():
            if asset.original_path == original_path:
                return asset
        return None

    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA-256 hash of a file"""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _get_audio_duration(self, file_path: Path) -> Optional[float]:
        """Get audio file duration in seconds"""
        try:
            import librosa

            y, sr = librosa.load(str(file_path), duration=None)
            duration = librosa.get_duration(y=y, sr=sr)
            return duration
        except Exception as e:
            logger.warning(f"Could not get audio duration: {e}")
            return None

    def _store_asset(self, asset: AssetMetadata, source_path: Path) -> Path:
        """
        Store asset in appropriate storage tier

        Args:
            asset: Asset metadata
            source_path: Source file path

        Returns:
            Destination path
        """
        policy = DEFAULT_STORAGE_POLICIES.get(asset.asset_type)
        if not policy:
            policy = StoragePolicy(
                asset_type=asset.asset_type,
                primary_tier=StorageTier.WARM,
                retention_days=365,
                backup_enabled=True,
                compression_enabled=False,
            )

        tier_path = self.base_path / policy.primary_tier.value / asset.asset_type.value
        tier_path.mkdir(parents=True, exist_ok=True)

        # Create destination path
        file_ext = source_path.suffix
        dest_path = tier_path / f"{asset.asset_id}{file_ext}"

        # Copy file
        import shutil

        shutil.copy2(source_path, dest_path)

        return dest_path

    def get_storage_statistics(self) -> Dict[str, Any]:
        """Get storage statistics"""
        stats = {
            "total_assets": len(self.registry._assets),
            "by_tier": {},
            "by_type": {},
            "total_size_bytes": 0,
        }

        for location in self._asset_locations.values():
            tier = location.tier.value
            stats["by_tier"][tier] = stats["by_tier"].get(tier, 0) + 1
            if location.size_bytes:
                stats["total_size_bytes"] += location.size_bytes

        for asset in self.registry._assets.values():
            asset_type = asset.asset_type.value
            stats["by_type"][asset_type] = stats["by_type"].get(asset_type, 0) + 1

        return stats


# Singleton instance
_core_asset_manager_instance: Optional[CoreAssetManager] = None


def get_core_asset_manager() -> CoreAssetManager:
    """Get singleton CoreAssetManager instance"""
    global _core_asset_manager_instance

    if _core_asset_manager_instance is None:
        _core_asset_manager_instance = CoreAssetManager()

    return _core_asset_manager_instance
