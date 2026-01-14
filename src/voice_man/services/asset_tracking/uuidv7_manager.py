"""
UUID v7 Tracking System for Forensic Pipeline Assets
SPEC-ASSET-001: Universal Asset Tracking with Time-Ordered UUIDs

UUID v7 provides:
- Time-ordered IDs (RFC 9562 compliant)
- 128-bit unique identifiers
- Monotonic randomness within same millisecond
- Sortable by generation time
- No central coordination required

Reference: https://www.rfc-editor.org/rfc/rfc9562.html#name-uuid-version-7
"""

import logging
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID

import struct

logger = logging.getLogger(__name__)


class AssetType(str, Enum):
    """Types of assets in the forensic pipeline"""

    AUDIO_FILE = "audio_file"
    TRANSCRIPT = "transcript"
    STT_RESULT = "stt_result"
    SER_RESULT = "ser_result"
    FORENSIC_SCORE = "forensic_score"
    GASLIGHTING_ANALYSIS = "gaslighting_analysis"
    CRIME_LANGUAGE = "crime_language"
    CROSS_VALIDATION = "cross_validation"
    REPORT_HTML = "report_html"
    REPORT_PDF = "report_pdf"
    BATCH_JOB = "batch_job"
    BATCH_STAGE = "batch_stage"


class AssetStatus(str, Enum):
    """Asset processing status"""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ARCHIVED = "archived"


@dataclass
class AssetMetadata:
    """Metadata for a tracked asset"""

    asset_id: str  # UUID v7 as string
    asset_type: AssetType
    original_path: Optional[str] = None
    file_hash: Optional[str] = None  # SHA-256 for deduplication
    parent_id: Optional[str] = None  # Parent asset ID (e.g., audio -> transcript)
    batch_id: Optional[str] = None  # Batch job ID
    status: AssetStatus = AssetStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    file_size_bytes: Optional[int] = None
    duration_seconds: Optional[float] = None  # For audio files
    storage_path: Optional[str] = None  # Where the asset is stored
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "asset_id": self.asset_id,
            "asset_type": self.asset_type.value,
            "original_path": self.original_path,
            "file_hash": self.file_hash,
            "parent_id": self.parent_id,
            "batch_id": self.batch_id,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "file_size_bytes": self.file_size_bytes,
            "duration_seconds": self.duration_seconds,
            "storage_path": self.storage_path,
            "error_message": self.error_message,
            "metadata": self.metadata,
        }


class UUIDv7Generator:
    """
    UUID v7 Generator (RFC 9562 compliant)

    UUID v7 format:
    - 48 bits: UNIX timestamp in milliseconds (since 1970-01-01)
    - 12 bits: random (within same millisecond)
    - 62 bits: random
    - 6 bits: version and variant

    Structure: XXXXXXXX-XXXX-7XXX-XXXX-XXXXXXXXXXXX
               ^^^^^^^^timestamp  ^^^rand  ^^^^random^^^
    """

    # UUID v7 version and variant bits
    VERSION_BITS = 0b0111 << 4  # Version 7
    VARIANT_BITS = 0b10 << 6  # RFC 4122 variant

    _last_timestamp_ms = 0
    _rand_bits = 0
    _lock = threading.Lock()

    @classmethod
    def generate(cls) -> UUID:
        """
        Generate a UUID v7

        Returns:
            UUID object with version 7
        """
        with cls._lock:
            # Get current timestamp in milliseconds
            timestamp_ms = int(time.time() * 1000)

            # Handle timestamp rollback and same millisecond
            if timestamp_ms < cls._last_timestamp_ms:
                # Clock rollback - use random bits
                timestamp_ms = cls._last_timestamp_ms
                logger.warning("Clock rollback detected, using previous timestamp")
            elif timestamp_ms == cls._last_timestamp_ms:
                # Same millisecond - increment random bits
                cls._rand_bits = (cls._rand_bits + 1) & 0xFFF
                if cls._rand_bits == 0:
                    # Random bits overflow - wait 1ms
                    time.sleep(0.001)
                    timestamp_ms = int(time.time() * 1000)
            else:
                # New millisecond - reset random bits
                cls._rand_bits = 0
                cls._last_timestamp_ms = timestamp_ms

            # Construct UUID v7 bytes
            # Timestamp: 48 bits (6 bytes)
            timestamp_bytes = timestamp_ms.to_bytes(6, byteorder="big")

            # Random: 12 bits + 62 bits = 74 bits (10 bytes minus version/variant)
            # Use cryptographically secure random for the main random part
            import secrets

            rand_a = cls._rand_bits.to_bytes(2, byteorder="big")  # 16 bits
            rand_b = secrets.token_bytes(8)  # 64 bits

            # Combine: timestamp (6) + rand_a (2) + rand_b (8) = 16 bytes
            uuid_bytes = timestamp_bytes + rand_a + rand_b

            # Set version and variant
            # Version goes into byte 6, bits 4-7
            uuid_bytes = bytearray(uuid_bytes)
            uuid_bytes[6] = (uuid_bytes[6] & 0x0F) | cls.VERSION_BITS
            # Variant goes into byte 8, bits 6-7
            uuid_bytes[8] = (uuid_bytes[8] & 0x3F) | cls.VARIANT_BITS

            return UUID(bytes=bytes(uuid_bytes))

    @classmethod
    def generate_str(cls) -> str:
        """
        Generate UUID v7 as string

        Returns:
            UUID v7 as hyphenated string (e.g., "018f1234-5678-7123-4567-89abcdef0123")
        """
        return str(cls.generate())

    @classmethod
    def get_timestamp_ms(cls, uuid_obj: UUID) -> int:
        """
        Extract timestamp from UUID v7

        Args:
            uuid_obj: UUID v7 object

        Returns:
            UNIX timestamp in milliseconds
        """
        uuid_bytes = uuid_obj.bytes
        # First 6 bytes (48 bits) are the timestamp
        timestamp_bytes = uuid_bytes[:6]
        return int.from_bytes(timestamp_bytes, byteorder="big")

    @classmethod
    def get_datetime(cls, uuid_obj: UUID) -> datetime:
        """
        Extract datetime from UUID v7

        Args:
            uuid_obj: UUID v7 object

        Returns:
            Datetime in UTC
        """
        timestamp_ms = cls.get_timestamp_ms(uuid_obj)
        return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)


class AssetRegistry:
    """
    Centralized asset registry with UUID v7 tracking

    Manages all assets in the forensic pipeline with full lineage tracking.
    """

    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize asset registry

        Args:
            storage_path: Path to store asset registry JSON database
        """
        self._assets: Dict[str, AssetMetadata] = {}
        self._storage_path = storage_path or Path("data/asset_registry.json")
        self._lock = threading.Lock()

        # Ensure storage directory exists
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing registry
        self._load()

    def register_asset(
        self,
        asset_type: AssetType,
        original_path: Optional[str] = None,
        file_hash: Optional[str] = None,
        parent_id: Optional[str] = None,
        batch_id: Optional[str] = None,
        **metadata,
    ) -> AssetMetadata:
        """
        Register a new asset with UUID v7

        Args:
            asset_type: Type of asset
            original_path: Original file path
            file_hash: SHA-256 hash for deduplication
            parent_id: Parent asset ID for lineage tracking
            batch_id: Batch job ID
            **metadata: Additional metadata

        Returns:
            AssetMetadata with generated UUID v7
        """
        asset_id = UUIDv7Generator.generate_str()

        asset = AssetMetadata(
            asset_id=asset_id,
            asset_type=asset_type,
            original_path=original_path,
            file_hash=file_hash,
            parent_id=parent_id,
            batch_id=batch_id,
            metadata=metadata,
        )

        with self._lock:
            self._assets[asset_id] = asset
            self._save()

        logger.info(f"Registered asset: {asset_id} (type={asset_type.value})")
        return asset

    def get_asset(self, asset_id: str) -> Optional[AssetMetadata]:
        """
        Get asset metadata by ID

        Args:
            asset_id: UUID v7 asset ID

        Returns:
            AssetMetadata or None if not found
        """
        return self._assets.get(asset_id)

    def update_asset(
        self,
        asset_id: str,
        status: Optional[AssetStatus] = None,
        storage_path: Optional[str] = None,
        error_message: Optional[str] = None,
        **metadata_updates,
    ) -> Optional[AssetMetadata]:
        """
        Update asset metadata

        Args:
            asset_id: UUID v7 asset ID
            status: New status
            storage_path: Where the asset is stored
            error_message: Error message if failed
            **metadata_updates: Additional metadata to update

        Returns:
            Updated AssetMetadata or None if not found
        """
        asset = self._assets.get(asset_id)
        if not asset:
            return None

        if status:
            asset.status = status
            if status == AssetStatus.COMPLETED:
                asset.completed_at = datetime.now(timezone.utc)

        if storage_path:
            asset.storage_path = storage_path

        if error_message:
            asset.error_message = error_message

        asset.updated_at = datetime.now(timezone.utc)
        asset.metadata.update(metadata_updates)

        with self._lock:
            self._save()

        return asset

    def get_assets_by_type(self, asset_type: AssetType) -> List[AssetMetadata]:
        """Get all assets of a specific type"""
        return [a for a in self._assets.values() if a.asset_type == asset_type]

    def get_assets_by_batch(self, batch_id: str) -> List[AssetMetadata]:
        """Get all assets in a batch"""
        return [a for a in self._assets.values() if a.batch_id == batch_id]

    def get_assets_by_parent(self, parent_id: str) -> List[AssetMetadata]:
        """Get all child assets of a parent"""
        return [a for a in self._assets.values() if a.parent_id == parent_id]

    def get_asset_lineage(self, asset_id: str) -> List[AssetMetadata]:
        """
        Get full lineage chain for an asset

        Args:
            asset_id: Asset ID to get lineage for

        Returns:
            List of assets from root to leaf (inclusive)
        """
        lineage = []
        current = self._assets.get(asset_id)

        while current:
            lineage.append(current)
            if current.parent_id:
                current = self._assets.get(current.parent_id)
            else:
                break

        return list(reversed(lineage))

    def find_by_hash(self, file_hash: str) -> Optional[AssetMetadata]:
        """
        Find asset by file hash (for deduplication)

        Args:
            file_hash: SHA-256 hash

        Returns:
            AssetMetadata or None if not found
        """
        for asset in self._assets.values():
            if asset.file_hash == file_hash:
                return asset
        return None

    def _load(self):
        """Load asset registry from disk"""
        if not self._storage_path.exists():
            logger.info(f"Asset registry not found, creating new: {self._storage_path}")
            return

        try:
            import json

            with open(self._storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for asset_data in data.get("assets", []):
                asset = AssetMetadata(
                    asset_id=asset_data["asset_id"],
                    asset_type=AssetType(asset_data["asset_type"]),
                    original_path=asset_data.get("original_path"),
                    file_hash=asset_data.get("file_hash"),
                    parent_id=asset_data.get("parent_id"),
                    batch_id=asset_data.get("batch_id"),
                    status=AssetStatus(asset_data.get("status", AssetStatus.PENDING)),
                    created_at=datetime.fromisoformat(asset_data["created_at"]),
                    updated_at=datetime.fromisoformat(asset_data["updated_at"]),
                    completed_at=datetime.fromisoformat(asset_data["completed_at"])
                    if asset_data.get("completed_at")
                    else None,
                    file_size_bytes=asset_data.get("file_size_bytes"),
                    duration_seconds=asset_data.get("duration_seconds"),
                    storage_path=asset_data.get("storage_path"),
                    error_message=asset_data.get("error_message"),
                    metadata=asset_data.get("metadata", {}),
                )
                self._assets[asset.asset_id] = asset

            logger.info(f"Loaded {len(self._assets)} assets from registry")

        except Exception as e:
            logger.error(f"Failed to load asset registry: {e}")

    def _save(self):
        """Save asset registry to disk"""
        import json

        data = {
            "version": "1.0",
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "total_assets": len(self._assets),
            "assets": [asset.to_dict() for asset in self._assets.values()],
        }

        with open(self._storage_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics"""
        from collections import Counter

        type_counts = Counter(a.asset_type.value for a in self._assets.values())
        status_counts = Counter(a.status.value for a in self._assets.values())

        return {
            "total_assets": len(self._assets),
            "by_type": dict(type_counts),
            "by_status": dict(status_counts),
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }


# Singleton instance
_asset_registry_instance: Optional[AssetRegistry] = None


def get_asset_registry() -> AssetRegistry:
    """
    Get singleton AssetRegistry instance

    Returns:
        AssetRegistry instance
    """
    global _asset_registry_instance

    if _asset_registry_instance is None:
        _asset_registry_instance = AssetRegistry()

    return _asset_registry_instance
