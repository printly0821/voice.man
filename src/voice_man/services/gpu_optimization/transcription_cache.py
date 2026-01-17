"""
Transcription Cache Service for WhisperX Pipeline

Phase 1 Quick Wins:
- L1 Memory Cache (LRU, 100MB default)
- L2 Disk Cache (TTL 24h default)
- Cache key: audio_hash + config_hash
- EARS Requirements: E4 (cache hit detection)

Reference: SPEC-GPUOPT-001 Phase 1
"""

import fcntl
import json
import logging
import pickle
import tempfile
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from voice_man.services.gpu_monitor_service import GPUMonitorService

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Single cache entry with metadata."""

    data: Any
    timestamp: float
    size_bytes: int
    access_count: int = 0
    key: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "key": self.key,
            "timestamp": self.timestamp,
            "size_bytes": self.size_bytes,
            "access_count": self.access_count,
        }


@dataclass
class CacheConfig:
    """Cache configuration."""

    # L1 Memory Cache
    l1_max_size_mb: int = 100  # 100MB default
    l1_max_entries: int = 100

    # L2 Disk Cache
    l2_enabled: bool = True
    l2_ttl_seconds: int = 86400  # 24 hours
    l2_max_size_mb: int = 1000  # 1GB default
    l2_cache_dir: Optional[str] = None

    # Cache Policy
    eviction_policy: str = "lru"  # lru, lfu, ttl

    # Memory Threshold for Auto-adjust
    gpu_memory_threshold: float = 80.0  # 80%


class TranscriptionCache:
    """
    Two-level transcription cache (L1 Memory + L2 Disk).

    L1 Memory Cache:
    - Fast in-memory LRU cache
    - Default: 100MB, 100 entries
    - Sub-millisecond access time

    L2 Disk Cache:
    - Persistent disk cache with TTL
    - Default: 1GB, 24h TTL
    - Used when L1 is full or for persistence

    EARS Requirements:
    - E4: Cache hit detection and skip pipeline
    - S6: TTL-based auto eviction
    - U3: Memory monitoring for auto-adjust
    """

    def __init__(self, config: Optional[CacheConfig] = None):
        """
        Initialize Transcription Cache.

        Args:
            config: Cache configuration (default: default config)
        """
        self.config = config or CacheConfig()

        # L1 Memory Cache (OrderedDict for LRU)
        self._l1_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._l1_lock = threading.RLock()
        self._l1_current_size_bytes = 0

        # L2 Disk Cache setup
        self._l2_enabled = self.config.l2_enabled
        self._l2_cache_dir: Optional[Path] = None
        self._l2_index: Dict[str, Dict[str, Any]] = {}
        self._l2_lock = threading.RLock()

        if self._l2_enabled:
            self._setup_l2_cache()

        # GPU Monitor for memory-based adjustments
        self._gpu_monitor = GPUMonitorService()

        # Statistics
        self._stats = {
            "l1_hits": 0,
            "l1_misses": 0,
            "l2_hits": 0,
            "l2_misses": 0,
            "evictions": 0,
        }

        logger.info(
            f"TranscriptionCache initialized: "
            f"L1={self.config.l1_max_size_mb}MB, "
            f"L2={'enabled' if self._l2_enabled else 'disabled'}"
        )

    def _setup_l2_cache(self) -> None:
        """Setup L2 disk cache directory and index."""
        if self.config.l2_cache_dir:
            self._l2_cache_dir = Path(self.config.l2_cache_dir)
        else:
            # Use temp directory
            cache_root = Path(tempfile.gettempdir()) / "whisperx_cache"
            cache_root.mkdir(parents=True, exist_ok=True)
            self._l2_cache_dir = cache_root

        # Create cache directory
        self._l2_cache_dir.mkdir(parents=True, exist_ok=True)

        # Load existing index
        index_file = self._l2_cache_dir / "index.json"
        if index_file.exists():
            try:
                with open(index_file, "r") as f:
                    self._l2_index = json.load(f)
                logger.info(f"Loaded L2 cache index: {len(self._l2_index)} entries")
            except Exception as e:
                logger.warning(f"Failed to load L2 cache index: {e}")
                self._l2_index = {}

        logger.info(f"L2 cache directory: {self._l2_cache_dir}")

    def _save_l2_index(self) -> None:
        """Save L2 cache index to disk."""
        if not self._l2_cache_dir:
            return

        index_file = self._l2_cache_dir / "index.json"
        try:
            # Write to temp file first, then atomic rename
            temp_file = index_file.with_suffix(".tmp")
            with open(temp_file, "w") as f:
                json.dump(self._l2_index, f)

            # Atomic rename (requires file locking for safety)
            temp_file.replace(index_file)
        except Exception as e:
            logger.warning(f"Failed to save L2 cache index: {e}")

    def _get_entry_size(self, data: Any) -> int:
        """
        Estimate size of cache entry in bytes.

        Args:
            data: Data to estimate size for

        Returns:
            Size in bytes
        """
        try:
            return len(pickle.dumps(data))
        except Exception:
            # Fallback: rough estimate
            return len(str(data).encode())

    def _evict_l1(self, required_bytes: int) -> None:
        """
        Evict entries from L1 cache to make room.

        Args:
            required_bytes: Bytes to free up
        """
        freed_bytes = 0
        while (
            self._l1_cache
            and self._l1_current_size_bytes + required_bytes
            > self.config.l1_max_size_mb * 1024 * 1024
        ):
            # Pop oldest entry (LRU)
            key, entry = self._l1_cache.popitem(last=False)
            freed_bytes += entry.size_bytes
            self._stats["evictions"] += 1

            logger.debug(f"Evicted L1 entry: {key} ({entry.size_bytes} bytes)")

        self._l1_current_size_bytes -= freed_bytes

    def _cleanup_l2_by_ttl(self) -> None:
        """Clean up expired L2 cache entries based on TTL."""
        if not self._l2_enabled or not self._l2_cache_dir:
            return

        current_time = time.time()
        expired_keys = []

        with self._l2_lock:
            for key, metadata in self._l2_index.items():
                if current_time - metadata.get("timestamp", 0) > self.config.l2_ttl_seconds:
                    expired_keys.append(key)

            for key in expired_keys:
                self._remove_l2_entry(key)

            if expired_keys:
                logger.info(f"Cleaned up {len(expired_keys)} expired L2 entries")
                self._save_l2_index()

    def _remove_l2_entry(self, key: str) -> None:
        """
        Remove entry from L2 cache.

        Args:
            key: Cache key to remove
        """
        if key in self._l2_index:
            # Delete cache file
            cache_file = self._l2_cache_dir / f"{key}.cache"
            if cache_file.exists():
                try:
                    cache_file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to delete cache file {cache_file}: {e}")

            # Remove from index
            del self._l2_index[key]

    def get(self, key: str) -> Optional[Any]:
        """
        Get cached transcription result.

        E4: Return cached result if available

        Args:
            key: Cache key (audio_hash + config_hash)

        Returns:
            Cached data or None if not found
        """
        # Check L1 first
        with self._l1_lock:
            if key in self._l1_cache:
                entry = self._l1_cache[key]
                entry.access_count += 1
                # Move to end (most recently used)
                self._l1_cache.move_to_end(key)
                self._stats["l1_hits"] += 1
                logger.debug(f"L1 cache hit: {key}")
                return entry.data
            else:
                self._stats["l1_misses"] += 1

        # Check L2
        if self._l2_enabled:
            with self._l2_lock:
                if key in self._l2_index:
                    metadata = self._l2_index[key]

                    # Check TTL
                    if time.time() - metadata.get("timestamp", 0) > self.config.l2_ttl_seconds:
                        # Expired
                        self._remove_l2_entry(key)
                        self._stats["l2_misses"] += 1
                        return None

                    # Load from disk
                    cache_file = self._l2_cache_dir / f"{key}.cache"
                    if cache_file.exists():
                        try:
                            with open(cache_file, "rb") as f:
                                data = pickle.load(f)

                            # Promote to L1
                            self._put_l1(key, data)

                            self._stats["l2_hits"] += 1
                            logger.debug(f"L2 cache hit: {key}")
                            return data
                        except Exception as e:
                            logger.warning(f"Failed to load L2 cache entry {key}: {e}")
                            self._remove_l2_entry(key)

                self._stats["l2_misses"] += 1

        return None

    def _put_l1(self, key: str, data: Any) -> bool:
        """
        Put data into L1 cache.

        Args:
            key: Cache key
            data: Data to cache

        Returns:
            True if successful
        """
        entry_size = self._get_entry_size(data)

        # Check if entry is too large for L1
        max_entry_size = self.config.l1_max_size_mb * 1024 * 1024 // 2  # 50% of max
        if entry_size > max_entry_size:
            logger.debug(f"Entry too large for L1: {key} ({entry_size} bytes)")
            return False

        # Evict if necessary
        self._evict_l1(entry_size)

        # Add entry
        entry = CacheEntry(
            data=data,
            timestamp=time.time(),
            size_bytes=entry_size,
            key=key,
        )

        self._l1_cache[key] = entry
        self._l1_current_size_bytes += entry_size
        return True

    def put(self, key: str, data: Any) -> None:
        """
        Cache transcription result.

        Args:
            key: Cache key (audio_hash + config_hash)
            data: Data to cache (transcription result)
        """
        # Try L1 first
        with self._l1_lock:
            if self._put_l1(key, data):
                logger.debug(f"Cached in L1: {key}")

        # Also cache to L2 for persistence
        if self._l2_enabled and self._l2_cache_dir:
            with self._l2_lock:
                try:
                    cache_file = self._l2_cache_dir / f"{key}.cache"

                    # Write to temp file first
                    temp_file = cache_file.with_suffix(".tmp")
                    with open(temp_file, "wb") as f:
                        pickle.dump(data, f)

                    # Atomic rename
                    temp_file.replace(cache_file)

                    # Update index
                    self._l2_index[key] = {
                        "timestamp": time.time(),
                        "size_bytes": self._get_entry_size(data),
                    }

                    # Save index
                    self._save_l2_index()

                    logger.debug(f"Cached in L2: {key}")

                except Exception as e:
                    logger.warning(f"Failed to cache to L2: {e}")

    def clear(self) -> None:
        """Clear all cache entries (L1 and L2)."""
        # Clear L1
        with self._l1_lock:
            self._l1_cache.clear()
            self._l1_current_size_bytes = 0

        # Clear L2
        if self._l2_enabled and self._l2_cache_dir:
            with self._l2_lock:
                for cache_file in self._l2_cache_dir.glob("*.cache"):
                    try:
                        cache_file.unlink()
                    except Exception as e:
                        logger.warning(f"Failed to delete {cache_file}: {e}")

                self._l2_index.clear()
                self._save_l2_index()

        logger.info("Cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        with self._l1_lock:
            l1_stats = {
                "entries": len(self._l1_cache),
                "size_bytes": self._l1_current_size_bytes,
                "size_mb": self._l1_current_size_bytes / (1024 * 1024),
                "max_size_mb": self.config.l1_max_size_mb,
                "utilization_percent": (
                    self._l1_current_size_bytes / (self.config.l1_max_size_mb * 1024 * 1024) * 100
                    if self.config.l1_max_size_mb > 0
                    else 0
                ),
            }

        with self._l2_lock:
            l2_stats = {
                "enabled": self._l2_enabled,
                "entries": len(self._l2_index),
                "cache_dir": str(self._l2_cache_dir) if self._l2_cache_dir else None,
            }

        total_requests = (
            self._stats["l1_hits"]
            + self._stats["l1_misses"]
            + self._stats["l2_hits"]
            + self._stats["l2_misses"]
        )
        total_hits = self._stats["l1_hits"] + self._stats["l2_hits"]

        return {
            "l1": l1_stats,
            "l2": l2_stats,
            "hits": total_hits,
            "misses": self._stats["l1_misses"] + self._stats["l2_misses"],
            "hit_rate": (total_hits / total_requests * 100) if total_requests > 0 else 0,
            "evictions": self._stats["evictions"],
        }

    def cleanup_expired(self) -> int:
        """
        Clean up expired cache entries.

        S6: TTL-based auto eviction

        Returns:
            Number of entries cleaned up
        """
        initial_count = len(self._l2_index)
        self._cleanup_l2_by_ttl()
        return initial_count - len(self._l2_index)

    def auto_adjust_memory(self) -> None:
        """
        Auto-adjust cache based on GPU memory status.

        U3: Memory monitoring for auto-adjust
        S1: CPU fallback when GPU memory critical
        """
        memory_status = self._gpu_monitor.check_memory_status()

        if memory_status.get("auto_adjust_recommended", False):
            # Reduce L1 cache size by 50%
            with self._l1_lock:
                target_size = self._l1_current_size_bytes // 2
                freed = 0

                while self._l1_cache and self._l1_current_size_bytes > target_size:
                    key, entry = self._l1_cache.popitem(last=False)
                    freed += entry.size_bytes
                    self._stats["evictions"] += 1

                self._l1_current_size_bytes -= freed

                logger.info(
                    f"Auto-adjusted L1 cache: freed {freed / (1024 * 1024):.2f}MB "
                    f"due to GPU memory pressure"
                )

    def __del__(self):
        """Cleanup on destruction."""
        try:
            self._save_l2_index()
        except Exception:
            pass
