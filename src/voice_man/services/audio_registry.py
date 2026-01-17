"""
AudioRegistry - Centralized audio loading cache to prevent duplicate loads.

Phase 2 Memory Optimization: Smart caching policy to prevent duplicate audio loading.

This module provides a singleton registry that loads audio files once and caches
them as numpy arrays for use across multiple services (STT, Forensic, etc.).
This prevents the same audio file from being loaded multiple times into memory.

Key Features:
- Singleton pattern (one registry per process)
- Audio data caching with numpy arrays
- Memory usage tracking
- Automatic cleanup when memory pressure is high
- Reference counting for cache entries

Usage:
    registry = AudioRegistry.get_instance()

    # Load audio once
    audio_data = registry.load_audio("audio.m4a", sample_rate=16000)

    # Get same audio data from cache (no reload)
    audio_data2 = registry.load_audio("audio.m4a", sample_rate=16000)
    # Returns same numpy array object (reference)

    # Clear cache when needed
    registry.clear_audio("audio.m4a")
"""

import gc
import logging
import threading
import weakref
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AudioCacheEntry:
    """Cached audio data with metadata."""

    file_path: str
    audio_data: np.ndarray
    sample_rate: int
    duration: float
    file_size_bytes: int
    last_accessed: datetime
    reference_count: int = 0
    memory_size_mb: float = 0.0


class AudioRegistry:
    """
    Centralized audio loading cache with memory pressure-aware cleanup.

    Implements singleton pattern to ensure only one registry exists per process.
    Audio files are loaded once and cached as numpy arrays, preventing duplicate
    loads across STT and Forensic services.

    Attributes:
        _instance: Singleton instance
        _cache: Dictionary mapping file paths to cache entries
        _lock: Thread lock for thread-safe access
        _max_cache_mb: Maximum cache size in MB before cleanup
        _cache_hits: Number of cache hits (for metrics)
        _cache_misses: Number of cache misses (for metrics)
    """

    _instance: Optional["AudioRegistry"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self, max_cache_mb: float = 512.0):
        """
        Initialize AudioRegistry.

        Args:
            max_cache_mb: Maximum cache size in MB before cleanup is triggered
        """
        self._cache: Dict[str, AudioCacheEntry] = {}
        self._max_cache_mb = max_cache_mb
        self._cache_hits = 0
        self._cache_misses = 0

        # Initialize memory tracking
        self._current_cache_mb = 0.0
        self._peak_cache_mb = 0.0

        logger.info(
            f"AudioRegistry initialized: max_cache={max_cache_mb}MB, "
            f"cache={len(self._cache)} entries"
        )

    @classmethod
    def get_instance(cls) -> "AudioRegistry":
        """
        Get the singleton AudioRegistry instance.

        Returns:
            The singleton AudioRegistry instance

        Example:
            >>> registry = AudioRegistry.get_instance()
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(max_cache_mb=512.0)
        return cls._instance

    def load_audio(self, file_path: str, sample_rate: int = 16000) -> Tuple[np.ndarray, int]:
        """
        Load audio file from cache or load if not cached.

        Args:
            file_path: Path to audio file
            sample_rate: Target sample rate (default: 16000Hz)

        Returns:
            Tuple of (audio_data, sample_rate)

        Example:
            >>> registry = AudioRegistry.get_instance()
            >>> audio, sr = registry.load_audio("test.m4a", sample_rate=16000)
            >>> audio.shape
            (16000 * duration,)
        """
        with self._lock:
            # Check cache
            if file_path in self._cache:
                entry = self._cache[file_path]
                entry.reference_count += 1
                entry.last_accessed = datetime.now(timezone.utc)
                self._cache_hits += 1

                logger.debug(
                    f"Cache hit: {file_path} "
                    f"(refs={entry.reference_count}, "
                    f"size={entry.memory_size_mb:.1f}MB)"
                )

                return entry.audio_data, entry.sample_rate

            # Cache miss - load audio file
            logger.debug(f"Cache miss: {file_path}")
            self._cache_misses += 1

            # Load audio using librosa
            import librosa

            try:
                audio_data, sr = librosa.load(file_path, sr=sample_rate)

                # Create cache entry
                entry = AudioCacheEntry(
                    file_path=file_path,
                    audio_data=audio_data,
                    sample_rate=sr,
                    duration=len(audio_data) / sr,
                    file_size_bytes=Path(file_path).stat().st_size,
                    last_accessed=datetime.now(timezone.utc),
                    reference_count=1,
                )

                # Calculate memory size
                entry.memory_size_mb = audio_data.nbytes / (1024 * 1024)

                # Add to cache
                self._cache[file_path] = entry
                self._current_cache_mb += entry.memory_size_mb
                self._peak_cache_mb = max(self._peak_cache_mb, self._current_cache_mb)

                logger.info(
                    f"Audio loaded and cached: {file_path} "
                    f"(size={entry.memory_size_mb:.1f}MB, duration={entry.duration:.1f}s)"
                )

                # Check cache size and cleanup if needed
                if self._current_cache_mb > self._max_cache_mb:
                    self._cleanup_excess_entries()

                return audio_data, sr

            except Exception as e:
                logger.error(f"Failed to load audio {file_path}: {e}")
                raise

    def get_audio_info(self, file_path: str) -> Optional[AudioCacheEntry]:
        """
        Get cached audio information without loading.

        Args:
            file_path: Path to audio file

        Returns:
            AudioCacheEntry if cached, None otherwise
        """
        with self._lock:
            return self._cache.get(file_path)

    def clear_audio(self, file_path: str) -> None:
        """
        Clear specific audio from cache.

        Args:
            file_path: Path to audio file to clear
        """
        with self._lock:
            if file_path in self._cache:
                entry = self._cache[file_path]

                # Update memory tracking
                self._current_cache_mb -= entry.memory_size_mb

                # Remove from cache
                del self._cache[file_path]

                # Force garbage collection
                del entry
                gc.collect()

                logger.debug(f"Cleared audio cache: {file_path}")
            else:
                logger.debug(f"Audio not in cache (skipping): {file_path}")

    def clear_all(self) -> None:
        """Clear all cached audio data and reset memory tracking."""
        with self._lock:
            # Calculate total memory to be freed
            total_freed = self._current_cache_mb

            # Clear cache dictionary
            self._cache.clear()

            # Reset memory tracking
            self._current_cache_mb = 0.0

            # Force garbage collection
            gc.collect()

            logger.info(f"Cleared all audio cache: {total_freed:.1f}MB freed")

    def get_cache_stats(self) -> Dict[str, any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache metrics
        """
        with self._lock:
            return {
                "total_entries": len(self._cache),
                "current_cache_mb": self._current_cache_mb,
                "peak_cache_mb": self._peak_cache_mb,
                "cache_hits": self._cache_hits,
                "cache_misses": self._cache_misses,
                "hit_rate": (
                    self._cache_hits / (self._cache_hits + self._cache_misses)
                    if (self._cache_hits + self._cache_misses) > 0
                    else 0.0
                ),
            }

    def _cleanup_excess_entries(self) -> int:
        """
        Clean up least recently used entries when cache is full.

        Removes entries until cache size is below threshold.

        Returns:
            Number of entries removed
        """
        with self._lock:
            # Sort entries by last accessed time (oldest first)
            entries_by_access = sorted(self._cache.values(), key=lambda e: e.last_accessed)

            freed_mb = 0
            removed_count = 0

            threshold_mb = self._max_cache_mb * 0.7  # Clean up to 70% capacity

            for entry in entries_by_access:
                # Stop if we're below threshold
                if self._current_cache_mb - freed_mb <= threshold_mb:
                    break

                # Remove entry
                del self._cache[entry.file_path]
                freed_mb += entry.memory_size_mb
                removed_count += 1

            self._current_cache_mb -= freed_mb

            logger.info(f"Cache cleanup: removed {removed_count} entries, freed {freed_mb:.1f}MB")

            # Force garbage collection
            gc.collect()

            return removed_count

    def get_memory_pressure(self) -> str:
        """
        Get current memory pressure level based on cache usage.

        Returns:
            Memory pressure level: LOW, MEDIUM, HIGH, or CRITICAL
        """
        with self._lock:
            usage_percent = (self._current_cache_mb / self._max_cache_mb) * 100

            if usage_percent < 50:
                return "LOW"
            elif usage_percent < 70:
                return "MEDIUM"
            elif usage_percent < 90:
                return "HIGH"
            else:
                return "CRITICAL"


# Convenience function for getting singleton instance
def get_audio_registry() -> AudioRegistry:
    """
    Get the singleton AudioRegistry instance.

    Example:
        >>> registry = get_audio_registry()
        >>> audio, sr = registry.load_audio("test.m4a", sample_rate=16000)
    """
    return AudioRegistry.get_instance()


def clear_audio_cache() -> None:
    """
    Clear all cached audio data.

    Example:
        >>> clear_audio_cache()
    """
    registry = get_audio_registry()
    registry.clear_all()


# Module-level exports
__all__ = [
    "AudioRegistry",
    "AudioCacheEntry",
    "get_audio_registry",
    "clear_audio_cache",
]
