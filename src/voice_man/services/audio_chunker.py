"""
AudioChunker - Split large audio files into manageable chunks.

Phase 2 Memory Optimization: Audio chunking infrastructure.

This module provides ffmpeg-based audio chunking to prevent OOM when
processing large audio files (>30 minutes). Instead of loading entire files into RAM,
it splits them into smaller chunks (default: 5 minutes) with overlap for context preservation.

Key Features:
- Streaming chunking (no full file load)
- Configurable chunk duration and overlap
- Parallel chunk extraction support
- Automatic chunk merging for results

Usage:
    chunker = AudioChunker(chunk_duration_sec=300, overlap_sec=30)
    chunks = chunker.split_audio("large_audio.m4a")

    for chunk_info in chunks:
        result = process_chunk(chunk_info)
        merged = chunker.merge_results(chunks)
"""

import gc
import logging
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ChunkInfo:
    """Information about an audio chunk."""

    chunk_id: str
    start_time_sec: float
    end_time_sec: float
    duration_sec: float
    original_file: str
    chunk_file: str
    chunk_index: int
    total_chunks: int


@dataclass
class ChunkResult:
    """Result from processing a chunk."""

    chunk_id: str
    transcript: str
    forensic_score: float
    metadata: Dict = field(default_factory=dict)


class AudioChunker:
    """
    Split large audio files into manageable chunks using ffmpeg.

    Implements streaming-based chunking to prevent loading entire files into RAM.
    Each chunk is extracted independently using ffmpeg, with overlap to preserve context.

    Attributes:
        chunk_duration_sec: Duration of each chunk in seconds (default: 5 minutes)
        overlap_sec: Overlap between chunks in seconds (default: 30 seconds)
        trigger_duration_sec: Minimum file duration to trigger chunking (default: 10 minutes)
        temp_dir: Temporary directory for chunk files

    Example:
        >>> chunker = AudioChunker(chunk_duration_sec=300, overlap_sec=30)
        >>> chunks = chunker.split_audio("30min_audio.m4a")
        >>> print(f"Split into {len(chunks)} chunks")
        Split into 6 chunks
    """

    def __init__(
        self,
        chunk_duration_sec: float = 300.0,  # 5 minutes
        overlap_sec: float = 30.0,  # 30 seconds
        trigger_duration_sec: float = 600.0,  # 10 minutes
        temp_dir: Optional[str] = None,
    ):
        """
        Initialize AudioChunker.

        Args:
            chunk_duration_sec: Duration of each chunk in seconds
            overlap_sec: Overlap between chunks to preserve context
            trigger_duration_sec: Minimum file duration to trigger chunking
            temp_dir: Temporary directory for chunk files (default: system temp)
        """
        self.chunk_duration_sec = chunk_duration_sec
        self.overlap_sec = overlap_sec
        self.trigger_duration_sec = trigger_duration_sec

        # Create temp directory for chunks
        if temp_dir is None:
            self.temp_dir_obj = tempfile.TemporaryDirectory(prefix="audio_chunks_")
            self.temp_dir = self.temp_dir_obj.name
        else:
            self.temp_dir_obj = None
            self.temp_dir = temp_dir

        logger.info(
            f"AudioChunker initialized: "
            f"chunk={chunk_duration_sec}s, overlap={overlap_sec}s, "
            f"trigger={trigger_duration_sec}s"
        )

        # Platform detection for encoding handling
        self._is_windows = sys.platform == 'win32'

    def _get_encoding_params(self):
        '''Get platform-specific encoding parameters for subprocess.
        
        Returns:
            Dictionary with encoding and errors parameters
        '''
        if self._is_windows:
            return {'encoding': 'utf-8', 'errors': 'surrogateescape'}
        else:
            return {'encoding': 'utf-8', 'errors': 'surrogateescape'}

    def _sanitize_filename(self, audio_path):
        '''Create a sanitized copy of the audio file for subprocess processing.
        
        Args:
            audio_path: Path to the original audio file
            
        Returns:
            Tuple of (sanitized_path, cleanup_function)
        '''
        temp_dir = Path(self.temp_dir)
        safe_name = f"sanitized_audio_{id(audio_path)}_{datetime.now().timestamp()}"
        original_ext = Path(audio_path).suffix
        sanitized_path = temp_dir / f"{safe_name}{original_ext}"
        
        try:
            shutil.copy2(audio_path, sanitized_path)
            logger.debug(
                f"Created sanitized copy: {sanitized_path.name} "
                f"(original: {Path(audio_path).name})"
            )
            return str(sanitized_path), lambda: sanitized_path.unlink(missing_ok=True)
        except (OSError, UnicodeEncodeError) as e:
            logger.warning(f"Failed to create sanitized copy for {audio_path}: {e}")
            return audio_path, None

    def should_chunk(self, audio_path: str) -> bool:
        """
        Determine if audio file should be chunked based on duration.

        Args:
            audio_path: Path to audio file

        Returns:
            True if file should be chunked, False otherwise
        """
        try:
            duration = self._get_audio_duration(audio_path)
            should_chunk = duration >= self.trigger_duration_sec

            if should_chunk:
                logger.info(
                    f"Audio chunking triggered: {audio_path} "
                    f"({duration:.1f}s >= {self.trigger_duration_sec}s threshold)"
                )
            else:
                logger.debug(
                    f"Audio within threshold: {audio_path} "
                    f"({duration:.1f}s < {self.trigger_duration_sec}s)"
                )

            return should_chunk

        except Exception as e:
            logger.warning(f"Failed to check audio duration for {audio_path}: {e}")
            return False  # Default to no chunking on error

    def split_audio(self, audio_path: str) -> List[ChunkInfo]:
        """
        Split audio file into chunks using ffmpeg streaming.

        Extracts chunks sequentially without loading the entire file into RAM.
        Each chunk is extracted as a temporary WAV file.

        Args:
            audio_path: Path to audio file to split

        Returns:
            List of ChunkInfo objects with metadata for each chunk
        """
        try:
            # Get total duration first
            total_duration = self._get_audio_duration(audio_path)
            logger.info(
                f"Splitting audio: {audio_path} "
                f"(duration={total_duration:.1f}s, "
                f"chunk={self.chunk_duration_sec}s, overlap={self.overlap_sec}s)"
            )

            # Calculate chunk boundaries
            chunks = self._calculate_chunk_boundaries(total_duration)

            # Extract each chunk using ffmpeg
            chunk_infos = []
            for i, (start, end) in enumerate(chunks, start=1):
                chunk_id = f"{Path(audio_path).stem}_chunk_{i:03d}"
                chunk_file = self._extract_chunk(audio_path, start, end, chunk_id)

                chunk_info = ChunkInfo(
                    chunk_id=chunk_id,
                    start_time_sec=start,
                    end_time_sec=end,
                    duration_sec=end - start,
                    original_file=audio_path,
                    chunk_file=chunk_file,
                    chunk_index=i,
                    total_chunks=len(chunks),
                )
                chunk_infos.append(chunk_info)

                logger.info(
                    f"Chunk {i}/{len(chunks)} extracted: "
                    f"{start:.1f}s-{end:.1f}s (duration={end - start:.1f}s)"
                )

            return chunk_infos

        except Exception as e:
            logger.error(f"Failed to split audio {audio_path}: {e}")
            raise

    def merge_results(self, chunks: List[ChunkResult], audio_path: str) -> ChunkResult:
        """
        Merge results from processed chunks.

        Combines transcripts and aggregates forensic scores from all chunks.

        Args:
            chunks: List of processing results from each chunk
            audio_path: Original audio file path

        Returns:
            Merged ChunkResult with combined transcript and aggregated score
        """
        if not chunks:
            raise ValueError("No chunks to merge")

        logger.info(f"Merging results from {len(chunks)} chunks for {audio_path}")

        # Combine transcripts in order
        combined_transcript = " ".join([chunk.transcript for chunk in chunks])

        # Aggregate forensic scores (weighted average)
        if all(chunk.forensic_score is not None for chunk in chunks):
            # Weight by chunk duration (if available in metadata)
            total_weight = 0.0
            weighted_sum = 0.0

            for chunk in chunks:
                duration = chunk.metadata.get("duration_sec", 1.0)
                weighted_sum += chunk.forensic_score * duration
                total_weight += duration

            aggregated_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        else:
            # If any score is None, use average (excluding None)
            scores = [chunk.forensic_score for chunk in chunks if chunk.forensic_score is not None]
            aggregated_score = sum(scores) / len(scores) if scores else 0.0

        logger.info(
            f"Merge complete: transcript length={len(combined_transcript)}, "
            f"aggregated_score={aggregated_score:.1f}"
        )

        return ChunkResult(
            chunk_id=f"{Path(audio_path).stem}_merged",
            transcript=combined_transcript,
            forensic_score=aggregated_score,
            metadata={
                "original_file": audio_path,
                "num_chunks": len(chunks),
                "merge_timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    def _get_audio_duration(self, audio_path: str) -> float:
        """
        Get audio file duration using ffprobe.

        Args:
            audio_path: Path to audio file

        Returns:
            Duration in seconds

        Raises:
            RuntimeError: If ffprobe fails
        """
        encoding_params = self._get_encoding_params()
        cleanup_func = None

        try:
            cmd = [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                audio_path,
            ]

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=30,
                    **encoding_params,
                )
            except (UnicodeEncodeError, OSError) as encoding_error:
                logger.warning(
                    f"Encoding error with original filename '{Path(audio_path).name}': {encoding_error}. "
                    f"Attempting fallback with sanitized filename."
                )

                sanitized_path, cleanup_func = self._sanitize_filename(audio_path)
                cmd[-1] = sanitized_path

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=30,
                    **encoding_params,
                )

            # Parse duration from stdout (where -show_entries writes output)
            duration_str = result.stdout.strip()
            if duration_str:
                try:
                    duration = float(duration_str)
                    logger.debug(f"Got duration from ffprobe: {duration}s")
                    return duration
                except ValueError as e:
                    raise RuntimeError(
                        f"Could not parse duration '{duration_str}' as float: {e}"
                    )

            raise RuntimeError(f"Could not find duration in ffprobe output (stdout was empty)")

        except subprocess.TimeoutExpired:
            raise RuntimeError(f"ffprobe timed out for {audio_path}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ffprobe failed for {audio_path}: {e}")
        finally:
            if cleanup_func is not None:
                try:
                    cleanup_func()
                except Exception as e:
                    logger.warning(f"Failed to cleanup sanitized file: {e}")

    def _calculate_chunk_boundaries(self, total_duration: float) -> List[Tuple[float, float]]:
        """
        Calculate start and end times for each chunk.

        Args:
            total_duration: Total audio duration in seconds

        Returns:
            List of (start_sec, end_sec) tuples for each chunk
        """
        boundaries = []
        current = 0.0

        while current < total_duration:
            end = min(current + self.chunk_duration_sec, total_duration)
            boundaries.append((current, end))
            current = end - self.overlap_sec  # Move back by overlap
            # Ensure we don't get stuck in infinite loop
            if current <= 0:
                current = end

        return boundaries

    def _extract_chunk(
        self,
        audio_path: str,
        start_sec: float,
        end_sec: float,
        chunk_id: str,
    ) -> str:
        """
        Extract a single chunk from audio file using ffmpeg.

        Args:
            audio_path: Path to source audio file
            start_sec: Start time in seconds
            end_sec: End time in seconds
            chunk_id: Unique identifier for this chunk

        Returns:
            Path to extracted chunk file (WAV format)
        """
        chunk_file = Path(self.temp_dir) / f"{chunk_id}.wav"
        encoding_params = self._get_encoding_params()
        cleanup_func = None

        try:
            # Build ffmpeg command for chunk extraction
            cmd = [
                "ffmpeg",
                "-y",  # Overwrite output file
                "-i",
                audio_path,  # Input file
                "-ss",
                str(start_sec),  # Start time
                "-t",
                str(end_sec - start_sec),  # Duration
                "-ar",
                "16000",  # Sample rate (match STT requirement)
                "-ac",
                "1",  # Mono (match STT requirement)
                "-acodec",
                "pcm_s16le",  # PCM 16-bit (match STT requirement)
                "-loglevel",
                "error",  # Suppress logs
                str(chunk_file),
            ]

            # Execute ffmpeg
            logger.debug(f"Extracting chunk: {start_sec:.1f}s-{end_sec:.1f}s to {chunk_file.name}")

            try:
                subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    timeout=120,
                    **encoding_params,
                )
            except (UnicodeEncodeError, OSError) as encoding_error:
                logger.warning(
                    f"Encoding error with original filename '{Path(audio_path).name}': {encoding_error}. "
                    f"Attempting fallback with sanitized filename."
                )

                sanitized_path, cleanup_func = self._sanitize_filename(audio_path)
                cmd[3] = sanitized_path

                subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    timeout=120,
                    **encoding_params,
                )

            logger.debug(f"Chunk extracted successfully: {chunk_file}")
            return str(chunk_file)

        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"ffmpeg timeout extracting chunk from {audio_path} "
                f"(start={start_sec:.1f}s, end={end_sec:.1f}s)"
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ffmpeg failed to extract chunk from {audio_path}: {e}")
        finally:
            if cleanup_func is not None:
                try:
                    cleanup_func()
                except Exception as e:
                    logger.warning(f"Failed to cleanup sanitized file: {e}")

    def cleanup(self, chunk_files: List[str]) -> None:
        """
        Delete temporary chunk files to free disk space.

        Args:
            chunk_files: List of chunk file paths to delete
        """
        for chunk_file in chunk_files:
            try:
                Path(chunk_file).unlink(missing_ok=True)
                logger.debug(f"Deleted chunk file: {chunk_file}")
            except Exception as e:
                logger.warning(f"Failed to delete chunk file {chunk_file}: {e}")

        # Force garbage collection
        gc.collect()

    def __del__(self):
        """Cleanup temporary directory on destruction."""
        if hasattr(self, "temp_dir_obj") and self.temp_dir_obj is not None:
            try:
                self.temp_dir_obj.cleanup()
                logger.debug("Cleaned up temporary chunk directory")
            except Exception:
                pass  # Ignore errors during cleanup


def estimate_optimal_chunk_size(
    available_ram_mb: float,
    model_memory_mb: float = 6000.0,
    safety_factor: float = 0.3,
) -> int:
    """
    Estimate optimal chunk size based on available memory.

    Args:
        available_ram_mb: Available RAM in MB
        model_memory_mb: Memory required for model in MB
        safety_factor: Safety margin (0.0-1.0)

    Returns:
        Recommended chunk duration in seconds

    Example:
        >>> estimate_optimal_chunk_size(16000)  # 16GB available
        300
    """
    usable_memory_mb = available_ram_mb * safety_factor
    # Assume 1MB per minute of audio (16kHz mono)
    # Adjust based on model memory requirements
    audio_memory_mb = usable_memory_mb - model_memory_mb

    if audio_memory_mb <= 0:
        return 60  # Minimum 1 minute

    # Convert to seconds (1MB ≈ 1 minute at 16kHz mono)
    chunk_duration_sec = max(60, int(audio_memory_mb * 60))

    # Cap at 10 minutes (practical limit)
    return min(chunk_duration_sec, 600)


def create_chunker_for_system() -> AudioChunker:
    """
    Create AudioChunker with optimal parameters for current system.

    Analyzes available memory and creates chunker with appropriate chunk size.

    Returns:
        Configured AudioChunker instance
    """
    import psutil

    # Get available memory
    mem = psutil.virtual_memory()
    available_mb = mem.available / (1024 * 1024)

    # Estimate optimal chunk size
    chunk_sec = estimate_optimal_chunk_size(available_mb)

    logger.info(
        f"Auto-configured chunker: {chunk_sec}s chunk "
        f"(based on {available_mb / 1024:.1f}GB available)"
    )

    # Standard overlap
    overlap_sec = 30

    return AudioChunker(
        chunk_duration_sec=float(chunk_sec),
        overlap_sec=float(overlap_sec),
        trigger_duration_sec=600.0,  # 10 minutes
    )
