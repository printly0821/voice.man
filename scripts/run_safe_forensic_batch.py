#!/usr/bin/env python3
"""
Memory-Safe Forensic Batch Processing Script with Checkpointing
===============================================================

Processes audio files in batches with fault-tolerant checkpointing:
- Checkpoint-based recovery from crashes
- Automatic resume from last checkpoint
- Error classification with intelligent retry
- Graceful shutdown with state preservation
- Progress tracking with ETA
- Memory safety: Aggressive cleanup between batches
- Auto/Prompt mode: Auto-continue if safe, prompt user if concerns detected

Hardware Considerations:
- RAM: 119GB total, 115GB available
- Swap: 15GB
- CPU: 20 cores (Cortex-X925, Cortex-A725)
- GPU: Temperature 38°C, Power draw 4.23W

Usage:
    # Normal run
    python scripts/run_safe_forensic_batch.py --auto --batch-size 10 @ref/*.m4a

    # Resume from checkpoint
    python scripts/run_safe_forensic_batch.py --auto --batch-size 10 --resume @ref/*.m4a

    # Custom checkpoint directory
    python scripts/run_safe_forensic_batch.py --checkpoint-dir /tmp/checkpoints --auto @ref/*.m4a
"""

import asyncio
import gc
import logging
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import checkpoint modules
from voice_man.services.checkpoint import (
    CheckpointManager,
    ErrorClassifier,
    ErrorCategory,
    GracefulShutdown,
    ProgressTracker,
)
from voice_man.services.checkpoint.state_store import WorkflowStatus

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/safe_batch_pipeline.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


@dataclass
class BatchResult:
    """Result of a single batch processing operation."""

    batch_number: int
    total_files: int
    successful: int
    failed: int
    skipped: int
    retries: int
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    memory_before_mb: float
    memory_after_mb: float
    memory_peak_mb: float
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_number": self.batch_number,
            "total_files": self.total_files,
            "successful": self.successful,
            "failed": self.failed,
            "skipped": self.skipped,
            "retries": self.retries,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration_seconds": self.duration_seconds,
            "memory_before_mb": self.memory_before_mb,
            "memory_after_mb": self.memory_after_mb,
            "memory_peak_mb": self.memory_peak_mb,
            "errors": self.errors,
        }


@dataclass
class DynamicBatchAdjuster:
    """
    Dynamically adjusts batch size based on memory pressure level.

    Implements proactive batch size adjustment to prevent OOM:
    - LOW pressure (<50%): Use full batch size
    - MEDIUM pressure (50-70%): Reduce to 50%
    - HIGH pressure (70-90%): Reduce to 33%
    - CRITICAL pressure (>90%): Use minimum batch size

    This complements the reactive OOMRecoveryHandler by preventing
    OOM before it occurs.
    """

    memory_monitor: "MemoryMonitor"
    initial_batch_size: int
    min_batch_size: int = 1
    max_batch_size: int = 10
    current_batch_size: int = 0

    def __post_init__(self):
        """Initialize current batch size."""
        if self.current_batch_size == 0:
            self.current_batch_size = self.initial_batch_size

    def get_memory_pressure_level(self) -> str:
        """Get current memory pressure level based on RAM usage."""
        stats = self.memory_monitor.get_memory_stats()
        ram_percent = stats["ram_used_mb"] / stats["ram_total_mb"] * 100

        if ram_percent < 50:
            return "LOW"
        elif ram_percent < 70:
            return "MEDIUM"
        elif ram_percent < 90:
            return "HIGH"
        else:
            return "CRITICAL"

    def calculate_batch_size(self, pressure_level: str) -> int:
        """Calculate optimal batch size based on memory pressure."""
        if pressure_level == "LOW":
            return min(self.max_batch_size, self.initial_batch_size)
        elif pressure_level == "MEDIUM":
            return max(self.min_batch_size, self.initial_batch_size // 2)
        elif pressure_level == "HIGH":
            return max(self.min_batch_size, self.initial_batch_size // 3)
        else:  # CRITICAL
            return self.min_batch_size

    def adjust_batch_size(self) -> int:
        """Adjust batch size based on current memory pressure.

        Returns the adjusted batch size and logs the change if any.
        """
        pressure_level = self.get_memory_pressure_level()
        new_batch_size = self.calculate_batch_size(pressure_level)

        if new_batch_size != self.current_batch_size:
            logger.info(
                f"Memory pressure: {pressure_level} ({self.get_ram_percent():.1f}%), "
                f"adjusting batch size: {self.current_batch_size} -> {new_batch_size}"
            )
            self.current_batch_size = new_batch_size

        return self.current_batch_size

    def get_ram_percent(self) -> float:
        """Get current RAM usage percentage."""
        stats = self.memory_monitor.get_memory_stats()
        return stats["ram_used_mb"] / stats["ram_total_mb"] * 100


@dataclass
class OOMRecoveryHandler:
    """
    Handle OOM errors with graceful recovery using batch size reduction.

    Strategy:
    - Detect OOM from RuntimeError exception
    - Reduce batch size by 50%
    - Clear memory aggressively
    - Retry with smaller batch
    """

    oom_count: int = 0
    max_oom_retries: int = 3
    min_batch_size: int = 1

    # Pre-compiled OOM patterns for efficient matching
    OOM_PATTERNS: tuple = (
        "out of memory",
        "oom",
        "memory error",
        "cuda.*out.*of.*memory",
        "gpu.*out.*of.*memory",
        "allocation.*failed",
        "cuda.*memory",
    )

    def handle_oom(self, exception: Exception, current_batch_size: int, logger) -> Optional[int]:
        """
        Handle OOM error by reducing batch size.

        Args:
            exception: The exception that occurred
            current_batch_size: Current batch size
            logger: Logger instance

        Returns:
            New batch size to retry with, or None if should not retry
        """
        error_msg = str(exception).lower()

        # Check if this is an OOM error using pre-compiled class patterns
        is_oom = any(pattern in error_msg for pattern in self.OOM_PATTERNS)

        if not is_oom:
            return None

        self.oom_count += 1

        if self.oom_count > self.max_oom_retries:
            logger.error(
                f"Max OOM retries reached ({self.max_oom_retries}), "
                f"cannot recover from batch size {current_batch_size}"
            )
            return None

        # Calculate new batch size (reduce more aggressively on first OOM)
        # First OOM: reduce to 1/3, subsequent OOMs: reduce by half
        if self.oom_count == 1:
            new_batch_size = max(self.min_batch_size, current_batch_size // 3)
        else:
            new_batch_size = max(self.min_batch_size, current_batch_size // 2)

        logger.warning(
            f"OOM detected (count={self.oom_count}/{self.max_oom_retries}), "
            f"reducing batch size: {current_batch_size} -> {new_batch_size}"
        )

        # Aggressive memory cleanup (gc already imported at module level)
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        logger.info("Memory cleared, ready to retry with smaller batch")

        return new_batch_size

    def reset(self) -> None:
        """Reset OOM counter (for new batch)."""
        self.oom_count = 0


@dataclass
class ProcessingResult:
    """Result of the entire processing operation."""

    total_files: int
    total_batches: int
    total_successful: int
    total_failed: int
    total_skipped: int
    total_retries: int
    total_duration_seconds: float
    batches: List[BatchResult] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_files": self.total_files,
            "total_batches": self.total_batches,
            "total_successful": self.total_successful,
            "total_failed": self.total_failed,
            "total_skipped": self.total_skipped,
            "total_retries": self.total_retries,
            "total_duration_seconds": self.total_duration_seconds,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "batches": [b.to_dict() for b in self.batches],
        }


class MemoryMonitor:
    """Monitor and manage system memory usage."""

    @staticmethod
    def get_memory_stats() -> Dict[str, float]:
        """Get current memory statistics in MB."""
        try:
            import psutil

            mem = psutil.virtual_memory()
            swap = psutil.swap_memory()

            return {
                "ram_total_mb": mem.total / (1024 * 1024),
                "ram_used_mb": mem.used / (1024 * 1024),
                "ram_available_mb": mem.available / (1024 * 1024),
                "ram_percent": mem.percent,
                "swap_total_mb": swap.total / (1024 * 1024),
                "swap_used_mb": swap.used / (1024 * 1024),
                "swap_percent": swap.percent,
            }
        except ImportError:
            # Fallback to /proc/meminfo on Linux
            try:
                import re

                meminfo = {}
                with open("/proc/meminfo") as f:
                    for line in f:
                        match = re.match(
                            r"(MemTotal|MemFree|MemAvailable|Cached|SwapTotal|SwapFree):\s+(\d+)\s+kB",
                            line,
                        )
                        if match:
                            meminfo[match.group(1)] = int(match.group(2)) / 1024  # Convert to MB

                total = meminfo.get("MemTotal", 0)
                available = meminfo.get(
                    "MemAvailable", meminfo.get("MemFree", 0) + meminfo.get("Cached", 0)
                )
                used = total - available

                return {
                    "ram_total_mb": total,
                    "ram_used_mb": used,
                    "ram_available_mb": available,
                    "ram_percent": (used / total * 100) if total > 0 else 0,
                    "swap_total_mb": meminfo.get("SwapTotal", 0),
                    "swap_used_mb": meminfo.get("SwapTotal", 0) - meminfo.get("SwapFree", 0),
                    "swap_percent": 0,
                }
            except Exception as e:
                logger.warning(f"Failed to get memory stats: {e}")
                return {
                    "ram_total_mb": 0,
                    "ram_used_mb": 0,
                    "ram_available_mb": 0,
                    "ram_percent": 0,
                    "swap_total_mb": 0,
                    "swap_used_mb": 0,
                    "swap_percent": 0,
                }

    @staticmethod
    def check_memory_pressure() -> Tuple[bool, str]:
        """
        Check if memory pressure is too high.

        Returns:
            Tuple of (is_critical, message)
        """
        stats = MemoryMonitor.get_memory_stats()

        # Conservative thresholds
        if stats["ram_percent"] > 85:
            return True, f"High RAM usage: {stats['ram_percent']:.1f}%"

        if stats["swap_percent"] > 50:
            return True, f"High swap usage: {stats['swap_percent']:.1f}%"

        if stats["ram_available_mb"] < 4096:  # Less than 4GB available
            return True, f"Low available RAM: {stats['ram_available_mb']:.1f}MB"

        return False, "Memory OK"

    @staticmethod
    def aggressive_cleanup() -> float:
        """
        Perform aggressive memory cleanup with enhanced GPU memory release.

        Returns:
            Memory freed in MB (estimated)
        """
        before_stats = MemoryMonitor.get_memory_stats()
        before_used = before_stats["ram_used_mb"]

        logger.info("Starting aggressive memory cleanup...")

        # Python garbage collection (multiple passes for better cleanup)
        # Phase 1: Aggressive garbage collection (oldest generation first)
        # This is more effective than simple repeated calls
        gc.collect(generation=2)  # Collect oldest generation first
        gc.collect(generation=1)  # Then middle generation
        gc.collect(generation=0)  # Finally youngest generation

        # Clear forensic model caches
        try:
            from voice_man.services.forensic.ser_service import SERService

            SERService.clear_model_cache()
            logger.debug("Cleared SER model cache")
        except Exception as e:
            logger.debug(f"Could not clear SER cache: {e}")

        # PyTorch GPU memory cleanup (enhanced)
        try:
            import torch

            if torch.cuda.is_available():
                # Clear CUDA memory allocator cache
                torch.cuda.empty_cache()

                # Force synchronization to ensure all operations complete
                torch.cuda.synchronize()

                # Additional cleanup for PyTorch 1.12+
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()
                    logger.debug("Performed IPC memory collection")

                # Get memory summary for debugging
                allocated = torch.cuda.memory_allocated() / (1024**3)
                reserved = torch.cuda.memory_reserved() / (1024**3)
                logger.debug(f"GPU memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

        except ImportError:
            logger.debug("PyTorch not available for GPU cleanup")
        except Exception as e:
            logger.warning(f"Error during GPU cleanup: {e}")

        # Phase 3: Final cleanup pass after GPU operations
        gc.collect()  # Single final pass is sufficient after previous generational collection

        after_stats = MemoryMonitor.get_memory_stats()
        after_used = after_stats["ram_used_mb"]

        freed = before_used - after_used
        if freed > 0:
            logger.info(f"Aggressive cleanup freed ~{freed:.1f}MB RAM")

        return max(0, freed)


class SafeBatchProcessor:
    """
    Memory-safe batch processor for forensic analysis with checkpointing.

    Conservative resource management:
    - Small batch sizes (5-8 files)
    - Aggressive memory cleanup between batches
    - Retry mechanism (3 attempts before skip)
    - Auto continue between batches
    - Checkpoint-based crash recovery
    - Error classification with intelligent retry
    - Graceful shutdown with state preservation
    """

    def __init__(
        self,
        audio_dir: Path = Path("ref/call"),
        batch_size: int = 3,  # Conservative default to prevent OOM
        max_retries: int = 3,
        max_files: Optional[int] = None,
        enable_gpu: bool = True,
        prompt_between_batches: bool = True,
        checkpoint_dir: str = "data/checkpoints",
        resume: bool = False,
    ):
        """
        Initialize Safe Batch Processor.

        Args:
            audio_dir: Directory containing audio files
            batch_size: Number of files per batch (default: 10)
            max_retries: Maximum retry attempts before skipping (default: 3)
            max_files: Maximum files to process (for testing)
            enable_gpu: Enable GPU acceleration
            prompt_between_batches: Prompt user before each batch (default: True)
            checkpoint_dir: Directory for checkpoint storage
            resume: Resume from last checkpoint
        """
        self.audio_dir = audio_dir
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.max_files = max_files
        self.enable_gpu = enable_gpu
        self.prompt_between_batches = prompt_between_batches
        self.resume = resume

        # Services (lazy initialization)
        self._stt_service = None
        self._forensic_service = None

        # Memory monitor
        self.memory_monitor = MemoryMonitor()

        # Dynamic batch adjuster for proactive memory management
        self.batch_adjuster = DynamicBatchAdjuster(
            memory_monitor=self.memory_monitor,
            initial_batch_size=batch_size,
            min_batch_size=1,
            max_batch_size=10,
        )

        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(checkpoint_dir=checkpoint_dir)

        # Error classifier
        self.error_classifier = ErrorClassifier()

        # Progress tracker
        self.progress_tracker: Optional[ProgressTracker] = None

        # Graceful shutdown handler
        self.shutdown_handler = GracefulShutdown(timeout=30.0)

        # OOM recovery handler
        self.oom_handler = OOMRecoveryHandler(
            oom_count=0,
            max_oom_retries=3,
            min_batch_size=1,
        )

        logger.info(
            f"SafeBatchProcessor initialized: "
            f"batch_size={self.batch_size}, "
            f"max_retries={self.max_retries}, "
            f"gpu={self.enable_gpu}, "
            f"checkpoint_dir={checkpoint_dir}, "
            f"resume={resume}"
        )

    def _initialize_services(self):
        """Initialize all required services."""
        logger.info("Initializing services...")

        # Asset tracking
        from voice_man.services.asset_tracking import AssetRegistry, CoreAssetManager
        from voice_man.services.whisperx_service import WhisperXService
        from voice_man.services.forensic.audio_feature_service import AudioFeatureService
        from voice_man.services.forensic.stress_analysis_service import StressAnalysisService
        from voice_man.services.forensic.ser_service import SERService
        from voice_man.services.forensic.crime_language_service import CrimeLanguageAnalysisService
        from voice_man.services.forensic.cross_validation_service import CrossValidationService
        from voice_man.services.forensic.forensic_scoring_service import ForensicScoringService

        # Asset tracking
        self.asset_registry = AssetRegistry()
        self.core_asset_manager = CoreAssetManager()

        # STT service
        device = "cuda" if self.enable_gpu else "cpu"
        self._stt_service = WhisperXService(device=device, language="ko")

        # Forensic services
        audio_feature_service = AudioFeatureService()
        stress_analysis_service = StressAnalysisService()
        ser_service = SERService()
        crime_language_service = CrimeLanguageAnalysisService()
        cross_validation_service = CrossValidationService(
            crime_language_service=crime_language_service,
            ser_service=ser_service,
        )

        self._forensic_service = ForensicScoringService(
            audio_feature_service=audio_feature_service,
            stress_analysis_service=stress_analysis_service,
            crime_language_service=crime_language_service,
            ser_service=ser_service,
            cross_validation_service=cross_validation_service,
        )

        logger.info("Services initialized successfully")

    def _initialize_stt_service(self):
        """Initialize or reinitialize STT service only."""
        from voice_man.services.whisperx_service import WhisperXService

        logger.info("Initializing STT service...")
        device = "cuda" if self.enable_gpu else "cpu"
        self._stt_service = WhisperXService(device=device, language="ko")
        logger.info("STT service initialized")

    def get_audio_files(self) -> List[Path]:
        """Get all audio files for processing."""
        audio_files = sorted(self.audio_dir.glob("*.m4a"))

        if self.max_files:
            audio_files = audio_files[: self.max_files]

        logger.info(f"Found {len(audio_files)} audio files for processing")
        return audio_files

    async def process_single_file(
        self,
        audio_file: Path,
        retry_count: int = 0,
    ) -> Tuple[bool, Optional[str], Optional[Dict]]:
        """
        Process a single file with retry mechanism.

        Args:
            audio_file: Path to audio file
            retry_count: Current retry attempt

        Returns:
            Tuple of (success, error_message, result_data)
        """
        file_path_str = str(audio_file)

        try:
            # Step 1: STT (transcription only)
            logger.info(f"STT processing: {audio_file.name} (attempt {retry_count + 1})")
            stt_result = await self._stt_service.transcribe_only(file_path_str)

            if "error" in stt_result:
                raise Exception(f"STT failed: {stt_result['error']}")

            segments = stt_result.get("segments", [])
            text = " ".join(seg.get("text", "") for seg in segments)

            if not text:
                raise Exception("Transcription produced empty text")

            logger.info(f"STT complete: {len(segments)} segments, {len(text)} chars")

            # Step 2: Forensic scoring
            logger.info(f"Forensic scoring: {audio_file.name}")
            forensic_result = await self._forensic_service.analyze(
                audio_path=file_path_str,
                transcript=text,
            )

            logger.info(f"Forensic complete: risk={forensic_result.overall_risk_score:.1f}")

            return (
                True,
                None,
                {
                    "file": file_path_str,
                    "transcript_segments": len(segments),
                    "transcript_chars": len(text),
                    "forensic_score": forensic_result.overall_risk_score,
                },
            )

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed to process {audio_file.name}: {error_msg}")
            return False, error_msg, None

    async def process_file_with_retry(
        self,
        audio_file: Path,
    ) -> Tuple[bool, int, Optional[str], Optional[Dict]]:
        """
        Process a file with retry mechanism and error classification.

        Args:
            audio_file: Path to audio file

        Returns:
            Tuple of (success, attempts, final_error, result_data)
        """
        last_error = None
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                success, error, result = await self.process_single_file(audio_file, attempt)

                if success:
                    return True, attempt + 1, None, result

                last_error = error

                # Classify error to determine retry strategy
                if last_exception:
                    category, severity = self.error_classifier.classify_error(last_exception)
                    strategy = self.error_classifier.determine_retry_strategy(category)

                    if not strategy.should_retry:
                        logger.warning(f"Error classified as {category.value}, not retrying")
                        break

                    # Cleanup if needed before retry
                    if strategy.cleanup_before_retry:
                        logger.info("Performing cleanup before retry...")
                        self.memory_monitor.aggressive_cleanup()

                    # Calculate backoff
                    backoff = self.error_classifier.get_backoff_seconds(attempt, category, strategy)
                    logger.warning(f"Retry in {backoff:.1f}s... (category: {category.value})")
                    await asyncio.sleep(backoff)

                    # Fallback to CPU if needed
                    if strategy.fallback_to_cpu and attempt == self.max_retries - 1:
                        logger.info("Attempting CPU fallback...")
                        self.enable_gpu = False
                        # Re-initialize STT service with CPU only (not all services)
                        if hasattr(self._stt_service, "unload"):
                            self._stt_service.unload()
                        self._stt_service = None
                        self._initialize_stt_service()

                elif attempt < self.max_retries - 1:
                    wait_time = 2**attempt  # Exponential backoff: 1, 2, 4 seconds
                    logger.warning(f"Retry in {wait_time}s...")
                    await asyncio.sleep(wait_time)

            except Exception as e:
                last_exception = e
                last_error = str(e)

                # Classify error
                category, severity = self.error_classifier.classify_error(e)
                strategy = self.error_classifier.determine_retry_strategy(category)

                logger.error(f"Attempt {attempt + 1} failed: {type(e).__name__}: {last_error}")

                if not strategy.should_retry:
                    logger.warning(f"Error classified as {category.value}, not retrying")
                    break

                if attempt < self.max_retries - 1:
                    # Cleanup if needed
                    if strategy.cleanup_before_retry:
                        logger.info("Performing cleanup before retry...")
                        self.memory_monitor.aggressive_cleanup()

                    # Calculate backoff
                    backoff = self.error_classifier.get_backoff_seconds(attempt, category, strategy)
                    logger.warning(f"Retry in {backoff:.1f}s... (category: {category.value})")
                    await asyncio.sleep(backoff)

        return False, self.max_retries, last_error, None

    async def process_batch(
        self,
        batch_number: int,
        files: List[Path],
    ) -> BatchResult:
        """
        Process a batch of files with memory management.

        Args:
            batch_number: Batch number (1-indexed)
            files: List of audio files to process

        Returns:
            BatchResult with processing statistics
        """
        start_time = datetime.now(timezone.utc)
        memory_before = self.memory_monitor.get_memory_stats()
        peak_memory = memory_before["ram_used_mb"]

        logger.info(f"\n{'=' * 70}")
        logger.info(f"BATCH {batch_number}: Processing {len(files)} files")
        logger.info(f"{'=' * 70}")

        # Reinitialize services if they were unloaded after previous batch
        if self._stt_service is None:
            logger.info("STT service not loaded, initializing...")
            self._initialize_stt_service()

        logger.info(
            f"Memory before: RAM {memory_before['ram_used_mb']:.1f}MB / {memory_before['ram_total_mb']:.1f}MB ({memory_before['ram_percent']:.1f}%)"
        )

        successful = 0
        failed = 0
        skipped = 0
        total_retries = 0
        errors = []

        for i, audio_file in enumerate(files, 1):
            file_path_str = str(audio_file)

            # Check if file already processed (from previous crash recovery)
            from voice_man.services.checkpoint.state_store import FileStatus

            existing_file_state = self.checkpoint_manager.state_store.get_file_state(file_path_str)
            if existing_file_state and existing_file_state.status == FileStatus.COMPLETED:
                logger.info(f"\n[{i}/{len(files)}] Skipping (already completed): {audio_file.name}")
                skipped += 1
                continue

            logger.info(f"\n[{i}/{len(files)}] Processing: {audio_file.name}")

            # Mark file as processing
            self.checkpoint_manager.mark_file_processing(file_path_str)

            # Check memory pressure before processing
            is_critical, mem_msg = self.memory_monitor.check_memory_pressure()
            if is_critical:
                logger.warning(f"Memory pressure detected: {mem_msg}")
                logger.info("Performing aggressive cleanup...")
                freed = self.memory_monitor.aggressive_cleanup()
                logger.info(f"Cleanup completed, freed ~{freed:.1f}MB")

            # Process file with retry
            success, attempts, error, result = await self.process_file_with_retry(audio_file)
            total_retries += attempts - 1

            if success:
                successful += 1
                logger.info(f"SUCCESS: {audio_file.name}")

                # Mark file as completed in checkpoint
                self.checkpoint_manager.mark_file_completed(
                    file_path_str,
                    result={"success": True, "attempts": attempts},
                )
            else:
                failed += 1
                error_msg = f"{audio_file.name}: {error}"
                errors.append(error_msg)
                logger.error(f"FAILED (after {attempts} attempts): {error_msg}")

                # Mark file as failed in checkpoint
                self.checkpoint_manager.mark_file_failed(
                    file_path_str,
                    error_message=error or "Unknown error",
                    attempts=attempts,
                )

            # Check peak memory
            memory_current = self.memory_monitor.get_memory_stats()
            peak_memory = max(peak_memory, memory_current["ram_used_mb"])

        # Aggressive cleanup after batch
        logger.info(f"\nBatch {batch_number} completed. Performing aggressive cleanup...")
        freed = self.memory_monitor.aggressive_cleanup()

        # CRITICAL FIX: Actually unload models between batches to prevent OOM
        if hasattr(self, "_stt_service") and self._stt_service is not None:
            logger.info("Unloading STT service models to free GPU memory...")
            try:
                self._stt_service.unload()
                self._stt_service = None
                logger.info("STT service models unloaded successfully")
            except Exception:
                logger.exception("Failed to unload STT service")

        # Clear forensic service model caches
        if hasattr(self, "_forensic_service"):
            from voice_man.services.forensic.ser_service import SERService

            logger.info("Clearing forensic model caches...")
            try:
                SERService.clear_model_cache()
            except Exception:
                logger.exception("Failed to clear SER cache")

        end_time = datetime.now(timezone.utc)
        duration = (end_time - start_time).total_seconds()

        memory_after = self.memory_monitor.get_memory_stats()

        result = BatchResult(
            batch_number=batch_number,
            total_files=len(files),
            successful=successful,
            failed=failed,
            skipped=skipped,
            retries=total_retries,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            memory_before_mb=memory_before["ram_used_mb"],
            memory_after_mb=memory_after["ram_used_mb"],
            memory_peak_mb=peak_memory,
            errors=errors,
        )

        # Save checkpoint after batch completion
        self._save_batch_checkpoint(batch_number, files, result)

        logger.info(f"\n{'=' * 70}")
        logger.info(f"BATCH {batch_number} SUMMARY:")
        logger.info(f"  Files: {successful}/{len(files)} successful")
        logger.info(f"  Failed: {failed}, Skipped: {skipped}")
        logger.info(f"  Retries: {total_retries}")
        logger.info(f"  Duration: {duration:.1f}s")
        logger.info(
            f"  Memory: Before {memory_before['ram_used_mb']:.1f}MB -> After {memory_after['ram_used_mb']:.1f}MB"
        )
        logger.info(f"  Peak: {peak_memory:.1f}MB, Freed: ~{freed:.1f}MB")

        # Apply dynamic batch size adjustment for next batch based on current memory pressure
        next_batch_size = self.batch_adjuster.adjust_batch_size()
        if next_batch_size != self.batch_size:
            logger.info(
                f"  Next batch size: {self.batch_size} -> {next_batch_size} (based on memory pressure)"
            )
            self.batch_size = next_batch_size
        else:
            logger.info(f"  Next batch size: {self.batch_size} (unchanged)")

        logger.info(f"{'=' * 70}\n")

        return result

    def _save_batch_checkpoint(
        self,
        batch_number: int,
        files: List[Path],
        result: BatchResult,
    ) -> None:
        """
        Save checkpoint after batch completion.

        Args:
            batch_number: Batch number
            files: Files in the batch
            result: Batch processing result
        """
        try:
            # Determine processed and failed files
            processed_files = []
            failed_files = []

            # Extract from result errors to determine which files failed
            failed_filenames = set()
            for error_msg in result.errors:
                # Extract filename from error message (format: "filename: error")
                if ":" in error_msg:
                    filename = error_msg.split(":", 1)[0].strip()
                    failed_filenames.add(filename)

            # All files in batch
            for file_path in files:
                if file_path.name in failed_filenames:
                    failed_files.append(str(file_path))
                else:
                    processed_files.append(str(file_path))

            # Save checkpoint
            batch_id = f"batch_{batch_number}"
            self.checkpoint_manager.save_batch_checkpoint(
                batch_id=batch_id,
                processed_files=processed_files,
                failed_files=failed_files,
                results={
                    "batch_number": batch_number,
                    "total_files": result.total_files,
                    "successful": result.successful,
                    "failed": result.failed,
                    "skipped": result.skipped,
                    "retries": result.retries,
                    "duration_seconds": result.duration_seconds,
                    "errors": result.errors,
                },
            )

            logger.info(f"Checkpoint saved for batch {batch_number}")

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    async def run_all_batches(self) -> ProcessingResult:
        """
        Run all batches with checkpoint-based crash recovery.

        Returns:
            ProcessingResult with overall statistics
        """
        start_time = datetime.now(timezone.utc)

        # Initialize services
        self._initialize_services()

        # Get all files
        all_files = self.get_audio_files()
        total_files = len(all_files)

        # Setup graceful shutdown first (before creating workflow)
        self.shutdown_handler.setup()

        # Check for resume state FIRST (before creating new workflow)
        workflow_id = None
        start_batch = 1
        processed_files = set()
        failed_files = set()

        if self.resume:
            # Try to find existing workflow to resume
            logger.info("Looking for existing workflow to resume...")

            # Check for CRASHED or RUNNING workflows
            from voice_man.services.checkpoint.state_store import WorkflowStatus

            crashed_workflows = self.checkpoint_manager.state_store.list_workflows(
                status=WorkflowStatus.CRASHED
            )
            running_workflows = self.checkpoint_manager.state_store.list_workflows(
                status=WorkflowStatus.RUNNING
            )

            existing_workflows = crashed_workflows + running_workflows

            if existing_workflows:
                # Use the most recent workflow
                workflow_id = existing_workflows[0].workflow_id
                logger.info(f"Found existing workflow: {workflow_id}")

                # Set current workflow in checkpoint manager
                self.checkpoint_manager._current_workflow_id = workflow_id

                # Get resume state
                resume_state = self.checkpoint_manager.get_resume_state()
                if resume_state and resume_state.can_resume:
                    logger.info(f"Resuming from checkpoint: {resume_state.workflow_id}")
                    logger.info(f"  Processed: {len(resume_state.processed_files)} files")
                    logger.info(f"  Failed: {len(resume_state.failed_files)} files")
                    logger.info(f"  Current batch: {resume_state.current_batch}")

                    processed_files = set(resume_state.processed_files)
                    failed_files = set(resume_state.failed_files)
                    start_batch = resume_state.current_batch

                    # Filter out already processed files
                    all_files = [f for f in all_files if str(f) not in processed_files]
                    logger.info(f"  Remaining files: {len(all_files)}")
                else:
                    logger.warning(f"Workflow {workflow_id} found but no valid resume state")
                    workflow_id = None

            if not workflow_id:
                logger.info("No existing workflow found, starting fresh")

        # Create new workflow only if not resuming
        if not workflow_id:
            workflow_id = f"workflow_{int(start_time.timestamp())}"
            self.checkpoint_manager.create_workflow(
                workflow_id=workflow_id,
                total_files=total_files,
                batch_size=self.batch_size,
                metadata={"audio_dir": str(self.audio_dir)},
            )

            # Register all files for tracking
            all_file_strs = [str(f) for f in all_files]
            for i, file_path in enumerate(all_file_strs):
                batch_id = f"batch_{(i // self.batch_size) + 1}"
                self.checkpoint_manager.register_files([file_path], batch_id)

            logger.info(f"Created new workflow: {workflow_id}")
        else:
            logger.info(f"Reusing existing workflow: {workflow_id}")
            # Update workflow status to RUNNING
            self.checkpoint_manager.state_store.update_workflow_state(
                workflow_id, status=WorkflowStatus.RUNNING
            )

            # Register remaining files for tracking (for resumed workflows)
            all_file_strs = [str(f) for f in all_files]
            for i, file_path in enumerate(all_file_strs):
                # Calculate batch number based on remaining files
                batch_idx_in_remaining = i // self.batch_size
                batch_number = start_batch + batch_idx_in_remaining
                batch_id = f"batch_{batch_number}"

                # Only register if not already registered
                existing_state = self.checkpoint_manager.state_store.get_file_state(file_path)
                if existing_state is None:
                    self.checkpoint_manager.register_files([file_path], batch_id)

            logger.info(f"Registered {len(all_file_strs)} remaining files for tracking")

        # Setup shutdown handler for this workflow
        @self.shutdown_handler.on_shutdown
        def save_on_shutdown():
            """Save checkpoint on shutdown."""
            logger.info("Shutdown signal received, saving checkpoint...")
            self.checkpoint_manager.state_store.update_workflow_state(
                workflow_id, status=WorkflowStatus.CRASHED
            )

        if not self.resume:
            logger.info("Resume mode disabled, starting fresh")

        # Apply dynamic batch size adjustment based on memory pressure
        adjusted_batch_size = self.batch_adjuster.adjust_batch_size()
        if adjusted_batch_size != self.batch_size:
            logger.info(f"Dynamic batch adjustment: {self.batch_size} -> {adjusted_batch_size}")
            self.batch_size = adjusted_batch_size

        # Create batches from remaining files
        batches = []
        for i in range(0, len(all_files), self.batch_size):
            batch = all_files[i : i + self.batch_size]
            batches.append(batch)

        # Adjust batch numbers
        total_batches = (total_files + self.batch_size - 1) // self.batch_size
        logger.info(f"\nProcessing plan: {len(all_files)} files in {len(batches)} batches")
        logger.info(f"Total files in workflow: {total_files}")
        logger.info(f"Starting from batch: {start_batch}/{total_batches}")
        logger.info(f"Batch size: {self.batch_size} files/batch")
        logger.info(f"Retry policy: {self.max_retries} attempts before skip")
        logger.info(
            f"System verification: {'Enabled' if self.prompt_between_batches else 'Auto-continue'}\n"
        )

        # Initialize progress tracker
        self.progress_tracker = ProgressTracker(
            total_items=total_files,
            enable_console=True,
        )

        # Process all batches
        batch_results = []
        total_successful = len(processed_files)
        total_failed = len(failed_files)
        total_skipped = 0
        total_retries = 0

        for batch_idx, batch_files in enumerate(batches, start=start_batch):
            # Check for shutdown signal
            if self.shutdown_handler.is_shutdown_requested():
                logger.info("Shutdown requested, stopping batch processing")
                break

            try:
                result = await self.process_batch(batch_idx, batch_files)
                batch_results.append(result)

                total_successful += result.successful
                total_failed += result.failed
                total_skipped += result.skipped
                total_retries += result.retries

                # Update progress
                self.progress_tracker.update(
                    current=total_successful,
                    failed=total_failed,
                )

                # System verification after batch
                if batch_idx < total_batches:  # Not the last batch
                    should_continue = await self._verify_and_prompt(
                        batch_idx, result, total_batches
                    )
                    if not should_continue:
                        logger.info("User requested to stop processing")
                        break

            except Exception as e:
                logger.error(f"Batch {batch_idx} failed with exception: {e}")
                logger.error(traceback.format_exc())

                # CRITICAL FIX: Check for OOM error and use recovery handler
                new_batch_size = self.oom_handler.handle_oom(
                    exception=e, current_batch_size=self.batch_size, logger=logger
                )
                if new_batch_size is not None:
                    # CRITICAL FIX: Store old batch size before reduction
                    old_batch_size = self.batch_size
                    self.batch_size = new_batch_size
                    logger.info(
                        f"OOM detected! Reduced batch size: {old_batch_size} -> {self.batch_size}"
                    )

                    # CRITICAL FIX: Calculate batch start position using original batch size
                    batch_start_idx = (batch_idx - 1) * old_batch_size
                    batch_end_idx = batch_start_idx + new_batch_size

                    # Validate indices to prevent infinite loop
                    if batch_start_idx >= len(all_files):
                        logger.error(
                            f"Batch start index {batch_start_idx} exceeds total files {len(all_files)}"
                        )
                        break

                    # Recreate batch with reduced size
                    batch_files = all_files[batch_start_idx:batch_end_idx]

                    if not batch_files:
                        # Check if there are remaining files
                        remaining_files = len(all_files) - batch_start_idx
                        if remaining_files > 0:
                            logger.error(
                                f"Cannot process {remaining_files} remaining files even with "
                                f"minimum batch size {new_batch_size}. Aborting to prevent infinite loop."
                            )
                            break
                        else:
                            logger.info("All files processed successfully")
                            break

                    logger.info(
                        f"Retrying with {len(batch_files)} files (original batch would have had {old_batch_size})"
                    )

                    # CRITICAL FIX: Reset OOM counter after successful recovery preparation
                    self.oom_handler.reset()

                    # Retry processing with reduced batch size
                    continue

                # Ask user if they want to continue after error (for non-OOM errors)
                if batch_idx < total_batches:
                    should_continue = await self._ask_continue_after_error(batch_idx, str(e))
                    if not should_continue:
                        logger.info("User requested to stop processing after error")
                        break

        # Cleanup services
        if self._stt_service:
            try:
                self._stt_service.unload()
            except Exception as e:
                logger.warning(f"Failed to unload STT service: {e}")

        # Final memory cleanup
        self.memory_monitor.aggressive_cleanup()

        # Complete workflow
        self.checkpoint_manager.complete_workflow(success=True)
        self.progress_tracker.finish("Complete!")

        # Teardown shutdown handler
        self.shutdown_handler.teardown()

        end_time = datetime.now(timezone.utc)
        total_duration = (end_time - start_time).total_seconds()

        processing_result = ProcessingResult(
            total_files=total_files,
            total_batches=total_batches,
            total_successful=total_successful,
            total_failed=total_failed,
            total_skipped=total_skipped,
            total_retries=total_retries,
            total_duration_seconds=total_duration,
            batches=batch_results,
            start_time=start_time,
            end_time=end_time,
        )

        return processing_result

    async def _verify_and_prompt(
        self,
        batch_idx: int,
        result: BatchResult,
        total_batches: int,
    ) -> bool:
        """
        Verify system status after batch and prompt user to continue.

        Args:
            batch_idx: Current batch number
            result: Batch result
            total_batches: Total number of batches

        Returns:
            True if user wants to continue, False otherwise
        """
        # Get current system status
        memory_stats = self.memory_monitor.get_memory_stats()

        # Check if there are any concerns
        concerns = []
        if memory_stats["ram_percent"] > 75:
            concerns.append(f"High RAM usage: {memory_stats['ram_percent']:.1f}%")
        if result.failed > 0:
            concerns.append(f"{result.failed} files failed in this batch")
        if memory_stats["swap_percent"] > 25:
            concerns.append(f"Swap usage: {memory_stats['swap_percent']:.1f}%")

        # Print system status
        print("\n" + "=" * 70)
        print(f"SYSTEM VERIFICATION after Batch {batch_idx}/{total_batches}")
        print("=" * 70)
        print("\nMemory Status:")
        print(
            f"  RAM:  {memory_stats['ram_used_mb']:.1f}MB / {memory_stats['ram_total_mb']:.1f}MB ({memory_stats['ram_percent']:.1f}%)"
        )
        print(
            f"  Swap: {memory_stats['swap_used_mb']:.1f}MB / {memory_stats['swap_total_mb']:.1f}MB ({memory_stats['swap_percent']:.1f}%)"
        )
        print(f"\nBatch {batch_idx} Results:")
        print(f"  Successful: {result.successful}/{result.total_files}")
        print(f"  Failed: {result.failed}")
        print(f"  Peak Memory: {result.memory_peak_mb:.1f}MB")

        if concerns:
            print("\n⚠️  CONCERNS DETECTED:")
            for concern in concerns:
                print(f"  - {concern}")

        print(f"\nProgress: {batch_idx}/{total_batches} batches completed")
        print(
            f"Remaining: {total_batches - batch_idx} batches (~{(total_batches - batch_idx) * self.batch_size} files)"
        )
        print("=" * 70)

        if not self.prompt_between_batches:
            # Auto mode: continue if no major concerns
            if memory_stats["ram_percent"] > 90 or result.failed > result.total_files / 2:
                logger.warning("Critical threshold reached, pausing for user intervention")
                return self._ask_user_to_continue()
            return True

        # Prompt mode: always ask user
        return self._ask_user_to_continue()

    def _ask_user_to_continue(self) -> bool:
        """
        Ask user if they want to continue to next batch.

        Returns:
            True if user wants to continue, False otherwise
        """
        print("\n" + "=" * 70)
        while True:
            response = input("Continue to next batch? [Y/n/a]: ").strip().lower()
            if response in ("", "y", "yes"):
                print("Continuing to next batch...\n")
                return True
            elif response in ("n", "no"):
                print("Stopping processing as requested.")
                return False
            elif response in ("a", "auto"):
                print("Switching to auto-continue mode (no more prompts)")
                self.prompt_between_batches = False
                return True
            else:
                print("Invalid input. Please enter Y (yes), N (no), or A (auto)")
        print("=" * 70 + "\n")

    async def _ask_continue_after_error(self, batch_idx: int, error_msg: str) -> bool:
        """
        Ask user if they want to continue after an error.

        Args:
            batch_idx: Batch number that failed
            error_msg: Error message

        Returns:
            True if user wants to continue, False otherwise
        """
        print("\n" + "=" * 70)
        print(f"⚠️  ERROR in Batch {batch_idx}")
        print("=" * 70)
        print(f"\nError: {error_msg}")
        print("\nThe pipeline encountered an error but can continue to the next batch.")
        print("=" * 70)

        while True:
            response = input("\nContinue to next batch? [Y/n]: ").strip().lower()
            if response in ("", "y", "yes"):
                print("Continuing to next batch...\n")
                return True
            elif response in ("n", "no"):
                print("Stopping processing as requested.")
                return False
            else:
                print("Invalid input. Please enter Y (yes) or N (no)")

    def print_summary(self, result: ProcessingResult):
        """Print final processing summary."""
        print("\n" + "=" * 70)
        print("FORENSIC PIPELINE - SAFE BATCH PROCESSING SUMMARY")
        print("=" * 70)
        print(f"\nTotal Files:    {result.total_files}")
        print(f"Total Batches:  {result.total_batches}")
        print(f"Batch Size:     {self.batch_size} files (conservative)")
        print("\nResults:")
        print(
            f"  Successful: {result.total_successful} ({result.total_successful / result.total_files * 100:.1f}%)"
        )
        print(f"  Failed:     {result.total_failed}")
        print(f"  Skipped:    {result.total_skipped}")
        print(f"  Retries:    {result.total_retries}")
        print("\nDuration:")
        print(f"  Total:      {result.total_duration_seconds / 60:.1f} minutes")
        print(f"  Avg/batch:  {result.total_duration_seconds / result.total_batches:.1f}s")
        print(f"  Avg/file:   {result.total_duration_seconds / result.total_files:.1f}s")

        # Memory summary
        if result.batches:
            peak_mb = max(b.memory_peak_mb for b in result.batches)
            freed_mb = sum(b.memory_before_mb - b.memory_after_mb for b in result.batches)
            print("\nMemory:")
            print(f"  Peak usage: {peak_mb:.1f}MB")
            print(f"  Total freed: ~{freed_mb:.1f}MB across all batches")

        print("\n" + "=" * 70)


@click.command()
@click.option(
    "--batch-size",
    type=int,
    default=10,
    help="Batch size (default: 10 files per batch)",
)
@click.option(
    "--max-files",
    type=int,
    default=None,
    help="Maximum files to process (for testing)",
)
@click.option(
    "--audio-dir",
    type=click.Path(exists=True, path_type=Path),
    default=Path("ref/call"),
    help="Directory containing audio files",
)
@click.option(
    "--gpu/--no-gpu",
    default=True,
    help="Enable GPU acceleration",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=Path("logs/safe_batch_result.json"),
    help="Output JSON file for results",
)
@click.option(
    "--prompt/--auto",
    "prompt_between_batches",
    default=True,
    help="Prompt user between batches (default: True). Use --auto to disable prompts.",
)
@click.option(
    "--resume",
    is_flag=True,
    default=False,
    help="Resume processing from last checkpoint",
)
@click.option(
    "--checkpoint-dir",
    type=str,
    default="data/checkpoints",
    help="Directory for checkpoint storage (default: data/checkpoints)",
)
def main(
    batch_size: int,
    max_files: Optional[int],
    audio_dir: Path,
    gpu: bool,
    output: Path,
    prompt_between_batches: bool,
    resume: bool,
    checkpoint_dir: str,
):
    """
    Run memory-safe forensic batch processing with checkpoint-based recovery.

    Processing flow:
    - Process files in batches with checkpointing after each batch
    - Automatic resume from checkpoint on crash (with --resume)
    - Error classification with intelligent retry
    - Graceful shutdown with state preservation
    - System verification after each batch
    - User confirmation to continue (unless --auto is used)
    - Aggressive memory cleanup between batches

    Examples:
        # Normal run
        python scripts/run_safe_forensic_batch.py --auto --batch-size 10

        # Resume from checkpoint
        python scripts/run_safe_forensic_batch.py --auto --batch-size 10 --resume

        # Custom checkpoint directory
        python scripts/run_safe_forensic_batch.py --checkpoint-dir /tmp/checkpoints --auto

        # Test with 20 files
        python scripts/run_safe_forensic_batch.py --max-files 20 --auto
    """
    # Ensure logs directory exists
    Path("logs").mkdir(parents=True, exist_ok=True)

    processor = SafeBatchProcessor(
        audio_dir=audio_dir,
        batch_size=batch_size,
        max_retries=3,
        max_files=max_files,
        enable_gpu=gpu,
        prompt_between_batches=prompt_between_batches,
        checkpoint_dir=checkpoint_dir,
        resume=resume,
    )

    # Run all batches
    result = asyncio.run(processor.run_all_batches())

    # Print summary
    processor.print_summary(result)

    # Save results to JSON
    output.parent.mkdir(parents=True, exist_ok=True)
    import json

    with open(output, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output}")

    # Exit code based on success rate
    success_rate = result.total_successful / result.total_files if result.total_files > 0 else 0
    if success_rate < 0.5:
        sys.exit(1)  # More than 50% failed

    return 0


if __name__ == "__main__":
    main()
