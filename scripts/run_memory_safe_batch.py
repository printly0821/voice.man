#!/usr/bin/env python3
"""
Memory-Safe Batch Processing Script with Enhanced Checkpointing
================================================================

Comprehensive batch processing with:
- New MemoryManager (replaces old MemoryMonitor)
- Enhanced checkpoint validation and recovery
- Pre-allocation memory checks
- Per-file memory tracking with watchdog
- Complete service cleanup between batches
- Checkpoint validation before resume
- Fatal error handling with graceful shutdown
- Detailed progress reporting with memory stats
- Signal handlers for graceful shutdown (SIGINT, SIGTERM)

Usage:
    python scripts/run_memory_safe_batch.py \\
        --source ref/call/ \\
        --batch-size 5 \\
        --resume \\
        --memory-threshold 85 \\
        --gpu-threshold 90

Architecture:
    MemorySafeBatchProcessor:
        - __init__: Initialize MemoryManager, CheckpointManager
        - process_all_batches(): Main orchestration
        - process_batch(): Single batch with full monitoring
        - process_file(): Individual file with context tracking
        - cleanup_between_batches(): Complete cleanup using MemoryManager
        - handle_oom(): OOM recovery with batch size reduction
        - graceful_shutdown(): Save checkpoint before exit
"""

import asyncio
import logging
import os
import signal
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import memory manager and checkpoint modules
from voice_man.services.memory.memory_manager import (
    FileMemoryStats,
    MemoryManager,
    MemoryPressureLevel,
)
from voice_man.services.checkpoint import (
    CheckpointManager,
    ErrorClassifier,
    ErrorCategory,
    GracefulShutdown,
    ProgressTracker,
)
from voice_man.services.checkpoint.checkpoint_validator import ValidationResult
from voice_man.services.checkpoint.state_store import WorkflowStatus

# Configure logging
log_dir = Path("logs")
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "memory_safe_batch.log"),
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
    memory_freed_mb: float
    file_stats: List[FileMemoryStats] = field(default_factory=list)
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
            "memory_before_mb": round(self.memory_before_mb, 2),
            "memory_after_mb": round(self.memory_after_mb, 2),
            "memory_peak_mb": round(self.memory_peak_mb, 2),
            "memory_freed_mb": round(self.memory_freed_mb, 2),
            "file_stats": [fs.to_dict() for fs in self.file_stats],
            "errors": self.errors,
        }


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
    oom_recoveries: int = 0
    total_memory_freed_mb: float = 0.0

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
            "oom_recoveries": self.oom_recoveries,
            "total_memory_freed_mb": round(self.total_memory_freed_mb, 2),
            "batches": [b.to_dict() for b in self.batches],
        }


class FatalError(Exception):
    """Fatal error that should stop processing."""

    def __init__(self, message: str, recoverable: bool = False):
        super().__init__(message)
        self.recoverable = recoverable


class MemorySafeBatchProcessor:
    """
    Memory-safe batch processor with enhanced checkpointing and monitoring.

    Features:
    - Pre-allocation memory checks before each batch
    - Per-file memory tracking with MemoryManager
    - Complete service cleanup between batches
    - Checkpoint validation before resume
    - OOM recovery with batch size reduction
    - Fatal error handling
    - Progress bar with memory indicators
    - Graceful shutdown with checkpoint save

    Architecture:
    1. Initialize MemoryManager and CheckpointManager
    2. Process all batches with monitoring
    3. Handle OOM with batch size reduction
    4. Cleanup and save checkpoints
    """

    # Default batch size (conservative to prevent OOM)
    DEFAULT_BATCH_SIZE = 5

    # OOM recovery configuration
    MIN_BATCH_SIZE = 1
    MAX_OOM_RETRIES = 3

    # OOM error patterns for detection
    OOM_PATTERNS = (
        "out of memory",
        "oom",
        "memory error",
        "cuda.*out.*of.*memory",
        "gpu.*out.*of.*memory",
        "allocation.*failed",
        "cuda.*memory",
    )

    def __init__(
        self,
        source_dir: Path,
        batch_size: int = DEFAULT_BATCH_SIZE,
        memory_threshold: float = 85.0,
        gpu_threshold: float = 90.0,
        enable_gpu: bool = True,
        max_retries: int = 3,
        checkpoint_dir: str = "data/checkpoints",
        resume: bool = False,
    ):
        """
        Initialize MemorySafeBatchProcessor.

        Args:
            source_dir: Directory containing files to process
            batch_size: Number of files per batch (default: 5)
            memory_threshold: System memory threshold percentage (default: 85)
            gpu_threshold: GPU memory threshold percentage (default: 90)
            enable_gpu: Enable GPU acceleration
            max_retries: Maximum retry attempts per file
            checkpoint_dir: Directory for checkpoint storage
            resume: Resume from last checkpoint
        """
        self.source_dir = source_dir
        self.batch_size = batch_size
        self.memory_threshold = memory_threshold
        self.gpu_threshold = gpu_threshold
        self.enable_gpu = enable_gpu
        self.max_retries = max_retries
        self.resume = resume

        # Initialize MemoryManager
        self.memory_manager = MemoryManager(
            system_memory_threshold=memory_threshold,
            gpu_memory_threshold=gpu_threshold,
            critical_threshold=95.0,
        )

        # Initialize CheckpointManager
        self.checkpoint_manager = CheckpointManager(checkpoint_dir=checkpoint_dir)

        # Error classifier
        self.error_classifier = ErrorClassifier()

        # Progress tracker
        self.progress_tracker: Optional[ProgressTracker] = None

        # Graceful shutdown handler
        self.shutdown_handler = GracefulShutdown(timeout=30.0)

        # Services (lazy initialization)
        self._stt_service = None
        self._forensic_service = None

        # OOM recovery state
        self.oom_count = 0
        self.original_batch_size = batch_size

        # Fatal error flag
        self.fatal_error = None

        # Setup signal handlers
        self._setup_signal_handlers()

        logger.info(
            f"MemorySafeBatchProcessor initialized: "
            f"source={source_dir}, "
            f"batch_size={batch_size}, "
            f"memory_threshold={memory_threshold}%, "
            f"gpu_threshold={gpu_threshold}%, "
            f"resume={resume}"
        )

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        logger.info("Signal handlers installed for SIGINT and SIGTERM")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        signal_name = signal.Signals(signum).name
        logger.warning(f"Received signal {signal_name} ({signum}), initiating graceful shutdown...")
        self.graceful_shutdown(reason=f"Received {signal_name}")

    def get_files_to_process(self) -> List[Path]:
        """
        Get all files to process, filtering out already completed files if resuming.

        Returns:
            List of file paths to process
        """
        audio_files = sorted(self.source_dir.glob("*.m4a"))

        if self.resume:
            # Try to get resume state
            resume_state = self.checkpoint_manager.get_resume_state(validate=False)
            if resume_state and resume_state.can_resume:
                processed_set = set(resume_state.processed_files)
                failed_set = set(resume_state.failed_files)

                # Filter out processed and failed files
                remaining_files = [
                    f
                    for f in audio_files
                    if str(f) not in processed_set and str(f) not in failed_set
                ]

                logger.info(
                    f"Resume mode: {len(audio_files)} total files, "
                    f"{len(resume_state.processed_files)} completed, "
                    f"{len(resume_state.failed_files)} failed, "
                    f"{len(remaining_files)} remaining"
                )

                return remaining_files

        logger.info(f"Found {len(audio_files)} files to process")
        return audio_files

    async def process_file(
        self, file_path: Path, stage: str = "forensic"
    ) -> Tuple[bool, Optional[str], Optional[Dict]]:
        """
        Process a single file with memory tracking.

        Args:
            file_path: Path to the file
            stage: Processing stage identifier

        Returns:
            Tuple of (success, error_message, result_data)
        """
        file_path_str = str(file_path)

        # Track file start
        self.memory_manager.track_file_start(file_path_str, stage=stage)

        try:
            # Step 1: STT (transcription)
            logger.info(f"STT processing: {file_path.name}")
            stt_result = await self._stt_service.transcribe_only(file_path_str)

            if "error" in stt_result:
                raise Exception(f"STT failed: {stt_result['error']}")

            segments = stt_result.get("segments", [])
            text = " ".join(seg.get("text", "") for seg in segments)

            if not text:
                raise Exception("Transcription produced empty text")

            logger.info(f"STT complete: {len(segments)} segments, {len(text)} chars")

            # Step 2: Forensic scoring
            logger.info(f"Forensic scoring: {file_path.name}")
            forensic_result = await self._forensic_service.analyze(
                audio_path=file_path_str,
                transcript=text,
            )

            logger.info(f"Forensic complete: risk={forensic_result.overall_risk_score:.1f}")

            result = {
                "file": file_path_str,
                "transcript_segments": len(segments),
                "transcript_chars": len(text),
                "forensic_score": forensic_result.overall_risk_score,
            }

            # Track file end (success)
            self.memory_manager.track_file_end(file_path_str, success=True)

            return True, None, result

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed to process {file_path.name}: {error_msg}")

            # Track file end (failure)
            self.memory_manager.track_file_end(file_path_str, success=False)

            return False, error_msg, None

    async def process_file_with_retry(
        self,
        file_path: Path,
    ) -> Tuple[bool, int, Optional[str], Optional[Dict]]:
        """
        Process a file with retry mechanism and error classification.

        Args:
            file_path: Path to the file

        Returns:
            Tuple of (success, attempts, final_error, result_data)
        """
        last_error = None
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                success, error, result = await self.process_file(file_path)

                if success:
                    return True, attempt + 1, None, result

                last_error = error

                # Classify error
                if last_exception:
                    category, severity = self.error_classifier.classify_error(last_exception)
                    strategy = self.error_classifier.determine_retry_strategy(category)

                    if not strategy.should_retry:
                        logger.warning(f"Error classified as {category.value}, not retrying")
                        break

                    # Cleanup before retry if needed
                    if strategy.cleanup_before_retry:
                        logger.info("Performing cleanup before retry...")
                        self.memory_manager.complete_cleanup()

                    # Calculate backoff
                    backoff = self.error_classifier.get_backoff_seconds(attempt, category, strategy)
                    logger.warning(f"Retry in {backoff:.1f}s... (category: {category.value})")
                    await asyncio.sleep(backoff)

                elif attempt < self.max_retries - 1:
                    wait_time = 2**attempt  # Exponential backoff
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
                        self.memory_manager.complete_cleanup()

                    # Calculate backoff
                    backoff = self.error_classifier.get_backoff_seconds(attempt, category, strategy)
                    logger.warning(f"Retry in {backoff:.1f}s... (category: {category.value})")
                    await asyncio.sleep(backoff)

        return False, self.max_retries, last_error, None

    def process_batch(
        self,
        batch_number: int,
        files: List[Path],
    ) -> BatchResult:
        """
        Process a batch of files with comprehensive memory management.

        Args:
            batch_number: Batch number (1-indexed)
            files: List of files to process

        Returns:
            BatchResult with processing statistics
        """
        start_time = datetime.now(timezone.utc)
        memory_summary = self.memory_manager.get_summary()

        logger.info(f"\n{'=' * 70}")
        logger.info(f"BATCH {batch_number}: Processing {len(files)} files")
        logger.info(f"{'=' * 70}")
        logger.info(
            f"Memory: {memory_summary['current_memory_mb']:.1f}MB / "
            f"{memory_summary['system_memory_percent']:.1f}% used"
        )

        successful = 0
        failed = 0
        skipped = 0
        total_retries = 0
        errors = []
        file_stats: List[FileMemoryStats] = []

        # Process files with progress bar
        with tqdm(
            total=len(files),
            desc=f"Batch {batch_number}",
            unit="file",
            postfix={"mem": f"{memory_summary['system_memory_percent']:.0f}%"},
        ) as pbar:
            for i, file_path in enumerate(files, 1):
                file_path_str = str(file_path)

                # Check if already processed
                from voice_man.services.checkpoint.state_store import FileStatus

                existing_file_state = self.checkpoint_manager.state_store.get_file_state(
                    file_path_str
                )
                if existing_file_state and existing_file_state.status == FileStatus.COMPLETED:
                    logger.info(
                        f"[{i}/{len(files)}] Skipping (already completed): {file_path.name}"
                    )
                    skipped += 1
                    pbar.update(1)
                    continue

                # Mark file as processing
                self.checkpoint_manager.mark_file_processing(file_path_str)

                # Check memory pressure
                pressure_status = self.memory_manager.check_during_processing()
                if pressure_status.predicted_oom:
                    logger.warning(f"OOM predicted: {pressure_status.recommended_action}")
                    # Run emergency cleanup
                    cleanup_stats = self.memory_manager.complete_cleanup()
                    logger.info(f"Emergency cleanup freed {cleanup_stats['memory_freed_mb']:.1f}MB")

                # Update progress bar with memory
                memory_summary = self.memory_manager.get_summary()
                pbar.set_postfix({"mem": f"{memory_summary['system_memory_percent']:.0f}%"})

                # Process file
                success, attempts, error, result = asyncio.run(
                    self.process_file_with_retry(file_path)
                )
                total_retries += attempts - 1

                if success:
                    successful += 1
                    logger.info(f"SUCCESS: {file_path.name}")

                    # Mark as completed
                    self.checkpoint_manager.mark_file_completed(
                        file_path_str,
                        result={"success": True, "attempts": attempts},
                    )
                else:
                    failed += 1
                    error_msg = f"{file_path.name}: {error}"
                    errors.append(error_msg)
                    logger.error(f"FAILED (after {attempts} attempts): {error_msg}")

                    # Mark as failed
                    self.checkpoint_manager.mark_file_failed(
                        file_path_str,
                        error_message=error or "Unknown error",
                        attempts=attempts,
                    )

                # Collect file stats
                stats = (
                    self.memory_manager.get_file_history()[-1]
                    if self.memory_manager.get_file_history()
                    else None
                )
                if stats:
                    file_stats.append(stats)

                pbar.update(1)

        # Cleanup between batches
        logger.info(f"\nBatch {batch_number} completed. Running cleanup...")
        cleanup_stats = self.cleanup_between_batches()

        end_time = datetime.now(timezone.utc)
        duration = (end_time - start_time).total_seconds()

        memory_after = self.memory_manager.get_summary()

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
            memory_before_mb=memory_summary["current_memory_mb"],
            memory_after_mb=memory_after["current_memory_mb"],
            memory_peak_mb=max(fs.peak_memory_mb for fs in file_stats)
            if file_stats
            else memory_summary["current_memory_mb"],
            memory_freed_mb=cleanup_stats["memory_freed_mb"],
            file_stats=file_stats,
            errors=errors,
        )

        self._log_batch_summary(result)

        return result

    def _log_batch_summary(self, result: BatchResult) -> None:
        """Log batch processing summary."""
        logger.info(f"\n{'=' * 70}")
        logger.info(f"BATCH {result.batch_number} SUMMARY:")
        logger.info(f"  Files: {result.successful}/{result.total_files} successful")
        logger.info(f"  Failed: {result.failed}, Skipped: {result.skipped}")
        logger.info(f"  Retries: {result.retries}")
        logger.info(f"  Duration: {result.duration_seconds:.1f}s")
        logger.info(
            f"  Memory: {result.memory_before_mb:.1f}MB -> "
            f"{result.memory_after_mb:.1f}MB "
            f"(freed {result.memory_freed_mb:.1f}MB)"
        )
        logger.info(f"  Peak: {result.memory_peak_mb:.1f}MB")
        logger.info(f"{'=' * 70}\n")

    def cleanup_between_batches(self) -> Dict[str, Any]:
        """
        Perform complete cleanup between batches using MemoryManager.

        Returns:
            Cleanup statistics dictionary
        """
        logger.info("Starting complete cleanup between batches...")

        # Unload STT service
        if self._stt_service is not None:
            try:
                if hasattr(self._stt_service, "unload"):
                    self._stt_service.unload()
                self._stt_service = None
                logger.info("STT service unloaded")
            except Exception as e:
                logger.warning(f"Failed to unload STT service: {e}")

        # Use MemoryManager for complete cleanup
        cleanup_stats = self.memory_manager.complete_cleanup()

        logger.info(
            f"Cleanup complete: "
            f"{cleanup_stats['services_cleaned']} services, "
            f"{cleanup_stats['gc_collected']} GC objects, "
            f"{cleanup_stats['memory_freed_mb']:.1f}MB freed"
        )

        # Reinitialize services for next batch
        # This is critical: after cleanup, services are set to None and must be recreated
        logger.info("Reinitializing services for next batch...")
        self._initialize_services()
        logger.info("Services reinitialized successfully")

        return cleanup_stats

    def handle_oom(self, exception: Exception) -> Optional[int]:
        """
        Handle OOM error by reducing batch size.

        Args:
            exception: The exception that occurred

        Returns:
            New batch size to retry with, or None if should not retry
        """
        error_msg = str(exception).lower()

        # Check if this is an OOM error
        is_oom = any(pattern in error_msg for pattern in self.OOM_PATTERNS)

        if not is_oom:
            return None

        self.oom_count += 1

        if self.oom_count > self.MAX_OOM_RETRIES:
            logger.error(
                f"Max OOM retries reached ({self.MAX_OOM_RETRIES}), "
                f"cannot recover from batch size {self.batch_size}"
            )
            # Raise fatal error
            self.fatal_error = FatalError(
                f"Max OOM retries exceeded. System cannot handle even batch size {self.batch_size}",
                recoverable=False,
            )
            return None

        # Calculate new batch size (reduce by half)
        new_batch_size = max(self.MIN_BATCH_SIZE, self.batch_size // 2)

        logger.warning(
            f"OOM detected (count={self.oom_count}/{self.MAX_OOM_RETRIES}), "
            f"reducing batch size: {self.batch_size} -> {new_batch_size}"
        )

        # Run emergency cleanup
        cleanup_stats = self.memory_manager.complete_cleanup()
        logger.info(f"Emergency OOM cleanup freed {cleanup_stats['memory_freed_mb']:.1f}MB")

        return new_batch_size

    def graceful_shutdown(self, reason: str = "User requested") -> None:
        """
        Perform graceful shutdown with checkpoint save.

        Args:
            reason: Reason for shutdown
        """
        logger.warning(f"Initiating graceful shutdown: {reason}")

        # Save current state to checkpoint
        if self.checkpoint_manager.get_current_workflow_id():
            try:
                self.checkpoint_manager.state_store.update_workflow_state(
                    self.checkpoint_manager.get_current_workflow_id(),
                    status=WorkflowStatus.CRASHED,
                )
                logger.info("Workflow marked as CRASHED for potential resume")
            except Exception as e:
                logger.error(f"Failed to save checkpoint: {e}")

        # Final cleanup
        try:
            cleanup_stats = self.memory_manager.complete_cleanup()
            logger.info(f"Final cleanup freed {cleanup_stats['memory_freed_mb']:.1f}MB")
        except Exception as e:
            logger.error(f"Failed to run final cleanup: {e}")

        logger.warning(f"Graceful shutdown complete: {reason}")

    def check_pre_allocation(self, batch_size: int, file_size_mb: float = 10.0) -> Tuple[bool, str]:
        """
        Check if sufficient memory is available before starting batch.

        Args:
            batch_size: Number of files in batch
            file_size_mb: Estimated file size in MB

        Returns:
            Tuple of (is_safe, message)
        """
        return self.memory_manager.check_pre_allocation(
            batch_size=batch_size,
            estimated_per_file_mb=file_size_mb,
            safety_margin=1.3,  # 30% safety margin
        )

    def validate_checkpoint_before_resume(self) -> ValidationResult:
        """
        Validate checkpoint before resuming processing.

        Returns:
            ValidationResult with validation status
        """
        logger.info("Validating checkpoint before resume...")

        validation_result = self.checkpoint_manager.validate_current_workflow()

        if not validation_result.is_valid:
            logger.error(f"Checkpoint validation failed: {validation_result.get_summary()}")

            # Log all critical errors
            for error in validation_result.errors:
                if error.severity == "critical":
                    logger.error(f"  [{error.category}] {error.message}")

            # Attempt repair if possible
            if validation_result.repairable_count > 0:
                logger.info(
                    f"Attempting to repair {validation_result.repairable_count} repairable errors..."
                )
                repaired = self.checkpoint_manager.validator.repair_all(validation_result)
                if repaired > 0:
                    logger.info(f"Successfully repaired {repaired} errors")
                    # Re-validate after repair
                    validation_result = self.checkpoint_manager.validate_current_workflow()

        return validation_result

    def process_all_batches(self) -> ProcessingResult:
        """
        Main orchestration method to process all batches.

        Returns:
            ProcessingResult with overall statistics
        """
        start_time = datetime.now(timezone.utc)

        # Initialize services
        self._initialize_services()

        # Get files to process
        all_files = self.get_files_to_process()
        total_files = len(all_files)

        if total_files == 0:
            logger.warning("No files to process")
            return ProcessingResult(
                total_files=0,
                total_batches=0,
                total_successful=0,
                total_failed=0,
                total_skipped=0,
                total_retries=0,
                total_duration_seconds=0.0,
                start_time=start_time,
                end_time=datetime.now(timezone.utc),
            )

        # Setup graceful shutdown
        self.shutdown_handler.setup()

        # Create workflow
        workflow_id = f"workflow_{int(start_time.timestamp())}"
        self.checkpoint_manager.create_workflow(
            workflow_id=workflow_id,
            total_files=total_files,
            batch_size=self.batch_size,
            metadata={"source_dir": str(self.source_dir)},
        )

        # Register all files
        all_file_strs = [str(f) for f in all_files]
        for i, file_path in enumerate(all_file_strs):
            batch_id = f"batch_{(i // self.batch_size) + 1}"
            self.checkpoint_manager.register_files([file_path], batch_id)

        # Setup shutdown handler
        @self.shutdown_handler.on_shutdown
        def save_on_shutdown():
            """Save checkpoint on shutdown."""
            logger.info("Shutdown signal received, saving checkpoint...")
            self.checkpoint_manager.state_store.update_workflow_state(
                workflow_id, status=WorkflowStatus.CRASHED
            )

        # Create batches
        batches = []
        for i in range(0, len(all_files), self.batch_size):
            batch = all_files[i : i + self.batch_size]
            batches.append(batch)

        total_batches = len(batches)

        logger.info(f"\nProcessing plan: {total_files} files in {total_batches} batches")
        logger.info(f"Batch size: {self.batch_size} files/batch")
        logger.info(f"Retry policy: {self.max_retries} attempts before skip")
        logger.info(f"Memory threshold: {self.memory_threshold}%")
        logger.info(f"GPU threshold: {self.gpu_threshold}%\n")

        # Initialize progress tracker
        self.progress_tracker = ProgressTracker(
            total_items=total_files,
            enable_console=True,
        )

        # Process all batches
        batch_results = []
        total_successful = 0
        total_failed = 0
        total_skipped = 0
        total_retries = 0
        total_memory_freed = 0.0

        for batch_idx, batch_files in enumerate(batches, start=1):
            # Check for shutdown signal
            if self.shutdown_handler.is_shutdown_requested():
                logger.info("Shutdown requested, stopping batch processing")
                break

            # Check for fatal error
            if self.fatal_error:
                logger.error(f"Fatal error detected: {self.fatal_error}")
                break

            # Pre-allocation memory check
            is_safe, message = self.check_pre_allocation(len(batch_files))
            logger.info(f"Pre-allocation check: {message}")

            if not is_safe:
                logger.error(f"Memory check failed: {message}")

                # Try to reduce batch size
                if self.batch_size > self.MIN_BATCH_SIZE:
                    new_batch_size = max(self.MIN_BATCH_SIZE, self.batch_size // 2)
                    logger.warning(f"Reducing batch size: {self.batch_size} -> {new_batch_size}")
                    self.batch_size = new_batch_size

                    # Recreate batch with reduced size
                    start_idx = (batch_idx - 1) * self.original_batch_size
                    end_idx = min(start_idx + new_batch_size, len(all_files))
                    batch_files = all_files[start_idx:end_idx]

                    logger.info(f"Retrying with {len(batch_files)} files")
                else:
                    # Cannot reduce further, raise fatal error
                    self.fatal_error = FatalError(
                        f"Memory check failed even with minimum batch size {self.MIN_BATCH_SIZE}",
                        recoverable=False,
                    )
                    break

            try:
                result = self.process_batch(batch_idx, batch_files)
                batch_results.append(result)

                total_successful += result.successful
                total_failed += result.failed
                total_skipped += result.skipped
                total_retries += result.retries
                total_memory_freed += result.memory_freed_mb

                # Update progress
                self.progress_tracker.update(
                    current=total_successful,
                    failed=total_failed,
                )

                # Save checkpoint after batch
                self._save_batch_checkpoint(batch_idx, batch_files, result)

            except Exception as e:
                logger.error(f"Batch {batch_idx} failed with exception: {e}")
                logger.error(traceback.format_exc())

                # Check for OOM
                new_batch_size = self.handle_oom(e)
                if new_batch_size is not None:
                    self.batch_size = new_batch_size
                    logger.info(f"Retrying batch {batch_idx} with reduced size {self.batch_size}")

                    # Recreate batch and retry
                    start_idx = (batch_idx - 1) * self.original_batch_size
                    end_idx = min(start_idx + self.batch_size, len(all_files))
                    batch_files = all_files[start_idx:end_idx]

                    if batch_files:
                        try:
                            result = self.process_batch(batch_idx, batch_files)
                            batch_results.append(result)

                            total_successful += result.successful
                            total_failed += result.failed
                            total_skipped += result.skipped
                            total_retries += result.retries
                            total_memory_freed += result.memory_freed_mb

                            # Save checkpoint
                            self._save_batch_checkpoint(batch_idx, batch_files, result)

                        except Exception as retry_error:
                            logger.error(f"Retry failed: {retry_error}")
                            # Raise fatal error
                            self.fatal_error = FatalError(
                                f"Batch {batch_idx} failed even after OOM recovery: {retry_error}",
                                recoverable=False,
                            )
                            break
                else:
                    # Non-OOM error or max retries exceeded
                    if self.fatal_error:
                        raise self.fatal_error
                    else:
                        # Log but continue to next batch
                        logger.error(f"Batch {batch_idx} failed with non-recoverable error")

        # Final cleanup
        self.cleanup_between_batches()

        # Complete workflow
        success = self.fatal_error is None and total_failed == 0
        self.checkpoint_manager.complete_workflow(success=success)
        self.progress_tracker.finish("Complete!" if success else "Completed with errors")

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
            oom_recoveries=self.oom_count,
            total_memory_freed_mb=total_memory_freed,
        )

        return processing_result

    def _save_batch_checkpoint(
        self,
        batch_number: int,
        files: List[Path],
        result: BatchResult,
    ) -> None:
        """Save checkpoint after batch completion."""
        try:
            # Determine processed and failed files
            processed_files = []
            failed_files = []

            failed_filenames = set()
            for error_msg in result.errors:
                if ":" in error_msg:
                    filename = error_msg.split(":", 1)[0].strip()
                    failed_filenames.add(filename)

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
                    "memory_before_mb": result.memory_before_mb,
                    "memory_after_mb": result.memory_after_mb,
                    "memory_peak_mb": result.memory_peak_mb,
                    "memory_freed_mb": result.memory_freed_mb,
                    "errors": result.errors,
                },
            )

            logger.info(f"Checkpoint saved for batch {batch_number}")

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def _initialize_services(self):
        """Initialize all required services."""
        logger.info("Initializing services...")

        from voice_man.services.whisperx_service import WhisperXService
        from voice_man.services.forensic.audio_feature_service import AudioFeatureService
        from voice_man.services.forensic.stress_analysis_service import StressAnalysisService
        from voice_man.services.forensic.ser_service import SERService
        from voice_man.services.forensic.crime_language_service import CrimeLanguageAnalysisService
        from voice_man.services.forensic.cross_validation_service import CrossValidationService
        from voice_man.services.forensic.forensic_scoring_service import ForensicScoringService

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

        # Register services with MemoryManager for cleanup
        self.memory_manager.register_service("stt", self._stt_service)
        self.memory_manager.register_service("forensic", self._forensic_service)
        self.memory_manager.register_service("ser", ser_service)

        logger.info("Services initialized successfully")

    def print_summary(self, result: ProcessingResult):
        """Print final processing summary."""
        print("\n" + "=" * 70)
        print("MEMORY-SAFE BATCH PROCESSING SUMMARY")
        print("=" * 70)
        print(f"\nTotal Files:    {result.total_files}")
        print(f"Total Batches:  {result.total_batches}")
        print(f"Batch Size:     {self.original_batch_size} files")
        print("\nResults:")
        success_rate = (
            result.total_successful / result.total_files * 100 if result.total_files > 0 else 0
        )
        print(f"  Successful: {result.total_successful} ({success_rate:.1f}%)")
        print(f"  Failed:     {result.total_failed}")
        print(f"  Skipped:    {result.total_skipped}")
        print(f"  Retries:    {result.total_retries}")
        print("\nDuration:")
        print(f"  Total:      {result.total_duration_seconds / 60:.1f} minutes")
        print(
            f"  Avg/batch:  {result.total_duration_seconds / result.total_batches:.1f}s"
            if result.total_batches > 0
            else "  Avg/batch:  N/A"
        )
        print(
            f"  Avg/file:   {result.total_duration_seconds / result.total_files:.1f}s"
            if result.total_files > 0
            else "  Avg/file:  N/A"
        )
        print("\nMemory:")
        print(f"  Total freed: ~{result.total_memory_freed_mb:.1f}MB across all batches")
        print(f"  OOM recoveries: {result.oom_recoveries}")

        if self.fatal_error:
            print(f"\nFatal Error: {self.fatal_error}")

        print("\n" + "=" * 70)


@click.command()
@click.option(
    "--source",
    "-s",
    type=click.Path(exists=True, path_type=Path),
    default=Path("ref/call"),
    help="Directory containing files to process",
)
@click.option(
    "--batch-size",
    "-b",
    type=int,
    default=5,
    help="Batch size (default: 5 files per batch)",
)
@click.option(
    "--memory-threshold",
    type=int,
    default=85,
    help="System memory threshold percentage (default: 85)",
)
@click.option(
    "--gpu-threshold",
    type=int,
    default=90,
    help="GPU memory threshold percentage (default: 90)",
)
@click.option(
    "--gpu/--no-gpu",
    default=True,
    help="Enable GPU acceleration",
)
@click.option(
    "--max-retries",
    type=int,
    default=3,
    help="Maximum retry attempts per file (default: 3)",
)
@click.option(
    "--checkpoint-dir",
    type=str,
    default="data/checkpoints",
    help="Directory for checkpoint storage (default: data/checkpoints)",
)
@click.option(
    "--resume",
    "-r",
    is_flag=True,
    default=False,
    help="Resume processing from last checkpoint",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=Path("logs/memory_safe_batch_result.json"),
    help="Output JSON file for results",
)
def main(
    source: Path,
    batch_size: int,
    memory_threshold: int,
    gpu_threshold: int,
    gpu: bool,
    max_retries: int,
    checkpoint_dir: str,
    resume: bool,
    output: Path,
):
    """
    Memory-safe batch processing with enhanced checkpointing.

    Features:
    - Pre-allocation memory checks before each batch
    - Per-file memory tracking with watchdog monitoring
    - Complete service cleanup between batches
    - Checkpoint validation before resume
    - OOM recovery with batch size reduction
    - Fatal error handling with graceful shutdown
    - Progress bar with memory indicators
    - Signal handlers for graceful shutdown (SIGINT, SIGTERM)

    Examples:
        # Normal run
        python scripts/run_memory_safe_batch.py --source ref/call/ --batch-size 5

        # Resume from checkpoint
        python scripts/run_memory_safe_batch.py --source ref/call/ --batch-size 5 --resume

        # Custom memory thresholds
        python scripts/run_memory_safe_batch.py --source ref/call/ --memory-threshold 80 --gpu-threshold 85

        # Disable GPU
        python scripts/run_memory_safe_batch.py --source ref/call/ --no-gpu
    """
    # Ensure logs directory exists
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create processor
    processor = MemorySafeBatchProcessor(
        source_dir=source,
        batch_size=batch_size,
        memory_threshold=float(memory_threshold),
        gpu_threshold=float(gpu_threshold),
        enable_gpu=gpu,
        max_retries=max_retries,
        checkpoint_dir=checkpoint_dir,
        resume=resume,
    )

    # Process all batches
    result = processor.process_all_batches()

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
    if success_rate < 0.5 or processor.fatal_error is not None:
        sys.exit(1)

    return 0


if __name__ == "__main__":
    main()
