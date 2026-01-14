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
        Perform aggressive memory cleanup.

        Returns:
            Memory freed in MB (estimated)
        """
        before_stats = MemoryMonitor.get_memory_stats()
        before_used = before_stats["ram_used_mb"]

        # Python garbage collection
        gc.collect()

        # PyTorch GPU memory cleanup (if available)
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            pass

        # Clear any caches
        try:
            from functools import lru_cache

            # Note: This doesn't clear user-defined lru_cache decorators
            # but helps with internal caches
            gc.collect()
        except Exception:
            pass

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
        batch_size: int = 10,
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

        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(checkpoint_dir=checkpoint_dir)

        # Error classifier
        self.error_classifier = ErrorClassifier()

        # Progress tracker
        self.progress_tracker: Optional[ProgressTracker] = None

        # Graceful shutdown handler
        self.shutdown_handler = GracefulShutdown(timeout=30.0)

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
                        # Re-initialize service with CPU
                        if hasattr(self._stt_service, "unload"):
                            self._stt_service.unload()
                        self._stt_service = None
                        self._initialize_services()

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
        logger.info(
            f"Memory before: RAM {memory_before['ram_used_mb']:.1f}MB / {memory_before['ram_total_mb']:.1f}MB ({memory_before['ram_percent']:.1f}%)"
        )

        successful = 0
        failed = 0
        skipped = 0
        total_retries = 0
        errors = []

        for i, audio_file in enumerate(files, 1):
            logger.info(f"\n[{i}/{len(files)}] Processing: {audio_file.name}")

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
            else:
                failed += 1
                error_msg = f"{audio_file.name}: {error}"
                errors.append(error_msg)
                logger.error(f"FAILED (after {attempts} attempts): {error_msg}")

            # Check peak memory
            memory_current = self.memory_monitor.get_memory_stats()
            peak_memory = max(peak_memory, memory_current["ram_used_mb"])

        # Aggressive cleanup after batch
        logger.info(f"\nBatch {batch_number} completed. Performing aggressive cleanup...")
        freed = self.memory_monitor.aggressive_cleanup()

        # Clear cache if exists
        if hasattr(self, "_stt_service") and hasattr(self._stt_service, "unload"):
            # Don't unload, just clear cache
            pass

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

        # Create workflow for checkpoint tracking
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

        # Setup graceful shutdown
        self.shutdown_handler.setup()

        @self.shutdown_handler.on_shutdown
        def save_on_shutdown():
            """Save checkpoint on shutdown."""
            logger.info("Shutdown signal received, saving checkpoint...")
            self.checkpoint_manager.state_store.update_workflow_state(
                workflow_id, status=WorkflowStatus.CRASHED
            )

        # Check for resume state
        start_batch = 1
        processed_files = set()
        failed_files = set()

        if self.resume:
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
                logger.info("No checkpoint found, starting fresh")
        else:
            logger.info("Resume mode disabled, starting fresh")

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

                # Ask user if they want to continue after error
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
