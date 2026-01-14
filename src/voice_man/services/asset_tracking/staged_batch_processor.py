"""
Staged Batch Processing System for Forensic Pipeline
SPEC-ASSET-002: Progressive Batch Processing with Checkpoints

Implements staged processing:
- Stage 1: Pilot batch (10 files) - validation and calibration
- Stage 2: Medium batch (50 files) - scale testing
- Stage 3: Full batch (183 files) - complete processing

Each stage can resume from previous stage checkpoints.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from .uuidv7_manager import AssetRegistry, AssetStatus, AssetType, get_asset_registry

logger = logging.getLogger(__name__)


class ProcessingStage(str, Enum):
    """Processing stages in the pipeline"""

    STT = "stt"  # Speech-to-Text
    ALIGNMENT = "alignment"  # Timestamp alignment
    DIARIZATION = "diarization"  # Speaker diarization
    SER = "ser"  # Speech Emotion Recognition
    FORENSIC_SCORING = "forensic_scoring"  # Forensic analysis
    GASLIGHTING = "gaslighting_analysis"  # Gaslighting detection
    CROSS_VALIDATION = "cross_validation"  # Cross-validation
    REPORTING = "reporting"  # Report generation


@dataclass
class BatchStageConfig:
    """Configuration for a processing stage"""

    stage: ProcessingStage
    batch_size: int
    gpu_memory_mb: int
    timeout_seconds: int = 3600
    retry_attempts: int = 3
    parallel_workers: int = 1


@dataclass
class StagedBatchConfig:
    """Configuration for staged batch processing"""

    # Stage 1: Pilot batch
    pilot_batch_size: int = 10
    pilot_stages: List[ProcessingStage] = field(
        default_factory=lambda: [
            ProcessingStage.STT,
            ProcessingStage.ALIGNMENT,
            ProcessingStage.SER,
            ProcessingStage.FORENSIC_SCORING,
        ]
    )

    # Stage 2: Medium batch
    medium_batch_size: int = 50
    medium_stages: List[ProcessingStage] = field(
        default_factory=lambda: [
            ProcessingStage.STT,
            ProcessingStage.ALIGNMENT,
            ProcessingStage.DIARIZATION,
            ProcessingStage.SER,
            ProcessingStage.FORENSIC_SCORING,
            ProcessingStage.GASLIGHTING,
        ]
    )

    # Stage 3: Full batch
    full_batch_size: int = 183  # All remaining files
    full_stages: List[ProcessingStage] = field(
        default_factory=lambda: [
            ProcessingStage.STT,
            ProcessingStage.ALIGNMENT,
            ProcessingStage.DIARIZATION,
            ProcessingStage.SER,
            ProcessingStage.FORENSIC_SCORING,
            ProcessingStage.GASLIGHTING,
            ProcessingStage.CROSS_VALIDATION,
            ProcessingStage.REPORTING,
        ]
    )

    # Checkpoint configuration
    checkpoint_interval: int = 5  # Save checkpoint every N files
    checkpoint_dir: str = "data/checkpoints"

    # Progress tracking
    progress_file: str = "data/batch_progress.json"


@dataclass
class StageResult:
    """Result of processing a stage"""

    stage: ProcessingStage
    batch_id: str
    total_files: int
    processed_files: int
    failed_files: int
    skipped_files: int
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    error_details: Dict[str, str] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.total_files == 0:
            return 0.0
        return self.processed_files / self.total_files


@dataclass
class BatchCheckpoint:
    """Checkpoint data for batch recovery"""

    batch_id: str
    stage: ProcessingStage
    processed_files: List[str]  # Asset IDs
    failed_files: List[str]  # Asset IDs with errors
    pending_files: List[str]  # Asset IDs yet to process
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "batch_id": self.batch_id,
            "stage": self.stage.value,
            "processed_files": self.processed_files,
            "failed_files": self.failed_files,
            "pending_files": self.pending_files,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BatchCheckpoint":
        """Create from dictionary"""
        return cls(
            batch_id=data["batch_id"],
            stage=ProcessingStage(data["stage"]),
            processed_files=data["processed_files"],
            failed_files=data["failed_files"],
            pending_files=data["pending_files"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {}),
        )


class StagedBatchProcessor:
    """
    Staged batch processor with checkpoint-based recovery

    Processes files in stages (10 → 50 → 183) with full recovery capability.
    """

    def __init__(
        self,
        config: Optional[StagedBatchConfig] = None,
        asset_registry: Optional[AssetRegistry] = None,
    ):
        """
        Initialize staged batch processor

        Args:
            config: Staged batch configuration
            asset_registry: Asset registry for tracking
        """
        self.config = config or StagedBatchConfig()
        self.asset_registry = asset_registry or get_asset_registry()

        # Batch tracking
        self._current_batch_id: Optional[str] = None
        self._current_stage: Optional[ProcessingStage] = None
        self._checkpoints: Dict[str, BatchCheckpoint] = {}

        # Progress tracking
        self._progress_data: Dict[str, Any] = {
            "stages_completed": [],
            "current_batch": None,
            "total_processed": 0,
            "total_failed": 0,
        }

        # Ensure checkpoint directory exists
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # Load existing progress
        self._load_progress()

    def create_batch_id(self, stage_name: str) -> str:
        """
        Create a unique batch ID

        Args:
            stage_name: Name of the stage (e.g., "pilot", "medium", "full")

        Returns:
            Unique batch ID
        """
        from .uuidv7_manager import UUIDv7Generator

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        uuid_short = UUIDv7Generator.generate_str()[:8]
        return f"batch_{stage_name}_{timestamp}_{uuid_short}"

    async def process_stage(
        self,
        stage: ProcessingStage,
        file_paths: List[str],
        batch_id: str,
        processor_fn: Callable[[List[str]], Awaitable[StageResult]],
        resume_from_checkpoint: bool = False,
    ) -> StageResult:
        """
        Process a single stage with checkpointing

        Args:
            stage: Stage to process
            file_paths: List of file paths to process
            batch_id: Batch ID for tracking
            processor_fn: Async function that processes the files
            resume_from_checkpoint: Whether to resume from existing checkpoint

        Returns:
            StageResult with processing statistics
        """
        self._current_stage = stage
        self._current_batch_id = batch_id

        # Register all files as assets
        asset_ids = []
        for file_path in file_paths:
            asset = self.asset_registry.register_asset(
                asset_type=AssetType.AUDIO_FILE,
                original_path=file_path,
                batch_id=batch_id,
            )
            asset_ids.append(asset.asset_id)

        # Check for existing checkpoint
        checkpoint = None
        if resume_from_checkpoint:
            checkpoint = self._load_checkpoint(batch_id, stage)

        if checkpoint:
            logger.info(
                f"Resuming from checkpoint: {len(checkpoint.processed_files)} files already processed"
            )
            pending_ids = checkpoint.pending_files
        else:
            pending_ids = asset_ids

        # Process files
        start_time = datetime.now(timezone.utc)
        processed = []
        failed = []
        skipped = []

        # Process in sub-batches
        sub_batch_size = self._get_sub_batch_size(stage)

        for i in range(0, len(pending_ids), sub_batch_size):
            sub_batch_ids = pending_ids[i : i + sub_batch_size]
            sub_batch_paths = [
                self.asset_registry.get_asset(asset_id).original_path
                for asset_id in sub_batch_ids
                if self.asset_registry.get_asset(asset_id)
            ]

            try:
                # Process sub-batch
                _ = await processor_fn(sub_batch_paths)

                # Update asset statuses
                for asset_id in sub_batch_ids:
                    self.asset_registry.update_asset(asset_id, status=AssetStatus.COMPLETED)

                processed.extend(sub_batch_ids)

                # Save checkpoint periodically
                if (i // sub_batch_size + 1) % self.config.checkpoint_interval == 0:
                    self._save_checkpoint(
                        batch_id,
                        stage,
                        processed,
                        failed,
                        [aid for aid in pending_ids if aid not in processed and aid not in failed],
                    )

            except Exception as e:
                logger.error(f"Sub-batch processing failed: {e}")
                failed.extend(sub_batch_ids)

                # Update asset statuses
                for asset_id in sub_batch_ids:
                    self.asset_registry.update_asset(
                        asset_id, status=AssetStatus.FAILED, error_message=str(e)
                    )

        end_time = datetime.now(timezone.utc)
        duration = (end_time - start_time).total_seconds()

        # Create final checkpoint
        self._save_checkpoint(
            batch_id,
            stage,
            processed,
            failed,
            [aid for aid in asset_ids if aid not in processed and aid not in failed],
        )

        result = StageResult(
            stage=stage,
            batch_id=batch_id,
            total_files=len(asset_ids),
            processed_files=len(processed),
            failed_files=len(failed),
            skipped_files=len(skipped),
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
        )

        # Update progress
        self._update_progress(stage, batch_id, result)

        return result

    def _get_sub_batch_size(self, stage: ProcessingStage) -> int:
        """Get optimal sub-batch size for a stage"""
        stage_sizes = {
            ProcessingStage.STT: 16,
            ProcessingStage.ALIGNMENT: 32,
            ProcessingStage.DIARIZATION: 8,
            ProcessingStage.SER: 8,
            ProcessingStage.FORENSIC_SCORING: 64,
            ProcessingStage.GASLIGHTING: 32,
            ProcessingStage.CROSS_VALIDATION: 32,
            ProcessingStage.REPORTING: 16,
        }
        return stage_sizes.get(stage, 16)

    def _save_checkpoint(
        self,
        batch_id: str,
        stage: ProcessingStage,
        processed_files: List[str],
        failed_files: List[str],
        pending_files: List[str],
    ):
        """Save checkpoint data"""
        checkpoint = BatchCheckpoint(
            batch_id=batch_id,
            stage=stage,
            processed_files=processed_files,
            failed_files=failed_files,
            pending_files=pending_files,
            timestamp=datetime.now(timezone.utc),
        )

        checkpoint_path = (
            Path(self.config.checkpoint_dir) / f"{batch_id}_{stage.value}_checkpoint.json"
        )

        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(checkpoint.to_dict(), f, indent=2, ensure_ascii=False)

        self._checkpoints[f"{batch_id}_{stage.value}"] = checkpoint
        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def _load_checkpoint(self, batch_id: str, stage: ProcessingStage) -> Optional[BatchCheckpoint]:
        """Load checkpoint data"""
        checkpoint_path = (
            Path(self.config.checkpoint_dir) / f"{batch_id}_{stage.value}_checkpoint.json"
        )

        if not checkpoint_path.exists():
            return None

        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            checkpoint = BatchCheckpoint.from_dict(data)
            self._checkpoints[f"{batch_id}_{stage.value}"] = checkpoint
            return checkpoint

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None

    def _update_progress(self, stage: ProcessingStage, batch_id: str, result: StageResult):
        """Update progress tracking"""
        if stage.value not in self._progress_data["stages_completed"]:
            self._progress_data["stages_completed"].append(stage.value)

        self._progress_data["current_batch"] = batch_id
        self._progress_data["total_processed"] += result.processed_files
        self._progress_data["total_failed"] += result.failed_files

        self._save_progress()

    def _save_progress(self):
        """Save progress to file"""
        progress_path = Path(self.config.progress_file)
        progress_path.parent.mkdir(parents=True, exist_ok=True)

        with open(progress_path, "w", encoding="utf-8") as f:
            json.dump(self._progress_data, f, indent=2, ensure_ascii=False)

    def _load_progress(self):
        """Load progress from file"""
        progress_path = Path(self.config.progress_file)

        if not progress_path.exists():
            return

        try:
            with open(progress_path, "r", encoding="utf-8") as f:
                self._progress_data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load progress: {e}")

    def get_progress_summary(self) -> Dict[str, Any]:
        """Get summary of batch processing progress"""
        return {
            **self._progress_data,
            "stages_total": len(ProcessingStage),
            "stages_completed_count": len(self._progress_data.get("stages_completed", [])),
            "completion_percentage": (
                len(self._progress_data.get("stages_completed", [])) / len(ProcessingStage) * 100
            ),
        }


class StagePipeline:
    """
    Complete staged pipeline (10 → 50 → 183)

    Coordinates all three stages with proper checkpointing and recovery.
    """

    def __init__(
        self,
        config: Optional[StagedBatchConfig] = None,
        asset_registry: Optional[AssetRegistry] = None,
    ):
        """
        Initialize stage pipeline

        Args:
            config: Staged batch configuration
            asset_registry: Asset registry for tracking
        """
        self.config = config or StagedBatchConfig()
        self.asset_registry = asset_registry or get_asset_registry()
        self.processor = StagedBatchProcessor(config, asset_registry)

    async def run_pilot_stage(
        self, file_paths: List[str], processor_fn: Callable[[List[str]], Awaitable[StageResult]]
    ) -> StageResult:
        """
        Run pilot stage (10 files)

        Args:
            file_paths: List of file paths (should be 10)
            processor_fn: Processing function

        Returns:
            StageResult
        """
        batch_id = self.processor.create_batch_id("pilot")
        logger.info(f"Starting pilot stage with {len(file_paths)} files (batch_id: {batch_id})")

        result = await self.processor.process_stage(
            ProcessingStage.STT, file_paths, batch_id, processor_fn, resume_from_checkpoint=False
        )

        logger.info(f"Pilot stage completed: {result.success_rate:.1%} success rate")
        return result

    async def run_medium_stage(
        self, file_paths: List[str], processor_fn: Callable[[List[str]], Awaitable[StageResult]]
    ) -> StageResult:
        """
        Run medium stage (50 files)

        Args:
            file_paths: List of file paths (should be 50)
            processor_fn: Processing function

        Returns:
            StageResult
        """
        batch_id = self.processor.create_batch_id("medium")
        logger.info(f"Starting medium stage with {len(file_paths)} files (batch_id: {batch_id})")

        result = await self.processor.process_stage(
            ProcessingStage.STT, file_paths, batch_id, processor_fn, resume_from_checkpoint=True
        )

        logger.info(f"Medium stage completed: {result.success_rate:.1%} success rate")
        return result

    async def run_full_stage(
        self, file_paths: List[str], processor_fn: Callable[[List[str]], Awaitable[StageResult]]
    ) -> StageResult:
        """
        Run full stage (183 files)

        Args:
            file_paths: List of file paths (should be 183)
            processor_fn: Processing function

        Returns:
            StageResult
        """
        batch_id = self.processor.create_batch_id("full")
        logger.info(f"Starting full stage with {len(file_paths)} files (batch_id: {batch_id})")

        result = await self.processor.process_stage(
            ProcessingStage.STT, file_paths, batch_id, processor_fn, resume_from_checkpoint=True
        )

        logger.info(f"Full stage completed: {result.success_rate:.1%} success rate")
        return result

    async def run_complete_pipeline(
        self,
        all_files: List[str],
        processor_fn: Callable[[List[str]], Awaitable[StageResult]],
    ) -> Dict[str, StageResult]:
        """
        Run complete staged pipeline (10 → 50 → 183)

        Args:
            all_files: All files to process
            processor_fn: Processing function

        Returns:
            Dictionary with results from each stage
        """
        results = {}

        # Stage 1: Pilot (first 10 files)
        pilot_files = all_files[: self.config.pilot_batch_size]
        results["pilot"] = await self.run_pilot_stage(pilot_files, processor_fn)

        # Stage 2: Medium (next 50 files, includes pilot if successful)
        medium_files = all_files[: self.config.medium_batch_size]
        results["medium"] = await self.run_medium_stage(medium_files, processor_fn)

        # Stage 3: Full (all 183 files)
        results["full"] = await self.run_full_stage(all_files, processor_fn)

        return results
