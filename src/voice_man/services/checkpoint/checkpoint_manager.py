"""
Checkpoint Manager for Batch Processing

Provides high-level checkpoint management for batch workflows:
- Save batch checkpoints with processed and failed files
- Load latest checkpoint for resume capability
- Get resume state to skip already processed files
- Clear checkpoints and list available checkpoints
- Automatic workflow state tracking
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .state_store import (
    CheckpointData,
    FileState,
    FileStatus,
    WorkflowState,
    WorkflowStateStore,
    WorkflowStatus,
)

logger = logging.getLogger(__name__)


@dataclass
class BatchCheckpoint:
    """
    Checkpoint data for a batch processing operation.

    Attributes:
        batch_id: Unique batch identifier
        batch_index: Batch index (1-indexed)
        processed_files: List of successfully processed file paths
        failed_files: List of failed file paths
        pending_files: List of pending file paths
        results: Batch processing results
        timestamp: Checkpoint creation timestamp
        metadata: Additional checkpoint metadata
    """

    batch_id: str
    batch_index: int
    processed_files: List[str]
    failed_files: List[str]
    pending_files: List[str]
    results: Dict[str, Any]
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "batch_id": self.batch_id,
            "batch_index": self.batch_index,
            "processed_files": self.processed_files,
            "failed_files": self.failed_files,
            "pending_files": self.pending_files,
            "results": self.results,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class CheckpointMetadata:
    """
    Metadata about a checkpoint.

    Attributes:
        batch_id: Batch identifier
        batch_index: Batch index
        processed_count: Number of processed files
        failed_count: Number of failed files
        created_at: Checkpoint creation time
    """

    batch_id: str
    batch_index: int
    processed_count: int
    failed_count: int
    created_at: datetime


@dataclass
class ResumeState:
    """
    State for resuming from a checkpoint.

    Attributes:
        workflow_id: Workflow identifier
        current_batch: Current batch index
        total_files: Total number of files
        processed_files: List of processed file paths
        failed_files: List of failed file paths
        pending_files: List of pending file paths
        last_checkpoint: Last checkpoint metadata
        can_resume: Whether resume is possible
    """

    workflow_id: str
    current_batch: int
    total_files: int
    processed_files: List[str]
    failed_files: List[str]
    pending_files: List[str]
    last_checkpoint: Optional[CheckpointMetadata]
    can_resume: bool


class CheckpointManager:
    """
    High-level checkpoint manager for batch processing workflows.

    Features:
    - Automatic workflow state tracking
    - Batch-level checkpointing
    - Resume from last checkpoint
    - File-based and SQLite-based storage
    - Thread-safe operations

    Usage:
        manager = CheckpointManager(checkpoint_dir="data/checkpoints")

        # Create new workflow
        manager.create_workflow(
            workflow_id="wf_001",
            total_files=100,
            batch_size=10
        )

        # Save checkpoint after batch
        manager.save_batch_checkpoint(
            batch_id="batch_1",
            processed_files=["file1.m4a", "file2.m4a"],
            failed_files=[],
            results={"total": 2, "successful": 2}
        )

        # Get resume state
        resume_state = manager.get_resume_state()
        if resume_state.can_resume:
            # Skip already processed files
            remaining_files = [f for f in all_files if f not in resume_state.processed_files]
    """

    def __init__(self, checkpoint_dir: str = "data/checkpoints"):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory for checkpoint files
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize state store
        db_path = str(self.checkpoint_dir / "state.db")
        self.state_store = WorkflowStateStore(db_path=db_path)

        # Current workflow
        self._current_workflow_id: Optional[str] = None

        logger.info(f"CheckpointManager initialized: {checkpoint_dir}")

    def create_workflow(
        self,
        workflow_id: str,
        total_files: int,
        batch_size: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> WorkflowState:
        """
        Create a new workflow for tracking.

        Args:
            workflow_id: Unique workflow identifier
            total_files: Total number of files to process
            batch_size: Number of files per batch
            metadata: Optional workflow metadata

        Returns:
            Created WorkflowState
        """
        # Create workflow in state store
        workflow_metadata = metadata or {}
        workflow_metadata["batch_size"] = batch_size

        state = self.state_store.create_workflow(
            workflow_id=workflow_id,
            total_files=total_files,
            metadata=workflow_metadata,
        )

        self._current_workflow_id = workflow_id
        logger.info(
            f"Created workflow: {workflow_id} ({total_files} files, batch_size={batch_size})"
        )

        return state

    def get_current_workflow_id(self) -> Optional[str]:
        """Get current workflow ID."""
        return self._current_workflow_id

    def save_batch_checkpoint(
        self,
        batch_id: str,
        processed_files: List[str],
        failed_files: List[str],
        results: Dict[str, Any],
        batch_index: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BatchCheckpoint:
        """
        Save checkpoint after batch completion.

        Args:
            batch_id: Batch identifier
            processed_files: List of successfully processed file paths
            failed_files: List of failed file paths
            results: Batch processing results
            batch_index: Batch index (auto-detected if not provided)
            metadata: Additional checkpoint metadata

        Returns:
            Created BatchCheckpoint

        Raises:
            ValueError: If no current workflow is set
        """
        if self._current_workflow_id is None:
            raise ValueError("No current workflow. Call create_workflow() first.")

        # Auto-detect batch index from workflow state
        if batch_index is None:
            workflow_state = self.state_store.get_workflow_state(self._current_workflow_id)
            if workflow_state is None:
                raise ValueError(f"Workflow not found: {self._current_workflow_id}")
            batch_index = workflow_state.current_batch

        # Get all registered files for this workflow
        all_file_states = self.state_store.get_files_by_status(
            self._current_workflow_id, FileStatus.PENDING
        )
        all_pending = [fs.file_path for fs in all_file_states]

        # Determine pending files
        processed_set = set(processed_files)
        failed_set = set(failed_files)
        pending_files = [f for f in all_pending if f not in processed_set and f not in failed_set]

        # Create checkpoint
        timestamp = datetime.now(timezone.utc)
        checkpoint = BatchCheckpoint(
            batch_id=batch_id,
            batch_index=batch_index,
            processed_files=processed_files,
            failed_files=failed_files,
            pending_files=pending_files,
            results=results,
            timestamp=timestamp,
            metadata=metadata or {},
        )

        # Save to state store
        checkpoint_data = CheckpointData(
            batch_id=batch_id,
            workflow_id=self._current_workflow_id,
            batch_index=batch_index,
            processed_files=processed_files,
            failed_files=failed_files,
            results_json=json.dumps(results),
            created_at=timestamp,
        )
        self.state_store.save_checkpoint(checkpoint_data)

        # Update workflow state
        workflow_state = self.state_store.get_workflow_state(self._current_workflow_id)
        if workflow_state:
            new_processed = workflow_state.processed_files + len(processed_files)
            new_failed = workflow_state.failed_files + len(failed_files)
            new_batch = batch_index + 1

            self.state_store.update_workflow_state(
                self._current_workflow_id,
                current_batch=new_batch,
                processed_files=new_processed,
                failed_files=new_failed,
            )

        # Save to JSON file for easy access
        self._save_checkpoint_json(checkpoint)

        logger.info(
            f"Saved checkpoint: {batch_id} (batch {batch_index}, "
            f"{len(processed_files)} processed, {len(failed_files)} failed)"
        )

        return checkpoint

    def load_latest_checkpoint(self) -> Optional[BatchCheckpoint]:
        """
        Load the most recent checkpoint.

        Returns:
            BatchCheckpoint or None if no checkpoints exist
        """
        if self._current_workflow_id is None:
            return None

        checkpoints = self.state_store.get_workflow_checkpoints(self._current_workflow_id)
        if not checkpoints:
            return None

        # Get the last checkpoint
        last_checkpoint_data = checkpoints[-1]

        # Get all files to determine pending
        workflow_state = self.state_store.get_workflow_state(self._current_workflow_id)
        all_files = []
        if workflow_state:
            all_file_states = self.state_store.get_files_by_status(
                self._current_workflow_id, FileStatus.PENDING
            )
            all_files = [fs.file_path for fs in all_file_states]

        processed_set = set(last_checkpoint_data.processed_files)
        failed_set = set(last_checkpoint_data.failed_files)
        pending_files = [f for f in all_files if f not in processed_set and f not in failed_set]

        checkpoint = BatchCheckpoint(
            batch_id=last_checkpoint_data.batch_id,
            batch_index=last_checkpoint_data.batch_index,
            processed_files=last_checkpoint_data.processed_files,
            failed_files=last_checkpoint_data.failed_files,
            pending_files=pending_files,
            results=json.loads(last_checkpoint_data.results_json),
            timestamp=last_checkpoint_data.created_at,
            metadata={},
        )

        logger.info(f"Loaded checkpoint: {checkpoint.batch_id}")
        return checkpoint

    def get_resume_state(self) -> Optional[ResumeState]:
        """
        Get the current resume state.

        Returns:
            ResumeState with information for resuming, or None if no workflow exists
        """
        if self._current_workflow_id is None:
            # Try to find the latest running workflow
            workflows = self.state_store.list_workflows(status=WorkflowStatus.RUNNING)
            if workflows:
                self._current_workflow_id = workflows[0].workflow_id
                logger.info(f"Auto-selected workflow: {self._current_workflow_id}")
            else:
                return None

        workflow_state = self.state_store.get_workflow_state(self._current_workflow_id)
        if workflow_state is None:
            return None

        # Get checkpoint information
        checkpoints = self.state_store.get_workflow_checkpoints(self._current_workflow_id)
        last_checkpoint = None
        if checkpoints:
            last_cp = checkpoints[-1]
            last_checkpoint = CheckpointMetadata(
                batch_id=last_cp.batch_id,
                batch_index=last_cp.batch_index,
                processed_count=len(last_cp.processed_files),
                failed_count=len(last_cp.failed_files),
                created_at=last_cp.created_at,
            )

        # Get file states
        processed = []
        failed = []
        pending = []

        all_file_states = self.state_store.get_files_by_status(
            self._current_workflow_id, FileStatus.PENDING
        )

        for fs in all_file_states:
            if fs.status == FileStatus.COMPLETED:
                processed.append(fs.file_path)
            elif fs.status == FileStatus.FAILED:
                failed.append(fs.file_path)
            else:
                pending.append(fs.file_path)

        can_resume = (
            workflow_state.status == WorkflowStatus.RUNNING
            or workflow_state.status == WorkflowStatus.CRASHED
        )

        resume_state = ResumeState(
            workflow_id=self._current_workflow_id,
            current_batch=workflow_state.current_batch,
            total_files=workflow_state.total_files,
            processed_files=processed,
            failed_files=failed,
            pending_files=pending,
            last_checkpoint=last_checkpoint,
            can_resume=can_resume,
        )

        logger.info(
            f"Resume state: {resume_state.current_batch}/{resume_state.total_files} batches, "
            f"{len(resume_state.processed_files)} processed, {len(resume_state.failed_files)} failed"
        )

        return resume_state

    def list_checkpoints(self) -> List[CheckpointMetadata]:
        """
        List all checkpoints for the current workflow.

        Returns:
            List of CheckpointMetadata
        """
        if self._current_workflow_id is None:
            return []

        checkpoints = self.state_store.get_workflow_checkpoints(self._current_workflow_id)

        metadata_list = []
        for cp in checkpoints:
            metadata_list.append(
                CheckpointMetadata(
                    batch_id=cp.batch_id,
                    batch_index=cp.batch_index,
                    processed_count=len(cp.processed_files),
                    failed_count=len(cp.failed_files),
                    created_at=cp.created_at,
                )
            )

        return metadata_list

    def clear_checkpoint(self, batch_id: str) -> bool:
        """
        Clear a specific checkpoint.

        Args:
            batch_id: Batch identifier

        Returns:
            True if checkpoint was cleared, False otherwise
        """
        if self._current_workflow_id is None:
            return False

        # Delete from database
        conn = self.state_store._get_connection()
        cursor = conn.execute(
            "DELETE FROM checkpoints WHERE batch_id = ? AND workflow_id = ?",
            (batch_id, self._current_workflow_id),
        )
        conn.commit()

        deleted = cursor.rowcount > 0

        # Delete JSON file
        json_path = self._get_checkpoint_json_path(batch_id)
        if json_path.exists():
            json_path.unlink()

        if deleted:
            logger.info(f"Cleared checkpoint: {batch_id}")

        return deleted

    def register_files(self, file_paths: List[str], batch_id: str) -> None:
        """
        Register files for processing in the current workflow.

        Args:
            file_paths: List of file paths to register
            batch_id: Batch ID for these files

        Raises:
            ValueError: If no current workflow is set
        """
        if self._current_workflow_id is None:
            raise ValueError("No current workflow. Call create_workflow() first.")

        for file_path in file_paths:
            self.state_store.register_file(file_path, self._current_workflow_id, batch_id)

        logger.info(f"Registered {len(file_paths)} files for batch {batch_id}")

    def mark_file_processing(
        self,
        file_path: str,
        started_at: Optional[datetime] = None,
    ) -> None:
        """
        Mark a file as currently being processed.

        Args:
            file_path: File path
            started_at: Processing start time (default: now)
        """
        if self._current_workflow_id is None:
            return

        self.state_store.update_file_state(
            file_path,
            status=FileStatus.IN_PROGRESS,
            started_at=started_at or datetime.now(timezone.utc),
        )

    def mark_file_completed(
        self,
        file_path: str,
        result: Optional[Dict[str, Any]] = None,
        completed_at: Optional[datetime] = None,
    ) -> None:
        """
        Mark a file as successfully completed.

        Args:
            file_path: File path
            result: Processing result
            completed_at: Completion time (default: now)
        """
        if self._current_workflow_id is None:
            return

        self.state_store.update_file_state(
            file_path,
            status=FileStatus.COMPLETED,
            result_json=json.dumps(result) if result else None,
            completed_at=completed_at or datetime.now(timezone.utc),
        )

    def mark_file_failed(
        self,
        file_path: str,
        error_message: str,
        attempts: int = 1,
        completed_at: Optional[datetime] = None,
    ) -> None:
        """
        Mark a file as failed.

        Args:
            file_path: File path
            error_message: Error message
            attempts: Number of attempts made
            completed_at: Failure time (default: now)
        """
        if self._current_workflow_id is None:
            return

        self.state_store.update_file_state(
            file_path,
            status=FileStatus.FAILED,
            last_error=error_message,
            attempts=attempts,
            completed_at=completed_at or datetime.now(timezone.utc),
        )

    def complete_workflow(self, success: bool = True) -> None:
        """
        Mark the current workflow as completed.

        Args:
            success: Whether the workflow completed successfully
        """
        if self._current_workflow_id is None:
            return

        status = WorkflowStatus.COMPLETED if success else WorkflowStatus.FAILED
        completed_at = datetime.now(timezone.utc)

        self.state_store.update_workflow_state(
            self._current_workflow_id,
            status=status,
            completed_at=completed_at,
        )

        logger.info(f"Workflow completed: {self._current_workflow_id} (status={status.value})")

    def cancel_workflow(self) -> None:
        """Mark the current workflow as cancelled."""
        if self._current_workflow_id is None:
            return

        self.state_store.update_workflow_state(
            self._current_workflow_id,
            status=WorkflowStatus.CANCELLED,
            completed_at=datetime.now(timezone.utc),
        )

        logger.info(f"Workflow cancelled: {self._current_workflow_id}")

    def _save_checkpoint_json(self, checkpoint: BatchCheckpoint) -> None:
        """Save checkpoint to JSON file for easy access."""
        json_path = self._get_checkpoint_json_path(checkpoint.batch_id)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(checkpoint.to_dict(), f, indent=2, ensure_ascii=False)

    def _get_checkpoint_json_path(self, batch_id: str) -> Path:
        """Get the JSON file path for a checkpoint."""
        return self.checkpoint_dir / f"{batch_id}_checkpoint.json"

    def close(self) -> None:
        """Close the checkpoint manager and database connection."""
        self.state_store.close()
        logger.info("CheckpointManager closed")
