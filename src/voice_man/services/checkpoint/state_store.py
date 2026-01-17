"""
SQLite-based Transactional State Storage for Workflow Checkpoints

Provides ACID-compliant state tracking for forensic pipeline workflows.
Thread-safe database operations with automatic connection pooling.

Database Schema:
- workflow_states: Top-level workflow metadata and status
- file_states: Per-file processing state tracking
- checkpoints: Batch-level checkpoint snapshots
"""

import json
import logging
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class WorkflowStatus(str, Enum):
    """Status of a workflow execution."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CRASHED = "crashed"
    CANCELLED = "cancelled"


class FileStatus(str, Enum):
    """Status of a file within a workflow."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class WorkflowState:
    """
    State of a workflow execution.

    Attributes:
        workflow_id: Unique workflow identifier
        current_batch: Current batch index (1-indexed)
        total_files: Total number of files to process
        processed_files: Number of successfully processed files
        failed_files: Number of failed files
        status: Current workflow status
        started_at: Workflow start timestamp
        updated_at: Last update timestamp
        completed_at: Workflow completion timestamp (if completed)
        metadata: Additional workflow metadata
    """

    workflow_id: str
    current_batch: int
    total_files: int
    processed_files: int = 0
    failed_files: int = 0
    status: WorkflowStatus = WorkflowStatus.RUNNING
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "workflow_id": self.workflow_id,
            "current_batch": self.current_batch,
            "total_files": self.total_files,
            "processed_files": self.processed_files,
            "failed_files": self.failed_files,
            "status": self.status.value,
            "started_at": self.started_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowState":
        """Create from dictionary."""
        return cls(
            workflow_id=data["workflow_id"],
            current_batch=data["current_batch"],
            total_files=data["total_files"],
            processed_files=data.get("processed_files", 0),
            failed_files=data.get("failed_files", 0),
            status=WorkflowStatus(data.get("status", WorkflowStatus.RUNNING)),
            started_at=datetime.fromisoformat(data["started_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            completed_at=datetime.fromisoformat(data["completed_at"])
            if data.get("completed_at")
            else None,
            metadata=data.get("metadata", {}),
        )


@dataclass
class FileState:
    """
    State of a single file within a workflow.

    Attributes:
        file_path: Path to the audio file
        workflow_id: Parent workflow ID
        batch_id: Batch ID this file belongs to
        status: Current file status
        attempts: Number of processing attempts
        last_error: Last error message (if failed)
        result_json: Processing result as JSON string
        started_at: File processing start timestamp
        completed_at: File processing completion timestamp
    """

    file_path: str
    workflow_id: str
    batch_id: str
    status: FileStatus = FileStatus.PENDING
    attempts: int = 0
    last_error: Optional[str] = None
    result_json: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "file_path": self.file_path,
            "workflow_id": self.workflow_id,
            "batch_id": self.batch_id,
            "status": self.status.value,
            "attempts": self.attempts,
            "last_error": self.last_error,
            "result_json": self.result_json,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FileState":
        """Create from dictionary."""
        return cls(
            file_path=data["file_path"],
            workflow_id=data["workflow_id"],
            batch_id=data["batch_id"],
            status=FileStatus(data.get("status", FileStatus.PENDING)),
            attempts=data.get("attempts", 0),
            last_error=data.get("last_error"),
            result_json=data.get("result_json"),
            started_at=datetime.fromisoformat(data["started_at"])
            if data.get("started_at")
            else None,
            completed_at=datetime.fromisoformat(data["completed_at"])
            if data.get("completed_at")
            else None,
        )


@dataclass
class CheckpointData:
    """
    Batch-level checkpoint snapshot.

    Attributes:
        batch_id: Batch identifier
        workflow_id: Parent workflow ID
        batch_index: Batch index (1-indexed)
        processed_files: JSON array of processed file paths
        failed_files: JSON array of failed file paths
        results_json: Batch results as JSON string
        created_at: Checkpoint creation timestamp
    """

    batch_id: str
    workflow_id: str
    batch_index: int
    processed_files: List[str]
    failed_files: List[str]
    results_json: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "batch_id": self.batch_id,
            "workflow_id": self.workflow_id,
            "batch_index": self.batch_index,
            "processed_files": self.processed_files,
            "failed_files": self.failed_files,
            "results_json": self.results_json,
            "created_at": self.created_at.isoformat(),
        }


class WorkflowStateStore:
    """
    SQLite-based workflow state store with ACID guarantees.

    Features:
    - Thread-safe database operations
    - Automatic connection pooling
    - Transactional updates for consistency
    - Automatic schema initialization
    - WAL mode for concurrent access
    - Foreign key constraints for data integrity

    Usage:
        store = WorkflowStateStore(db_path="data/checkpoints/state.db")

        # Create workflow
        store.create_workflow(
            workflow_id="wf_001",
            total_files=100,
            metadata={"batch_size": 10}
        )

        # Update file state
        store.update_file_state(
            file_path="audio.m4a",
            status=FileStatus.COMPLETED,
            result_json='{"score": 0.85}'
        )

        # Get workflow state
        state = store.get_workflow_state("wf_001")
    """

    def __init__(self, db_path: str = "data/checkpoints/state.db"):
        """
        Initialize state store.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._local = threading.local()
        self._lock = threading.Lock()

        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # Initialize database schema
        self._init_schema()

        logger.info(f"WorkflowStateStore initialized: {db_path}")

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
            # Enable WAL mode for concurrent access
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            # Enable foreign keys
            self._local.conn.execute("PRAGMA foreign_keys=ON")
        return self._local.conn

    def _init_schema(self):
        """Initialize database schema."""
        conn = self._get_connection()

        # Create workflow_states table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS workflow_states (
                workflow_id TEXT PRIMARY KEY,
                current_batch INTEGER NOT NULL DEFAULT 1,
                total_files INTEGER NOT NULL,
                processed_files INTEGER NOT NULL DEFAULT 0,
                failed_files INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'running',
                started_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                completed_at TEXT,
                metadata TEXT DEFAULT '{}'
            )
        """
        )

        # Create file_states table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS file_states (
                file_path TEXT PRIMARY KEY,
                workflow_id TEXT NOT NULL,
                batch_id TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                attempts INTEGER NOT NULL DEFAULT 0,
                last_error TEXT,
                result_json TEXT,
                started_at TEXT,
                completed_at TEXT,
                FOREIGN KEY (workflow_id) REFERENCES workflow_states(workflow_id) ON DELETE CASCADE
            )
        """
        )

        # Create checkpoints table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS checkpoints (
                batch_id TEXT PRIMARY KEY,
                workflow_id TEXT NOT NULL,
                batch_index INTEGER NOT NULL,
                processed_files TEXT NOT NULL,
                failed_files TEXT NOT NULL,
                results_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (workflow_id) REFERENCES workflow_states(workflow_id) ON DELETE CASCADE
            )
        """
        )

        # Create indexes for performance
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_file_states_workflow ON file_states(workflow_id)"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_file_states_batch ON file_states(batch_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_file_states_status ON file_states(status)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_checkpoints_workflow ON checkpoints(workflow_id)"
        )

        conn.commit()
        logger.info("Database schema initialized")

    def create_workflow(
        self,
        workflow_id: str,
        total_files: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> WorkflowState:
        """
        Create a new workflow.

        Args:
            workflow_id: Unique workflow identifier
            total_files: Total number of files to process
            metadata: Optional workflow metadata

        Returns:
            Created WorkflowState
        """
        conn = self._get_connection()
        now = datetime.now(timezone.utc).isoformat()

        state = WorkflowState(
            workflow_id=workflow_id,
            current_batch=1,
            total_files=total_files,
            processed_files=0,
            failed_files=0,
            status=WorkflowStatus.RUNNING,
            started_at=datetime.fromisoformat(now),
            updated_at=datetime.fromisoformat(now),
            metadata=metadata or {},
        )

        conn.execute(
            """
            INSERT INTO workflow_states (
                workflow_id, current_batch, total_files, processed_files, failed_files,
                status, started_at, updated_at, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                state.workflow_id,
                state.current_batch,
                state.total_files,
                state.processed_files,
                state.failed_files,
                state.status.value,
                state.started_at.isoformat(),
                state.updated_at.isoformat(),
                json.dumps(state.metadata),
            ),
        )

        conn.commit()
        logger.info(f"Created workflow: {workflow_id} ({total_files} files)")

        return state

    def get_workflow_state(self, workflow_id: str) -> Optional[WorkflowState]:
        """
        Get workflow state by ID.

        Args:
            workflow_id: Workflow identifier

        Returns:
            WorkflowState or None if not found
        """
        conn = self._get_connection()
        cursor = conn.execute("SELECT * FROM workflow_states WHERE workflow_id = ?", (workflow_id,))
        row = cursor.fetchone()

        if row is None:
            return None

        return WorkflowState(
            workflow_id=row["workflow_id"],
            current_batch=row["current_batch"],
            total_files=row["total_files"],
            processed_files=row["processed_files"],
            failed_files=row["failed_files"],
            status=WorkflowStatus(row["status"]),
            started_at=datetime.fromisoformat(row["started_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            completed_at=datetime.fromisoformat(row["completed_at"])
            if row["completed_at"]
            else None,
            metadata=json.loads(row["metadata"]),
        )

    def update_workflow_state(
        self,
        workflow_id: str,
        current_batch: Optional[int] = None,
        processed_files: Optional[int] = None,
        failed_files: Optional[int] = None,
        status: Optional[WorkflowStatus] = None,
        completed_at: Optional[datetime] = None,
        metadata_updates: Optional[Dict[str, Any]] = None,
    ) -> Optional[WorkflowState]:
        """
        Update workflow state.

        Args:
            workflow_id: Workflow identifier
            current_batch: New current batch index
            processed_files: New processed files count
            failed_files: New failed files count
            status: New workflow status
            completed_at: Completion timestamp
            metadata_updates: Metadata updates to merge

        Returns:
            Updated WorkflowState or None if not found
        """
        conn = self._get_connection()

        # Get current state
        current = self.get_workflow_state(workflow_id)
        if current is None:
            return None

        # Update fields
        if current_batch is not None:
            current.current_batch = current_batch
        if processed_files is not None:
            current.processed_files = processed_files
        if failed_files is not None:
            current.failed_files = failed_files
        if status is not None:
            current.status = status
        if completed_at is not None:
            current.completed_at = completed_at
        if metadata_updates:
            current.metadata.update(metadata_updates)

        current.updated_at = datetime.now(timezone.utc)

        # Update database
        conn.execute(
            """
            UPDATE workflow_states
            SET current_batch = ?, processed_files = ?, failed_files = ?,
                status = ?, completed_at = ?, updated_at = ?, metadata = ?
            WHERE workflow_id = ?
        """,
            (
                current.current_batch,
                current.processed_files,
                current.failed_files,
                current.status.value,
                current.completed_at.isoformat() if current.completed_at else None,
                current.updated_at.isoformat(),
                json.dumps(current.metadata),
                workflow_id,
            ),
        )

        conn.commit()
        return current

    def register_file(
        self,
        file_path: str,
        workflow_id: str,
        batch_id: str,
    ) -> FileState:
        """
        Register a file for processing.

        Args:
            file_path: Path to the file
            workflow_id: Parent workflow ID
            batch_id: Batch ID

        Returns:
            Created FileState
        """
        conn = self._get_connection()

        state = FileState(
            file_path=file_path,
            workflow_id=workflow_id,
            batch_id=batch_id,
            status=FileStatus.PENDING,
        )

        conn.execute(
            """
            INSERT INTO file_states (
                file_path, workflow_id, batch_id, status, attempts
            ) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(file_path) DO UPDATE SET
                workflow_id = excluded.workflow_id,
                batch_id = excluded.batch_id
        """,
            (
                state.file_path,
                state.workflow_id,
                state.batch_id,
                state.status.value,
                state.attempts,
            ),
        )

        conn.commit()
        return state

    def get_file_state(self, file_path: str) -> Optional[FileState]:
        """
        Get file state by path.

        Args:
            file_path: File path

        Returns:
            FileState or None if not found
        """
        conn = self._get_connection()
        cursor = conn.execute("SELECT * FROM file_states WHERE file_path = ?", (file_path,))
        row = cursor.fetchone()

        if row is None:
            return None

        return FileState(
            file_path=row["file_path"],
            workflow_id=row["workflow_id"],
            batch_id=row["batch_id"],
            status=FileStatus(row["status"]),
            attempts=row["attempts"],
            last_error=row["last_error"],
            result_json=row["result_json"],
            started_at=datetime.fromisoformat(row["started_at"]) if row["started_at"] else None,
            completed_at=datetime.fromisoformat(row["completed_at"])
            if row["completed_at"]
            else None,
        )

    def update_file_state(
        self,
        file_path: str,
        status: Optional[FileStatus] = None,
        attempts: Optional[int] = None,
        last_error: Optional[str] = None,
        result_json: Optional[str] = None,
        started_at: Optional[datetime] = None,
        completed_at: Optional[datetime] = None,
    ) -> Optional[FileState]:
        """
        Update file state.

        Args:
            file_path: File path
            status: New file status
            attempts: New attempt count
            last_error: Error message
            result_json: Result JSON string
            started_at: Processing start time
            completed_at: Processing completion time

        Returns:
            Updated FileState or None if not found
        """
        conn = self._get_connection()

        # Get current state
        current = self.get_file_state(file_path)
        if current is None:
            return None

        # Update fields
        if status is not None:
            current.status = status
        if attempts is not None:
            current.attempts = attempts
        if last_error is not None:
            current.last_error = last_error
        if result_json is not None:
            current.result_json = result_json
        if started_at is not None:
            current.started_at = started_at
        if completed_at is not None:
            current.completed_at = completed_at

        # Update database
        conn.execute(
            """
            UPDATE file_states
            SET status = ?, attempts = ?, last_error = ?, result_json = ?,
                started_at = ?, completed_at = ?
            WHERE file_path = ?
        """,
            (
                current.status.value,
                current.attempts,
                current.last_error,
                current.result_json,
                current.started_at.isoformat() if current.started_at else None,
                current.completed_at.isoformat() if current.completed_at else None,
                file_path,
            ),
        )

        conn.commit()
        return current

    def save_checkpoint(self, checkpoint: CheckpointData, fatal_on_failure: bool = True) -> None:
        """
        Save batch checkpoint.

        Args:
            checkpoint: Checkpoint data to save
            fatal_on_failure: Raise exception on save failure (default: True)

        Raises:
            RuntimeError: If checkpoint save fails and fatal_on_failure is True
        """
        try:
            conn = self._get_connection()

            # Validate checkpoint before saving
            self._validate_checkpoint_data(checkpoint)

            conn.execute(
                """
                INSERT INTO checkpoints (
                    batch_id, workflow_id, batch_index, processed_files, failed_files, results_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(batch_id) DO UPDATE SET
                    workflow_id = excluded.workflow_id,
                    batch_index = excluded.batch_index,
                    processed_files = excluded.processed_files,
                    failed_files = excluded.failed_files,
                    results_json = excluded.results_json,
                    created_at = datetime('now')
            """,
                (
                    checkpoint.batch_id,
                    checkpoint.workflow_id,
                    checkpoint.batch_index,
                    json.dumps(checkpoint.processed_files),
                    json.dumps(checkpoint.failed_files),
                    checkpoint.results_json,
                ),
            )

            conn.commit()
            logger.info(f"Checkpoint saved: {checkpoint.batch_id}")

        except Exception as e:
            error_msg = f"Failed to save checkpoint {checkpoint.batch_id}: {e}"
            logger.error(error_msg)

            if fatal_on_failure:
                raise RuntimeError(error_msg) from e

    def get_checkpoint(self, batch_id: str) -> Optional[CheckpointData]:
        """
        Get checkpoint by batch ID.

        Args:
            batch_id: Batch identifier

        Returns:
            CheckpointData or None if not found
        """
        conn = self._get_connection()
        cursor = conn.execute("SELECT * FROM checkpoints WHERE batch_id = ?", (batch_id,))
        row = cursor.fetchone()

        if row is None:
            return None

        return CheckpointData(
            batch_id=row["batch_id"],
            workflow_id=row["workflow_id"],
            batch_index=row["batch_index"],
            processed_files=json.loads(row["processed_files"]),
            failed_files=json.loads(row["failed_files"]),
            results_json=row["results_json"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def get_workflow_checkpoints(self, workflow_id: str) -> List[CheckpointData]:
        """
        Get all checkpoints for a workflow.

        Args:
            workflow_id: Workflow identifier

        Returns:
            List of checkpoints ordered by batch_index
        """
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT * FROM checkpoints WHERE workflow_id = ? ORDER BY batch_index",
            (workflow_id,),
        )

        checkpoints = []
        for row in cursor.fetchall():
            checkpoints.append(
                CheckpointData(
                    batch_id=row["batch_id"],
                    workflow_id=row["workflow_id"],
                    batch_index=row["batch_index"],
                    processed_files=json.loads(row["processed_files"]),
                    failed_files=json.loads(row["failed_files"]),
                    results_json=row["results_json"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                )
            )

        return checkpoints

    def get_files_by_status(self, workflow_id: str, status: FileStatus) -> List[FileState]:
        """
        Get all files with a specific status.

        Args:
            workflow_id: Workflow identifier
            status: File status to filter by

        Returns:
            List of FileState objects
        """
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT * FROM file_states WHERE workflow_id = ? AND status = ?",
            (workflow_id, status.value),
        )

        files = []
        for row in cursor.fetchall():
            files.append(
                FileState(
                    file_path=row["file_path"],
                    workflow_id=row["workflow_id"],
                    batch_id=row["batch_id"],
                    status=FileStatus(row["status"]),
                    attempts=row["attempts"],
                    last_error=row["last_error"],
                    result_json=row["result_json"],
                    started_at=datetime.fromisoformat(row["started_at"])
                    if row["started_at"]
                    else None,
                    completed_at=datetime.fromisoformat(row["completed_at"])
                    if row["completed_at"]
                    else None,
                )
            )

        return files

    def list_workflows(self, status: Optional[WorkflowStatus] = None) -> List[WorkflowState]:
        """
        List all workflows, optionally filtered by status.

        Args:
            status: Optional status filter

        Returns:
            List of WorkflowState objects
        """
        conn = self._get_connection()

        if status:
            cursor = conn.execute(
                "SELECT * FROM workflow_states WHERE status = ? ORDER BY started_at DESC",
                (status.value,),
            )
        else:
            cursor = conn.execute("SELECT * FROM workflow_states ORDER BY started_at DESC")

        workflows = []
        for row in cursor.fetchall():
            workflows.append(
                WorkflowState(
                    workflow_id=row["workflow_id"],
                    current_batch=row["current_batch"],
                    total_files=row["total_files"],
                    processed_files=row["processed_files"],
                    failed_files=row["failed_files"],
                    status=WorkflowStatus(row["status"]),
                    started_at=datetime.fromisoformat(row["started_at"]),
                    updated_at=datetime.fromisoformat(row["updated_at"]),
                    completed_at=datetime.fromisoformat(row["completed_at"])
                    if row["completed_at"]
                    else None,
                    metadata=json.loads(row["metadata"]),
                )
            )

        return workflows

    def delete_workflow(self, workflow_id: str) -> bool:
        """
        Delete a workflow and all associated data.

        Args:
            workflow_id: Workflow identifier

        Returns:
            True if deleted, False if not found
        """
        conn = self._get_connection()
        cursor = conn.execute("DELETE FROM workflow_states WHERE workflow_id = ?", (workflow_id,))
        conn.commit()

        deleted = cursor.rowcount > 0
        if deleted:
            logger.info(f"Deleted workflow: {workflow_id}")

        return deleted

    def validate_checkpoint(self, batch_id: str) -> bool:
        """
        Validate a checkpoint before loading or resuming.

        Performs comprehensive validation:
        - Database connection check
        - Checkpoint existence
        - Data structure validity
        - JSON parsing for results
        - Workflow reference integrity

        Args:
            batch_id: Batch checkpoint identifier

        Returns:
            True if checkpoint is valid, False otherwise
        """
        try:
            # Check database connection
            conn = self._get_connection()

            # Check checkpoint exists
            cursor = conn.execute("SELECT * FROM checkpoints WHERE batch_id = ?", (batch_id,))
            row = cursor.fetchone()

            if row is None:
                logger.warning(f"Checkpoint validation failed: {batch_id} not found")
                return False

            # Validate data structure
            checkpoint = self.get_checkpoint(batch_id)
            if checkpoint is None:
                logger.warning(f"Checkpoint validation failed: {batch_id} could not be loaded")
                return False

            # Validate checkpoint data
            self._validate_checkpoint_data(checkpoint)

            # Check parent workflow exists
            workflow = self.get_workflow_state(checkpoint.workflow_id)
            if workflow is None:
                logger.warning(
                    f"Checkpoint validation failed: {batch_id} references "
                    f"non-existent workflow {checkpoint.workflow_id}"
                )
                return False

            logger.info(f"Checkpoint validation passed: {batch_id}")
            return True

        except Exception as e:
            logger.error(f"Checkpoint validation failed with exception: {e}")
            return False

    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the checkpoint system.

        Returns:
            Dictionary with health status information
        """
        health = {
            "healthy": True,
            "database_accessible": False,
            "workflow_count": 0,
            "checkpoint_count": 0,
            "file_state_count": 0,
            "issues": [],
        }

        try:
            # Test database connection
            conn = self._get_connection()

            # Check database accessibility
            cursor = conn.execute("SELECT 1")
            if cursor.fetchone():
                health["database_accessible"] = True

            # Count records
            cursor = conn.execute("SELECT COUNT(*) FROM workflow_states")
            health["workflow_count"] = cursor.fetchone()[0]

            cursor = conn.execute("SELECT COUNT(*) FROM checkpoints")
            health["checkpoint_count"] = cursor.fetchone()[0]

            cursor = conn.execute("SELECT COUNT(*) FROM file_states")
            health["file_state_count"] = cursor.fetchone()[0]

            # Check for orphaned records
            cursor = conn.execute(
                """
                SELECT COUNT(*) FROM file_states fs
                LEFT JOIN workflow_states ws ON fs.workflow_id = ws.workflow_id
                WHERE ws.workflow_id IS NULL
            """
            )
            orphaned_files = cursor.fetchone()[0]
            if orphaned_files > 0:
                health["issues"].append(f"{orphaned_files} orphaned file states found")
                health["healthy"] = False

            # Check for workflows in invalid state
            cursor = conn.execute(
                "SELECT COUNT(*) FROM workflow_states WHERE status NOT IN ('running', 'completed', 'failed', 'crashed', 'cancelled')"
            )
            invalid_states = cursor.fetchone()[0]
            if invalid_states > 0:
                health["issues"].append(f"{invalid_states} workflows with invalid status")
                health["healthy"] = False

        except Exception as e:
            health["healthy"] = False
            health["issues"].append(f"Health check failed: {e}")
            logger.error(f"Health check failed: {e}")

        return health

    def _validate_checkpoint_data(self, checkpoint: CheckpointData) -> None:
        """
        Validate checkpoint data structure.

        Args:
            checkpoint: Checkpoint data to validate

        Raises:
            ValueError: If checkpoint data is invalid
        """
        # Check required fields
        if not checkpoint.batch_id:
            raise ValueError("Checkpoint has empty batch_id")

        if not checkpoint.workflow_id:
            raise ValueError("Checkpoint has empty workflow_id")

        if checkpoint.batch_index < 1:
            raise ValueError(f"Checkpoint has invalid batch_index: {checkpoint.batch_index}")

        # Check processed_files is a list
        if not isinstance(checkpoint.processed_files, list):
            raise ValueError("Checkpoint processed_files is not a list")

        # Check failed_files is a list
        if not isinstance(checkpoint.failed_files, list):
            raise ValueError("Checkpoint failed_files is not a list")

        # Validate results_json is valid JSON
        if checkpoint.results_json:
            try:
                json.loads(checkpoint.results_json)
            except json.JSONDecodeError as e:
                raise ValueError(f"Checkpoint results_json is invalid: {e}") from e

    def close(self):
        """Close database connection."""
        if hasattr(self._local, "conn"):
            self._local.conn.close()
            delattr(self._local, "conn")
            logger.info("Database connection closed")

    def __del__(self):
        """Cleanup on deletion."""
        self.close()
