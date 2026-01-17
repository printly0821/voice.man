"""
Checkpoint and Crash Recovery Service for Forensic Pipeline

Provides fault-tolerant batch processing with:
- SQLite-based transactional state storage
- Per-file state tracking (PENDING, IN_PROGRESS, COMPLETED, FAILED)
- Batch-level checkpointing with automatic resume
- Error classification (TRANSIENT, RECOVERABLE, PERMANENT)
- Graceful shutdown handling
- Progress monitoring with ETA calculation
- Thread-safe database operations
- Comprehensive checkpoint validation with repair capabilities

Usage:
    from voice_man.services.checkpoint import CheckpointManager, CheckpointValidator

    manager = CheckpointManager(checkpoint_dir="data/checkpoints")
    state = manager.get_resume_state()

    if state:
        print(f"Resuming from batch {state.current_batch}")

    # Process files...
    manager.save_batch_checkpoint(
        batch_id="batch_1",
        processed_files=["file1.m4a", "file2.m4a"],
        failed_files=[],
        results={"total": 2, "successful": 2}
    )

    # Validate checkpoint before resume
    result = manager.validate_current_workflow()
    if not result.is_valid:
        print(f"Validation failed: {result.get_summary()}")
"""

from .checkpoint_manager import (
    BatchCheckpoint,
    CheckpointManager,
    CheckpointMetadata,
    ResumeState,
)
from .checkpoint_validator import (
    CheckpointValidator,
    ValidationError,
    ValidationResult,
)
from .error_classifier import (
    ErrorCategory,
    ErrorClassifier,
    ErrorSeverity,
    RetryStrategy,
)
from .graceful_shutdown import GracefulShutdown, ShutdownHandler
from .progress_tracker import ProgressTracker, ProgressUpdate
from .state_store import (
    FileState,
    WorkflowState,
    WorkflowStateStore,
    WorkflowStatus,
)

__all__ = [
    # Checkpoint Manager
    "CheckpointManager",
    "BatchCheckpoint",
    "CheckpointMetadata",
    "ResumeState",
    # Checkpoint Validator
    "CheckpointValidator",
    "ValidationResult",
    "ValidationError",
    # State Store
    "WorkflowStateStore",
    "WorkflowState",
    "FileState",
    "WorkflowStatus",
    # Error Classifier
    "ErrorClassifier",
    "ErrorCategory",
    "ErrorSeverity",
    "RetryStrategy",
    # Progress Tracker
    "ProgressTracker",
    "ProgressUpdate",
    # Graceful Shutdown
    "GracefulShutdown",
    "ShutdownHandler",
]
