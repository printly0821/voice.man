"""
Checkpoint Validator for Database and State Consistency

Provides comprehensive validation for checkpoint system:
- Database integrity verification
- File existence validation for all processed files
- Metadata consistency checks
- Repair capability for minor corruptions
- Health check functionality
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

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
class ValidationError:
    """
    Represents a validation error found during checkpoint validation.

    Attributes:
        severity: Error severity (critical, warning, info)
        category: Type of validation error
        message: Human-readable error message
        details: Additional error details
        affected_entity: ID of the affected entity (workflow_id, batch_id, file_path)
        repairable: Whether this error can be automatically repaired
    """

    severity: str
    category: str
    message: str
    details: Optional[str] = None
    affected_entity: Optional[str] = None
    repairable: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "severity": self.severity,
            "category": self.category,
            "message": self.message,
            "details": self.details,
            "affected_entity": self.affected_entity,
            "repairable": self.repairable,
        }


@dataclass
class ValidationResult:
    """
    Result of checkpoint validation.

    Attributes:
        is_valid: Whether checkpoint passed all critical checks
        errors: List of validation errors found
        critical_count: Number of critical errors
        warning_count: Number of warnings
        repairable_count: Number of repairable errors
        validated_at: Validation timestamp
    """

    is_valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    critical_count: int = 0
    warning_count: int = 0
    repairable_count: int = 0
    validated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def add_error(self, error: ValidationError) -> None:
        """Add a validation error to the result."""
        self.errors.append(error)
        if error.severity == "critical":
            self.critical_count += 1
        elif error.severity == "warning":
            self.warning_count += 1
        if error.repairable:
            self.repairable_count += 1

        # Update overall validity
        self.is_valid = self.critical_count == 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "is_valid": self.is_valid,
            "critical_count": self.critical_count,
            "warning_count": self.warning_count,
            "repairable_count": self.repairable_count,
            "validated_at": self.validated_at.isoformat(),
            "errors": [e.to_dict() for e in self.errors],
        }

    def get_summary(self) -> str:
        """Get human-readable summary of validation result."""
        if self.is_valid:
            return f"Checkpoint valid: {self.warning_count} warnings"

        return (
            f"Checkpoint invalid: {self.critical_count} critical, "
            f"{self.warning_count} warnings ({self.repairable_count} repairable)"
        )


class CheckpointValidator:
    """
    Comprehensive checkpoint validation with repair capabilities.

    Features:
    - Database integrity verification
    - File existence validation
    - Workflow state consistency checks
    - Metadata validation
    - Automatic repair for minor issues
    - Health check functionality

    Usage:
        validator = CheckpointValidator(state_store)

        # Validate entire checkpoint system
        result = validator.validate_all()
        if not result.is_valid:
            print(f"Validation failed: {result.get_summary()}")
            if result.repairable_count > 0:
                validator.repair_all(result)

        # Validate specific workflow
        result = validator.validate_workflow("workflow_123")

        # Health check
        is_healthy = validator.health_check()
    """

    def __init__(self, state_store: WorkflowStateStore):
        """
        Initialize checkpoint validator.

        Args:
            state_store: WorkflowStateStore instance to validate
        """
        self.state_store = state_store

    def validate_all(self) -> ValidationResult:
        """
        Validate entire checkpoint system.

        Performs comprehensive validation:
        - Database schema integrity
        - All workflows consistency
        - All files existence
        - All checkpoints validity

        Returns:
            ValidationResult with all errors found
        """
        result = ValidationResult(is_valid=True)

        logger.info("Starting comprehensive checkpoint validation...")

        # Step 1: Database integrity
        db_valid = self._validate_database_integrity(result)
        if not db_valid:
            logger.error("Database integrity check failed")
            return result

        # Step 2: Validate all workflows
        workflows = self.state_store.list_workflows()
        logger.info(f"Validating {len(workflows)} workflows...")

        for workflow in workflows:
            self._validate_workflow_state(workflow, result)

        # Step 3: Validate orphaned records
        self._validate_orphaned_records(result)

        logger.info(f"Validation complete: {result.get_summary()}")
        return result

    def validate_workflow(self, workflow_id: str) -> ValidationResult:
        """
        Validate a specific workflow and all its dependencies.

        Args:
            workflow_id: Workflow identifier to validate

        Returns:
            ValidationResult with errors found
        """
        result = ValidationResult(is_valid=True)

        logger.info(f"Validating workflow: {workflow_id}")

        # Get workflow state
        workflow = self.state_store.get_workflow_state(workflow_id)
        if workflow is None:
            result.add_error(
                ValidationError(
                    severity="critical",
                    category="workflow_not_found",
                    message=f"Workflow not found: {workflow_id}",
                    affected_entity=workflow_id,
                    repairable=False,
                )
            )
            return result

        # Validate workflow state
        self._validate_workflow_state(workflow, result)

        # Validate files
        self._validate_workflow_files(workflow_id, result)

        # Validate checkpoints
        self._validate_workflow_checkpoints(workflow_id, result)

        logger.info(f"Workflow validation complete: {result.get_summary()}")
        return result

    def validate_checkpoint(self, batch_id: str) -> ValidationResult:
        """
        Validate a specific checkpoint.

        Args:
            batch_id: Batch checkpoint identifier

        Returns:
            ValidationResult with errors found
        """
        result = ValidationResult(is_valid=True)

        logger.info(f"Validating checkpoint: {batch_id}")

        # Get checkpoint
        checkpoint = self.state_store.get_checkpoint(batch_id)
        if checkpoint is None:
            result.add_error(
                ValidationError(
                    severity="critical",
                    category="checkpoint_not_found",
                    message=f"Checkpoint not found: {batch_id}",
                    affected_entity=batch_id,
                    repairable=False,
                )
            )
            return result

        # Validate checkpoint data
        self._validate_checkpoint_data(checkpoint, result)

        # Validate parent workflow
        workflow = self.state_store.get_workflow_state(checkpoint.workflow_id)
        if workflow is None:
            result.add_error(
                ValidationError(
                    severity="critical",
                    category="orphaned_checkpoint",
                    message=f"Checkpoint references non-existent workflow: {checkpoint.workflow_id}",
                    affected_entity=batch_id,
                    details=f"Checkpoint {batch_id} has workflow_id={checkpoint.workflow_id} but workflow not found",
                    repairable=False,
                )
            )

        logger.info(f"Checkpoint validation complete: {result.get_summary()}")
        return result

    def health_check(self) -> bool:
        """
        Quick health check for checkpoint system.

        Checks:
        - Database connection
        - Schema integrity
        - Basic data consistency

        Returns:
            True if system is healthy, False otherwise
        """
        try:
            # Test database connection
            conn = self.state_store._get_connection()

            # Test basic query
            cursor = conn.execute("SELECT COUNT(*) FROM workflow_states")
            workflow_count = cursor.fetchone()[0]

            # Check schema
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]

            required_tables = {"workflow_states", "file_states", "checkpoints"}
            if not required_tables.issubset(set(tables)):
                logger.error(f"Missing required tables: {required_tables - set(tables)}")
                return False

            logger.info(f"Health check passed: {workflow_count} workflows, {len(tables)} tables")
            return True

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def repair_all(self, validation_result: ValidationResult) -> int:
        """
        Attempt to repair all repairable errors in validation result.

        Args:
            validation_result: ValidationResult from previous validation

        Returns:
            Number of repairs successfully applied
        """
        repairs_applied = 0

        for error in validation_result.errors:
            if error.repairable:
                try:
                    if error.category == "orphaned_file_state":
                        if self._repair_orphaned_file_state(error):
                            repairs_applied += 1
                    elif error.category == "inconsistent_counts":
                        if self._repair_inconsistent_counts(error):
                            repairs_applied += 1
                    elif error.category == "missing_timestamp":
                        if self._repair_missing_timestamp(error):
                            repairs_applied += 1

                except Exception as e:
                    logger.warning(f"Failed to repair {error.category}: {e}")

        logger.info(f"Applied {repairs_applied}/{validation_result.repairable_count} repairs")
        return repairs_applied

    def _validate_database_integrity(self, result: ValidationResult) -> bool:
        """Validate database schema and basic integrity."""
        try:
            conn = self.state_store._get_connection()

            # Check all required tables exist
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = {row[0] for row in cursor.fetchall()}

            required_tables = {"workflow_states", "file_states", "checkpoints"}
            missing_tables = required_tables - tables

            if missing_tables:
                for table in missing_tables:
                    result.add_error(
                        ValidationError(
                            severity="critical",
                            category="missing_table",
                            message=f"Required table missing: {table}",
                            affected_entity=table,
                            repairable=False,
                        )
                    )
                return False

            # Check foreign key constraints are enabled
            cursor = conn.execute("PRAGMA foreign_keys")
            fk_enabled = cursor.fetchone()[0]

            if not fk_enabled:
                result.add_error(
                    ValidationError(
                        severity="warning",
                        category="foreign_keys_disabled",
                        message="Foreign key constraints are not enabled",
                        details="Data integrity may be compromised",
                        repairable=True,
                    )
                )

            return True

        except Exception as e:
            result.add_error(
                ValidationError(
                    severity="critical",
                    category="database_error",
                    message=f"Database integrity check failed: {e}",
                    repairable=False,
                )
            )
            return False

    def _validate_workflow_state(self, workflow: WorkflowState, result: ValidationResult) -> None:
        """Validate a single workflow state."""
        # Check workflow ID
        if not workflow.workflow_id:
            result.add_error(
                ValidationError(
                    severity="critical",
                    category="invalid_workflow_id",
                    message="Workflow has empty ID",
                    affected_entity=workflow.workflow_id,
                    repairable=False,
                )
            )

        # Check timestamps
        if not workflow.started_at:
            result.add_error(
                ValidationError(
                    severity="critical",
                    category="missing_timestamp",
                    message="Workflow missing started_at timestamp",
                    affected_entity=workflow.workflow_id,
                    repairable=True,
                )
            )

        if not workflow.updated_at:
            result.add_error(
                ValidationError(
                    severity="warning",
                    category="missing_timestamp",
                    message="Workflow missing updated_at timestamp",
                    affected_entity=workflow.workflow_id,
                    repairable=True,
                )
            )

        # Check counts consistency
        total_processed = workflow.processed_files + workflow.failed_files
        if total_processed > workflow.total_files:
            result.add_error(
                ValidationError(
                    severity="critical",
                    category="inconsistent_counts",
                    message=f"Processed + failed count ({total_processed}) exceeds total files ({workflow.total_files})",
                    affected_entity=workflow.workflow_id,
                    repairable=True,
                )
            )

        # Validate metadata JSON
        if workflow.metadata:
            try:
                json.dumps(workflow.metadata)
            except Exception as e:
                result.add_error(
                    ValidationError(
                        severity="warning",
                        category="invalid_metadata",
                        message=f"Workflow metadata is not JSON-serializable: {e}",
                        affected_entity=workflow.workflow_id,
                        repairable=False,
                    )
                )

    def _validate_workflow_files(self, workflow_id: str, result: ValidationResult) -> None:
        """Validate all files for a workflow."""
        try:
            conn = self.state_store._get_connection()

            # Get all file states for workflow
            cursor = conn.execute("SELECT * FROM file_states WHERE workflow_id = ?", (workflow_id,))
            files = cursor.fetchall()

            for file_row in files:
                file_path = file_row["file_path"]

                # Check file exists
                path = Path(file_path)
                if not path.exists():
                    result.add_error(
                        ValidationError(
                            severity="warning",
                            category="file_not_found",
                            message=f"File not found: {file_path}",
                            affected_entity=file_path,
                            details="File was processed but no longer exists",
                            repairable=False,
                        )
                    )

                # Check file state consistency
                status = FileStatus(file_row["status"])

                # If completed, should have result or completion time
                if status == FileStatus.COMPLETED:
                    if not file_row["result_json"] and not file_row["completed_at"]:
                        result.add_error(
                            ValidationError(
                                severity="warning",
                                category="inconsistent_file_state",
                                message=f"Completed file missing result and completion time: {file_path}",
                                affected_entity=file_path,
                                repairable=True,
                            )
                        )

                # If failed, should have error message
                if status == FileStatus.FAILED and not file_row["last_error"]:
                    result.add_error(
                        ValidationError(
                            severity="warning",
                            category="inconsistent_file_state",
                            message=f"Failed file missing error message: {file_path}",
                            affected_entity=file_path,
                            repairable=True,
                        )
                    )

        except Exception as e:
            result.add_error(
                ValidationError(
                    severity="critical",
                    category="file_validation_error",
                    message=f"Failed to validate workflow files: {e}",
                    affected_entity=workflow_id,
                    repairable=False,
                )
            )

    def _validate_workflow_checkpoints(self, workflow_id: str, result: ValidationResult) -> None:
        """Validate all checkpoints for a workflow."""
        try:
            checkpoints = self.state_store.get_workflow_checkpoints(workflow_id)

            for checkpoint in checkpoints:
                self._validate_checkpoint_data(checkpoint, result)

        except Exception as e:
            result.add_error(
                ValidationError(
                    severity="critical",
                    category="checkpoint_validation_error",
                    message=f"Failed to validate workflow checkpoints: {e}",
                    affected_entity=workflow_id,
                    repairable=False,
                )
            )

    def _validate_checkpoint_data(
        self, checkpoint: CheckpointData, result: ValidationResult
    ) -> None:
        """Validate checkpoint data structure."""
        # Check batch ID
        if not checkpoint.batch_id:
            result.add_error(
                ValidationError(
                    severity="critical",
                    category="invalid_checkpoint_id",
                    message="Checkpoint has empty batch_id",
                    affected_entity=checkpoint.batch_id,
                    repairable=False,
                )
            )

        # Check workflow ID
        if not checkpoint.workflow_id:
            result.add_error(
                ValidationError(
                    severity="critical",
                    category="orphaned_checkpoint",
                    message="Checkpoint has empty workflow_id",
                    affected_entity=checkpoint.batch_id,
                    repairable=False,
                )
            )

        # Check results JSON is valid
        if checkpoint.results_json:
            try:
                json.loads(checkpoint.results_json)
            except json.JSONDecodeError as e:
                result.add_error(
                    ValidationError(
                        severity="critical",
                        category="invalid_results_json",
                        message=f"Checkpoint results_json is invalid: {e}",
                        affected_entity=checkpoint.batch_id,
                        repairable=False,
                    )
                )

        # Check processed/failed files consistency
        if checkpoint.processed_files is None:
            result.add_error(
                ValidationError(
                    severity="critical",
                    category="invalid_file_list",
                    message="Checkpoint has None for processed_files",
                    affected_entity=checkpoint.batch_id,
                    repairable=False,
                )
            )

        if checkpoint.failed_files is None:
            result.add_error(
                ValidationError(
                    severity="critical",
                    category="invalid_file_list",
                    message="Checkpoint has None for failed_files",
                    affected_entity=checkpoint.batch_id,
                    repairable=False,
                )
            )

    def _validate_orphaned_records(self, result: ValidationResult) -> None:
        """Validate no orphaned records exist."""
        try:
            conn = self.state_store._get_connection()

            # Check for file states with non-existent workflows
            cursor = conn.execute(
                """
                SELECT fs.file_path, fs.workflow_id
                FROM file_states fs
                LEFT JOIN workflow_states ws ON fs.workflow_id = ws.workflow_id
                WHERE ws.workflow_id IS NULL
            """
            )

            orphaned_files = cursor.fetchall()
            for file_row in orphaned_files:
                result.add_error(
                    ValidationError(
                        severity="warning",
                        category="orphaned_file_state",
                        message=f"File state references non-existent workflow: {file_row['workflow_id']}",
                        affected_entity=file_row["file_path"],
                        details=f"File {file_row['file_path']} has workflow_id={file_row['workflow_id']} but workflow not found",
                        repairable=True,
                    )
                )

            # Check for checkpoints with non-existent workflows
            cursor = conn.execute(
                """
                SELECT cp.batch_id, cp.workflow_id
                FROM checkpoints cp
                LEFT JOIN workflow_states ws ON cp.workflow_id = ws.workflow_id
                WHERE ws.workflow_id IS NULL
            """
            )

            orphaned_checkpoints = cursor.fetchall()
            for checkpoint_row in orphaned_checkpoints:
                result.add_error(
                    ValidationError(
                        severity="warning",
                        category="orphaned_checkpoint",
                        message=f"Checkpoint references non-existent workflow: {checkpoint_row['workflow_id']}",
                        affected_entity=checkpoint_row["batch_id"],
                        details=f"Checkpoint {checkpoint_row['batch_id']} has workflow_id={checkpoint_row['workflow_id']} but workflow not found",
                        repairable=True,
                    )
                )

        except Exception as e:
            result.add_error(
                ValidationError(
                    severity="critical",
                    category="orphan_validation_error",
                    message=f"Failed to validate orphaned records: {e}",
                    repairable=False,
                )
            )

    def _repair_orphaned_file_state(self, error: ValidationError) -> bool:
        """Repair orphaned file state by deleting it."""
        try:
            conn = self.state_store._get_connection()
            cursor = conn.execute(
                "DELETE FROM file_states WHERE file_path = ?", (error.affected_entity,)
            )
            conn.commit()

            if cursor.rowcount > 0:
                logger.info(f"Repaired orphaned file state: {error.affected_entity}")
                return True

            return False

        except Exception as e:
            logger.error(f"Failed to repair orphaned file state: {e}")
            return False

    def _repair_inconsistent_counts(self, error: ValidationError) -> bool:
        """Repair inconsistent workflow counts by recalculating from file states."""
        try:
            workflow_id = error.affected_entity
            if not workflow_id:
                return False

            conn = self.state_store._get_connection()

            # Recalculate counts from file states
            cursor = conn.execute(
                """
                SELECT
                    COUNT(*) FILTER (WHERE status = 'completed') as completed,
                    COUNT(*) FILTER (WHERE status = 'failed') as failed
                FROM file_states
                WHERE workflow_id = ?
            """,
                (workflow_id,),
            )

            row = cursor.fetchone()
            if row:
                new_processed = row[0] or 0
                new_failed = row[1] or 0

                # Update workflow state
                conn.execute(
                    """
                    UPDATE workflow_states
                    SET processed_files = ?, failed_files = ?
                    WHERE workflow_id = ?
                """,
                    (new_processed, new_failed, workflow_id),
                )
                conn.commit()

                logger.info(
                    f"Repaired inconsistent counts for {workflow_id}: {new_processed} processed, {new_failed} failed"
                )
                return True

            return False

        except Exception as e:
            logger.error(f"Failed to repair inconsistent counts: {e}")
            return False

    def _repair_missing_timestamp(self, error: ValidationError) -> bool:
        """Repair missing timestamp by setting it to now."""
        try:
            if (
                error.category == "missing_timestamp"
                and error.details
                and "started_at" in error.details
            ):
                workflow_id = error.affected_entity
                if not workflow_id:
                    return False

                conn = self.state_store._get_connection()
                now = datetime.now(timezone.utc).isoformat()

                conn.execute(
                    "UPDATE workflow_states SET started_at = ? WHERE workflow_id = ?",
                    (now, workflow_id),
                )
                conn.commit()

                logger.info(f"Repaired missing started_at for {workflow_id}")
                return True

            return False

        except Exception as e:
            logger.error(f"Failed to repair missing timestamp: {e}")
            return False
