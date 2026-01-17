"""
Immutable Audit Logger for Forensic Evidence Access Tracking.

Implements append-only audit logging with hash chain integrity verification
according to ISO/IEC 27037 and Korean Criminal Procedure Law.

TAG: [FORENSIC-EVIDENCE-001]
"""

from datetime import datetime, timezone
import json
import hashlib
from typing import Any, Dict, List, Optional
from pathlib import Path


class ImmutableAuditLogger:
    """
    Immutable audit logger for forensic evidence access tracking.

    Provides append-only logging with hash chain verification for
    tamper detection and legal compliance.
    """

    def __init__(self, log_file_path: str):
        """
        Initialize the immutable audit logger.

        Args:
            log_file_path: Path to the audit log file
        """
        self.log_file_path = Path(log_file_path)
        self.last_hash: Optional[str] = None

        # Create log file if it doesn't exist
        if not self.log_file_path.exists():
            self.log_file_path.parent.mkdir(parents=True, exist_ok=True)
            self.log_file_path.touch()
        else:
            # Load last hash from existing log
            self._load_last_hash()

    def _load_last_hash(self) -> None:
        """Load the last event hash from the log file."""
        try:
            with open(self.log_file_path, "r") as f:
                lines = f.readlines()
                if lines:
                    last_line = lines[-1].strip()
                    if last_line:
                        last_event = json.loads(last_line)
                        self.last_hash = last_event.get("event_hash")
        except Exception:
            self.last_hash = None

    def log_event(
        self,
        event_type: str,
        asset_uuid: str,
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Log an audit event with hash chain.

        Args:
            event_type: Type of event (upload, analysis, report, etc.)
            asset_uuid: UUID of the asset being accessed
            user_id: User performing the action
            metadata: Additional event metadata

        Returns:
            Dict containing the logged event with hash
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        event = {
            "event_type": event_type,
            "asset_uuid": asset_uuid,
            "user_id": user_id,
            "timestamp": timestamp,
            "metadata": metadata or {},
            "previous_hash": self.last_hash,
        }

        # Compute event hash
        event_hash = self.compute_event_hash(event)
        event["event_hash"] = event_hash

        # Append to log file (append-only)
        with open(self.log_file_path, "a") as f:
            f.write(json.dumps(event) + "\n")

        # Update last hash for next event
        self.last_hash = event_hash

        return event

    def compute_event_hash(self, event_data: Dict[str, Any]) -> str:
        """
        Compute SHA-256 hash of event data.

        Args:
            event_data: Event data to hash

        Returns:
            str: SHA-256 hash in hexadecimal
        """
        # Create deterministic string representation
        hash_input = json.dumps(
            {
                "event_type": event_data["event_type"],
                "asset_uuid": event_data["asset_uuid"],
                "user_id": event_data["user_id"],
                "timestamp": event_data["timestamp"],
                "previous_hash": event_data["previous_hash"],
                "metadata": event_data.get("metadata", {}),
            },
            sort_keys=True,
        )

        return hashlib.sha256(hash_input.encode("utf-8")).hexdigest()

    def verify_integrity(self) -> bool:
        """
        Verify hash chain integrity of the entire log.

        Returns:
            bool: True if log integrity is intact, False if tampered
        """
        try:
            events = self.read_all_events()

            if not events:
                return True  # Empty log is valid

            # Verify first event has no previous hash
            if events[0]["previous_hash"] is not None:
                return False

            # Verify hash chain
            for i, event in enumerate(events):
                # Recompute hash
                computed_hash = self.compute_event_hash(event)

                if computed_hash != event["event_hash"]:
                    return False

                # Verify chain linkage
                if i > 0:
                    if event["previous_hash"] != events[i - 1]["event_hash"]:
                        return False

            return True

        except Exception:
            return False

    def read_all_events(self) -> List[Dict[str, Any]]:
        """
        Read all events from the log file.

        Returns:
            List of event dictionaries
        """
        events = []

        if not self.log_file_path.exists():
            return events

        with open(self.log_file_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))

        return events

    def get_events_by_asset(self, asset_uuid: str) -> List[Dict[str, Any]]:
        """
        Get all events for a specific asset.

        Args:
            asset_uuid: UUID of the asset

        Returns:
            List of events for the asset
        """
        all_events = self.read_all_events()
        return [e for e in all_events if e["asset_uuid"] == asset_uuid]
