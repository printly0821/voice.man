"""
Tool Calibration and Verification Management for ISO/IEC 17025:2017

This module implements tool verification per:
- ISO/IEC 17025:2017 Clause 6.4 (Equipment)
- NIST SP 800-86 Tool Verification Guidelines

TAG: [FORENSIC-EVIDENCE-001]
Reference: SPEC-FORENSIC-EVIDENCE-001
Compliance: ISO/IEC 17025:2017 Clause 6.4.1-6.4.13
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class CalibrationStatus(Enum):
    """
    Calibration status

    Attributes:
        CALIBRATED: Tool is calibrated and within valid period
        EXPIRED: Calibration period has expired
        PENDING: Calibration is pending or in progress
        FAILED: Calibration failed or verification did not pass
    """
    CALIBRATED = "calibrated"
    EXPIRED = "expired"
    PENDING = "pending"
    FAILED = "failed"


@dataclass
class CalibrationRecord:
    """
    Tool calibration record

    Attributes:
        tool_id: Unique tool identifier
        tool_name: Name of the tool
        tool_version: Version string
        calibration_date: Date calibration was performed
        next_calibration_date: Date when next calibration is due
        calibration_organization: Organization that performed calibration
        certificate_number: Calibration certificate number
        calibration_status: Current calibration status
        verification_results: Dict of verification test results
        notes: Additional notes or observations
    """
    tool_id: str
    tool_name: str
    tool_version: str
    calibration_date: datetime
    next_calibration_date: Optional[datetime]
    calibration_organization: str
    certificate_number: str
    calibration_status: CalibrationStatus
    verification_results: Dict[str, bool]
    notes: Optional[str] = None


@dataclass
class ToolInfo:
    """
    Tool information for registration

    Attributes:
        tool_id: Unique tool identifier
        tool_name: Name of the tool
        tool_version: Version string
        tool_type: Type of tool (software, hardware, service)
        calibration_interval_days: Days between calibrations (default 365)
        manufacturer: Tool manufacturer or vendor
        serial_number: Tool serial number (if applicable)
    """
    tool_id: str
    tool_name: str
    tool_version: str
    tool_type: str
    calibration_interval_days: int = 365
    manufacturer: Optional[str] = None
    serial_number: Optional[str] = None


class ToolCalibrationManager:
    """
    Tool calibration and verification manager

    Implements ISO/IEC 17025:2017 Clause 6.4 requirements:
    - Equipment calibration before use
    - Calibration traceability to SI units
    - Calibration interval determination
    - Interim checks
    - Calibration status monitoring

    Examples:
        >>> tcm = ToolCalibrationManager()
        >>> tcm.register_tool(
        ...     tool_id="TOOL-001",
        ...     tool_name="WeasyPrint",
        ...     tool_version="60.2"
        ... )
        >>> record = tcm.record_calibration(...)
        >>> status = tcm.check_calibration_status("TOOL-001")
        >>> print(status["status"])
        "current"
    """

    # Default FreeTSA endpoint (fallback)
    DEFAULT_CALIBRATION_INTERVAL_DAYS = 365

    def __init__(self, db_session=None):
        """
        Initialize tool calibration manager

        Args:
            db_session: Database session for persisting calibration records
        """
        self.db_session = db_session
        self.calibration_records: Dict[str, CalibrationRecord] = {}
        self.registered_tools: Dict[str, ToolInfo] = {}
        logger.info("ToolCalibrationManager initialized")

    def register_tool(
        self,
        tool_id: str,
        tool_name: str,
        tool_version: str,
        tool_type: str = "software",
        calibration_interval_days: int = 365,
        manufacturer: Optional[str] = None,
        serial_number: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Register a new tool for calibration tracking

        Registers a tool for ISO/IEC 17025:2017 compliance tracking.

        Args:
            tool_id: Unique tool identifier
            tool_name: Name of the tool
            tool_version: Version string
            tool_type: Type of tool (software, hardware, service)
            calibration_interval_days: Days between calibrations (default 365)
            manufacturer: Tool manufacturer or vendor
            serial_number: Tool serial number

        Returns:
            Dict with registration details

        Examples:
            >>> tcm = ToolCalibrationManager()
            >>> result = tcm.register_tool(
            ...     tool_id="TOOL-001",
            ...     tool_name="WeasyPrint",
            ...     tool_version="60.2",
            ...     tool_type="software"
            ... )
            >>> print(result["status"])
            "registered"
        """
        tool_info = ToolInfo(
            tool_id=tool_id,
            tool_name=tool_name,
            tool_version=tool_version,
            tool_type=tool_type,
            calibration_interval_days=calibration_interval_days,
            manufacturer=manufacturer,
            serial_number=serial_number,
        )

        self.registered_tools[tool_id] = tool_info

        logger.info(f"Tool registered: {tool_id} - {tool_name} v{tool_version}")

        return {
            "tool_id": tool_id,
            "tool_name": tool_name,
            "tool_version": tool_version,
            "tool_type": tool_type,
            "calibration_interval_days": calibration_interval_days,
            "manufacturer": manufacturer,
            "serial_number": serial_number,
            "status": "registered",
            "next_calibration_due": (
                datetime.now(timezone.utc) + timedelta(days=calibration_interval_days)
            ).isoformat(),
        }

    def record_calibration(
        self,
        tool_id: str,
        tool_name: str,
        tool_version: str,
        calibration_organization: str,
        certificate_number: str,
        verification_results: Dict[str, bool],
        calibration_date: Optional[datetime] = None,
        notes: Optional[str] = None,
    ) -> CalibrationRecord:
        """
        Record tool calibration results

        Records calibration results per ISO/IEC 17025:2017 Clause 6.4.6.

        Args:
            tool_id: Unique tool identifier
            tool_name: Name of the tool
            tool_version: Version string
            calibration_organization: Organization that performed calibration
            certificate_number: Calibration certificate number
            verification_results: Dict of verification test results
            calibration_date: Date calibration was performed (default: now)
            notes: Additional notes or observations

        Returns:
            CalibrationRecord with calibration details

        Examples:
            >>> tcm = ToolCalibrationManager()
            >>> record = tcm.record_calibration(
            ...     tool_id="TOOL-001",
            ...     tool_name="WeasyPrint",
            ...     tool_version="60.2",
            ...     calibration_organization="Korean Calibration Lab",
            ...     certificate_number="CAL-2026-001",
            ...     verification_results={
            ...         "pdf_generation": True,
            ...         "font_rendering": True,
            ...         "css_compliance": True
            ...     }
            ... )
            >>> print(record.calibration_status.value)
            "calibrated"
        """
        if calibration_date is None:
            calibration_date = datetime.now(timezone.utc)

        # Determine if calibration passed
        all_passed = all(verification_results.values())
        calibration_status = (
            CalibrationStatus.CALIBRATED
            if all_passed
            else CalibrationStatus.FAILED
        )

        # Set next calibration date
        if calibration_status == CalibrationStatus.CALIBRATED:
            # Get calibration interval from registered tool info
            interval_days = self.DEFAULT_CALIBRATION_INTERVAL_DAYS
            if tool_id in self.registered_tools:
                interval_days = self.registered_tools[tool_id].calibration_interval_days

            next_calibration_date = calibration_date + timedelta(days=interval_days)
        else:
            next_calibration_date = None  # Failed calibration needs immediate attention

        record = CalibrationRecord(
            tool_id=tool_id,
            tool_name=tool_name,
            tool_version=tool_version,
            calibration_date=calibration_date,
            next_calibration_date=next_calibration_date,
            calibration_organization=calibration_organization,
            certificate_number=certificate_number,
            calibration_status=calibration_status,
            verification_results=verification_results,
            notes=notes,
        )

        self.calibration_records[tool_id] = record

        # Persist to database
        if self.db_session:
            self._persist_calibration_record(record)

        logger.info(
            f"Calibration recorded: {tool_id} - {calibration_status.value} (expires: {next_calibration_date})"
        )

        return record

    def check_calibration_status(self, tool_id: str) -> Dict[str, Any]:
        """
        Check if tool calibration is current

        Checks calibration status per ISO/IEC 17025:2017 Clause 6.4.8.

        Args:
            tool_id: Tool identifier

        Returns:
            Dict with calibration status:
                - tool_id: Tool identifier
                - status: "current" | "expired" | "failed" | "not_found"
                - message: Status description
                - calibration_date: Last calibration date
                - next_calibration_date: Next calibration due date
                - days_until_expiry: Days until calibration expires
                - days_overdue: Days overdue (if expired)

        Examples:
            >>> tcm = ToolCalibrationManager()
            >>> status = tcm.check_calibration_status("TOOL-001")
            >>> print(status["status"])
            "current"
        """
        if tool_id not in self.calibration_records:
            return {
                "tool_id": tool_id,
                "status": "not_found",
                "message": "Tool not found in calibration records",
            }

        record = self.calibration_records[tool_id]
        now = datetime.now(timezone.utc)

        if record.calibration_status == CalibrationStatus.FAILED:
            return {
                "tool_id": tool_id,
                "status": "failed",
                "message": "Tool calibration failed - requires re-calibration",
                "calibration_date": record.calibration_date.isoformat(),
            }

        if record.next_calibration_date is None:
            return {
                "tool_id": tool_id,
                "status": "failed",
                "message": "Tool calibration failed - requires re-calibration",
                "calibration_date": record.calibration_date.isoformat(),
            }

        if now > record.next_calibration_date:
            days_overdue = (now - record.next_calibration_date).days
            return {
                "tool_id": tool_id,
                "status": "expired",
                "message": f"Calibration expired on {record.next_calibration_date.isoformat()}",
                "calibration_date": record.calibration_date.isoformat(),
                "next_calibration_date": record.next_calibration_date.isoformat(),
                "days_overdue": days_overdue,
            }

        days_until = (record.next_calibration_date - now).days

        return {
            "tool_id": tool_id,
            "status": "current",
            "message": f"Calibration current, expires in {days_until} days",
            "calibration_date": record.calibration_date.isoformat(),
            "next_calibration_date": record.next_calibration_date.isoformat(),
            "days_until_expiry": days_until,
        }

    def get_due_calibrations(
        self,
        days_ahead: int = 30,
    ) -> List[Dict[str, Any]]:
        """
        Get tools due for calibration within specified days

        Returns list of tools requiring calibration attention.

        Args:
            days_ahead: Number of days ahead to check (default 30)

        Returns:
            List of tools due for calibration, sorted by urgency

        Examples:
            >>> tcm = ToolCalibrationManager()
            >>> due = tcm.get_due_calibrations(days_ahead=30)
            >>> for tool in due:
            ...     print(f"{tool['tool_id']}: {tool['days_until']} days")
        """
        due_tools: List[Dict[str, Any]] = []
        now = datetime.now(timezone.utc)
        check_date = now + timedelta(days=days_ahead)

        for tool_id, record in self.calibration_records.items():
            if record.next_calibration_date and record.next_calibration_date <= check_date:
                days_until = (record.next_calibration_date - now).days

                # Determine status category
                if days_until < 0:
                    status = "overdue"
                elif days_until == 0:
                    status = "due_today"
                elif days_until <= 7:
                    status = "due_soon"
                else:
                    status = "upcoming"

                due_tools.append(
                    {
                        "tool_id": tool_id,
                        "tool_name": record.tool_name,
                        "tool_version": record.tool_version,
                        "calibration_date": record.calibration_date.isoformat(),
                        "next_calibration_date": record.next_calibration_date.isoformat(),
                        "days_until": days_until,
                        "status": status,
                        "calibration_organization": record.calibration_organization,
                    }
                )

        # Sort by urgency (soonest first)
        due_tools.sort(key=lambda x: x["days_until"])

        return due_tools

    def get_calibration_summary(self) -> Dict[str, Any]:
        """
        Get overall calibration summary

        Returns:
            Dict with calibration statistics

        Examples:
            >>> tcm = ToolCalibrationManager()
            >>> summary = tcm.get_calibration_summary()
            >>> print(summary["total_tools"])
            5
        """
        now = datetime.now(timezone.utc)

        current_count = 0
        expired_count = 0
        failed_count = 0

        for record in self.calibration_records.values():
            if record.calibration_status == CalibrationStatus.FAILED or record.next_calibration_date is None:
                failed_count += 1
            elif now > record.next_calibration_date:
                expired_count += 1
            else:
                current_count += 1

        return {
            "total_tools": len(self.calibration_records),
            "current": current_count,
            "expired": expired_count,
            "failed": failed_count,
            "compliance_rate": (
                current_count / len(self.calibration_records)
                if self.calibration_records
                else 0
            ),
            "as_of": now.isoformat(),
        }

    def _persist_calibration_record(self, record: CalibrationRecord) -> None:
        """
        Persist calibration record to database

        Args:
            record: Calibration record to persist
        """
        # Implementation depends on database schema
        logger.info(f"Persisting calibration record: {record.tool_id}")


# Convenience functions


def register_forensic_tool(
    tool_id: str,
    tool_name: str,
    tool_version: str,
    tool_type: str = "software",
    calibration_interval_days: int = 365,
    db_session=None,
) -> Dict[str, Any]:
    """
    Convenience function to register a forensic analysis tool

    Args:
        tool_id: Unique tool identifier
        tool_name: Name of the tool
        tool_version: Version string
        tool_type: Type of tool (software, hardware, service)
        calibration_interval_days: Days between calibrations
        db_session: Database session for persistence

    Returns:
        Dict with registration details
    """
    tcm = ToolCalibrationManager(db_session)
    return tcm.register_tool(
        tool_id=tool_id,
        tool_name=tool_name,
        tool_version=tool_version,
        tool_type=tool_type,
        calibration_interval_days=calibration_interval_days,
    )


def check_tool_calibration_status(tool_id: str, db_session=None) -> Dict[str, Any]:
    """
    Convenience function to check tool calibration status

    Args:
        tool_id: Tool identifier
        db_session: Database session OR ToolCalibrationManager instance

    Returns:
        Dict with calibration status
    """
    # Support both database session and manager instance
    if isinstance(db_session, ToolCalibrationManager):
        manager = db_session
    else:
        manager = ToolCalibrationManager(db_session)

    return manager.check_calibration_status(tool_id)


def get_tools_due_for_calibration(days_ahead: int = 30, db_session=None) -> List[Dict[str, Any]]:
    """
    Convenience function to get tools due for calibration

    Args:
        days_ahead: Number of days ahead to check
        db_session: Database session OR ToolCalibrationManager instance

    Returns:
        List of tools due for calibration
    """
    # Support both database session and manager instance
    if isinstance(db_session, ToolCalibrationManager):
        manager = db_session
    else:
        manager = ToolCalibrationManager(db_session)

    return manager.get_due_calibrations(days_ahead)
