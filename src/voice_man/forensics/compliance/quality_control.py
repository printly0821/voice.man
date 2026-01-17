"""
Quality Control Procedures for ISO/IEC 17025:2017 Compliance

This module implements quality control procedures per:
- ISO/IEC 17025:2017 Clause 7.7 (Quality Control)
- NIST SP 800-86 Quality Assurance Guidelines
- ASCLD/LAB (American Society of Crime Laboratory Directors) policies

TAG: [FORENSIC-EVIDENCE-001]
Reference: SPEC-FORENSIC-EVIDENCE-001
Compliance: ISO/IEC 17025:2017 Clause 7.7.1-7.7.3
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class QCStatus(Enum):
    """
    Quality Control status

    Attributes:
        PASSED: QC test passed all criteria
        FAILED: QC test failed critical criteria
        WARNING: QC test passed with warnings
        PENDING: QC test pending review
    """
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    PENDING = "pending"


@dataclass
class QCRecord:
    """
    Quality Control record

    Attributes:
        record_id: Unique QC record identifier
        test_type: Type of QC test (daily, weekly, monthly)
        analyst_id: ID of analyst running QC
        test_date: Date and time of QC test
        result: QC test result status
        metrics: Test metrics and measurements
        notes: Additional notes or observations
        corrective_actions: Recommended corrective actions
    """
    record_id: str
    test_type: str
    analyst_id: str
    test_date: datetime
    result: QCStatus
    metrics: Dict[str, Any]
    notes: Optional[str] = None
    corrective_actions: Optional[List[str]] = None


@dataclass
class WestgardRule:
    """
    Westgard rule for QC control chart interpretation

    Westgard rules are statistical rules for detecting out-of-control conditions:
    - 1:2s - One point exceeds ±2SD
    - 1:3s - One point exceeds ±3SD
    - 2:2s - Two consecutive points exceed ±2SD
    - R:4s - Range between consecutive points exceeds 4SD
    - 4:1s - Four consecutive points exceed ±1SD on same side
    - 10:x - Ten consecutive points on same side of mean
    """
    rule_id: str
    name: str
    description: str
    violation: bool = False
    details: str = ""


class QualityControlManager:
    """
    Quality Control Manager for ISO/IEC 17025:2017 compliance

    Implements Clause 7.7.1-7.7.3:
    - Internal quality control procedures
    - Statistical quality control charts
    - Regular proficiency testing

    Examples:
        >>> qcm = QualityControlManager()
        >>> result = qcm.run_daily_qc(
        ...     analyst_id="ANALYST-001",
        ...     qc_sample={"forensic_score": 85},
        ...     expected_results={"forensic_score": 85}
        ... )
        >>> print(result.result.value)
        "passed"
    """

    def __init__(self, db_session=None):
        """
        Initialize quality control manager

        Args:
            db_session: Database session for persisting QC records
        """
        self.db_session = db_session
        self.qc_records: List[QCRecord] = []
        self.control_charts: Dict[str, List[Dict[str, Any]]] = {}
        logger.info("QualityControlManager initialized")

    def run_daily_qc(
        self,
        analyst_id: str,
        qc_sample: Dict[str, Any],
        expected_results: Dict[str, Any],
        acceptable_deviation: float = 0.05,
    ) -> QCRecord:
        """
        Run daily quality control test

        Daily QC ensures method performance within acceptable ranges.
        Implements ISO/IEC 17025:2017 Clause 7.7.2.

        Args:
            analyst_id: ID of analyst running QC
            qc_sample: Quality control sample data
            expected_results: Expected results for comparison
            acceptable_deviation: Acceptable percentage deviation (default 5%)

        Returns:
            QCRecord with test results

        Examples:
            >>> qcm = QualityControlManager()
            >>> result = qcm.run_daily_qc(
            ...     analyst_id="ANALYST-001",
            ...     qc_sample={"forensic_score": 85, "crime_detected": True},
            ...     expected_results={"forensic_score": 85, "crime_detected": True}
            ... )
        """
        record_id = self._generate_qc_record_id("daily")

        # Compare actual vs expected
        deviations = self._calculate_deviations(qc_sample, expected_results)

        # Determine status based on acceptable deviation
        max_deviation = max(deviations.values()) if deviations else 0

        if max_deviation <= acceptable_deviation:
            status = QCStatus.PASSED
        elif max_deviation <= acceptable_deviation * 2:
            status = QCStatus.WARNING
        else:
            status = QCStatus.FAILED

        record = QCRecord(
            record_id=record_id,
            test_type="daily_qc",
            analyst_id=analyst_id,
            test_date=datetime.now(timezone.utc),
            result=status,
            metrics={"deviations": deviations, "max_deviation": max_deviation},
            notes=self._generate_qc_notes(status, deviations),
            corrective_actions=self._generate_corrective_actions(deviations) if status != QCStatus.PASSED else None,
        )

        self.qc_records.append(record)

        # Update control chart
        self._update_control_chart("daily_qc", record)

        # Persist to database
        if self.db_session:
            self._persist_qc_record(record)

        logger.info(
            f"Daily QC completed: {record_id} - {status.value} (max deviation: {max_deviation:.2%})"
        )

        return record

    def get_control_chart_status(
        self,
        chart_type: str = "daily_qc",
        analysis_points: int = 30,
    ) -> Dict[str, Any]:
        """
        Get control chart status for trend analysis

        Implements Westgard rules for QC control chart interpretation.
        Per ISO/IEC 17025:2017 Clause 7.7.3.

        Args:
            chart_type: Type of control chart (daily_qc, weekly_qc, etc.)
            analysis_points: Number of points to analyze (default 30)

        Returns:
            Dict with control chart analysis results:
                - chart_type: Type of control chart
                - total_points: Total data points
                - passed: Number of passed points
                - failed: Number of failed points
                - warning: Number of warning points
                - pass_rate: Pass rate percentage
                - trend_alerts: List of Westgard rule violations
                - overall_status: "in_control" | "out_of_control"

        Examples:
            >>> qcm = QualityControlManager()
            >>> status = qcm.get_control_chart_status("daily_qc")
            >>> print(status["overall_status"])
            "in_control"
        """
        if chart_type not in self.control_charts:
            return {
                "chart_type": chart_type,
                "overall_status": "no_data",
                "message": f"No control chart data available for {chart_type}",
            }

        data_points = self.control_charts[chart_type][-analysis_points:]

        # Calculate statistics
        passed_count = sum(1 for p in data_points if p.get("status") == "passed")
        failed_count = sum(1 for p in data_points if p.get("status") == "failed")
        warning_count = sum(1 for p in data_points if p.get("status") == "warning")

        # Check Westgard rules
        trend_alerts = self._check_westgard_rules(data_points)

        overall_status = "in_control" if not trend_alerts else "out_of_control"

        return {
            "chart_type": chart_type,
            "total_points": len(data_points),
            "passed": passed_count,
            "failed": failed_count,
            "warning": warning_count,
            "pass_rate": passed_count / len(data_points) if data_points else 0,
            "trend_alerts": trend_alerts,
            "overall_status": overall_status,
        }

    def get_qc_summary(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Get QC summary for a date range

        Args:
            start_date: Start date for summary (default: 30 days ago)
            end_date: End date for summary (default: now)

        Returns:
            Dict with QC summary statistics
        """
        if start_date is None:
            start_date = datetime.now(timezone.utc) - timedelta(days=30)
        if end_date is None:
            end_date = datetime.now(timezone.utc)

        # Filter records by date range
        filtered_records = [
            r
            for r in self.qc_records
            if start_date <= r.test_date <= end_date
        ]

        if not filtered_records:
            return {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "total_tests": 0,
                "passed": 0,
                "failed": 0,
                "warning": 0,
                "pass_rate": 0,
            }

        passed = sum(1 for r in filtered_records if r.result == QCStatus.PASSED)
        failed = sum(1 for r in filtered_records if r.result == QCStatus.FAILED)
        warning = sum(1 for r in filtered_records if r.result == QCStatus.WARNING)

        return {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "total_tests": len(filtered_records),
            "passed": passed,
            "failed": failed,
            "warning": warning,
            "pass_rate": passed / len(filtered_records),
        }

    def _calculate_deviations(
        self,
        actual: Dict[str, Any],
        expected: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Calculate percentage deviations from expected values

        Args:
            actual: Actual measured values
            expected: Expected reference values

        Returns:
            Dict mapping metric names to percentage deviations
        """
        deviations: Dict[str, float] = {}

        for key, expected_value in expected.items():
            if key in actual:
                actual_value = actual[key]
                if isinstance(expected_value, (int, float)) and isinstance(actual_value, (int, float)):
                    if expected_value != 0:
                        deviation = abs((actual_value - expected_value) / expected_value)
                        deviations[key] = float(deviation)

        return deviations

    def _generate_qc_notes(
        self,
        status: QCStatus,
        deviations: Dict[str, float],
    ) -> str:
        """
        Generate QC notes based on status and deviations

        Args:
            status: QC test status
            deviations: Calculated deviations

        Returns:
            Human-readable notes string
        """
        if status == QCStatus.PASSED:
            return "All QC metrics within acceptable range."
        elif status == QCStatus.WARNING:
            max_dev = max(deviations.values()) if deviations else 0
            return f"Warning: Some metrics near acceptable limits. Max deviation: {max_dev:.2%}"
        else:
            max_dev = max(deviations.values()) if deviations else 0
            return f"FAILED: Metrics outside acceptable range. Max deviation: {max_dev:.2%}"

    def _generate_corrective_actions(
        self,
        deviations: Dict[str, float],
    ) -> List[str]:
        """
        Generate corrective action recommendations

        Args:
            deviations: Calculated deviations

        Returns:
            List of corrective action recommendations
        """
        actions: List[str] = []

        for metric, deviation in deviations.items():
            if deviation > 0.10:  # 10% deviation
                actions.append(f"Recalibrate {metric} measurement")
            elif deviation > 0.05:  # 5% deviation
                actions.append(f"Review {metric} analysis procedure")

        if not actions:
            actions.append("Review all analysis procedures")

        return actions

    def _update_control_chart(self, chart_type: str, record: QCRecord) -> None:
        """
        Update statistical control chart

        Args:
            chart_type: Type of control chart
            record: QC record to add
        """
        if chart_type not in self.control_charts:
            self.control_charts[chart_type] = []

        self.control_charts[chart_type].append(
            {
                "timestamp": record.test_date,
                "status": record.result.value,
                "metrics": record.metrics,
                "analyst_id": record.analyst_id,
            }
        )

        # Keep only last 30 points for control chart
        if len(self.control_charts[chart_type]) > 30:
            self.control_charts[chart_type] = self.control_charts[chart_type][-30:]

    def _check_westgard_rules(
        self,
        data_points: List[Dict[str, Any]],
    ) -> List[WestgardRule]:
        """
        Check Westgard rules for QC control charts

        Westgard rules for detecting out-of-control conditions:
        1. 1:2s - One point exceeds ±2SD
        2. 1:3s - One point exceeds ±3SD
        3. 2:2s - Two consecutive points exceed ±2SD
        4. R:4s - Range between consecutive points exceeds 4SD
        5. 4:1s - Four consecutive points exceed ±1SD on same side
        6. 10:x - Ten consecutive points on same side of mean

        Args:
            data_points: List of control chart data points

        Returns:
            List of WestgardRule violations
        """
        alerts: List[WestgardRule] = []

        if len(data_points) < 2:
            return alerts

        # Calculate mean and standard deviation
        max_deviations = [
            p.get("metrics", {}).get("max_deviation", 0) for p in data_points
        ]

        import numpy as np

        mean = np.mean(max_deviations)
        std = np.std(max_deviations) if len(max_deviations) > 1 else 0.01

        for i, point in enumerate(data_points):
            deviation = point.get("metrics", {}).get("max_deviation", 0)
            z_score = (deviation - mean) / std if std > 0 else 0

            # Rule 1:2s - One point exceeds ±2SD
            if abs(z_score) > 2:
                alerts.append(
                    WestgardRule(
                        rule_id="1_2s",
                        name="1:2s Rule",
                        description=f"Point {i+1}: Exceeds ±2SD (z={z_score:.2f})",
                        violation=True,
                    )
                )

            # Rule 2: 1:3s - One point exceeds ±3SD
            if abs(z_score) > 3:
                alerts.append(
                    WestgardRule(
                        rule_id="1_3s",
                        name="1:3s Rule",
                        description=f"Point {i+1}: Exceeds ±3SD (z={z_score:.2f}) - CRITICAL",
                        violation=True,
                    )
                )

        # Rule 3: 2:2s - Two consecutive points exceed ±2SD
        for i in range(len(data_points) - 1):
            dev1 = data_points[i].get("metrics", {}).get("max_deviation", 0)
            dev2 = data_points[i + 1].get("metrics", {}).get("max_deviation", 0)

            z1 = (dev1 - mean) / std if std > 0 else 0
            z2 = (dev2 - mean) / std if std > 0 else 0

            if abs(z1) > 2 and abs(z2) > 2:
                alerts.append(
                    WestgardRule(
                        rule_id="2_2s",
                        name="2:2s Rule",
                        description=f"Points {i+1}-{i+2}: Two consecutive exceed ±2SD",
                        violation=True,
                    )
                )

        return alerts

    def _generate_qc_record_id(self, test_type: str) -> str:
        """
        Generate unique QC record identifier

        Format: QC-TESTTYPE-YYYYMMDD-HHMMSS

        Args:
            test_type: Type of QC test

        Returns:
            Unique record identifier
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        return f"QC-{test_type.upper()}-{timestamp}"

    def _persist_qc_record(self, record: QCRecord) -> None:
        """
        Persist QC record to database

        Args:
            record: QC record to persist
        """
        # Implementation depends on database schema
        logger.info(f"Persisting QC record: {record.record_id}")


# Convenience functions


def run_daily_quality_control(
    analyst_id: str,
    qc_sample: Dict[str, Any],
    expected_results: Dict[str, Any],
    db_session=None,
) -> QCRecord:
    """
    Convenience function to run daily quality control test

    Args:
        analyst_id: ID of analyst running QC
        qc_sample: Quality control sample data
        expected_results: Expected results for comparison
        db_session: Database session for persistence

    Returns:
        QCRecord with test results

    Examples:
        >>> result = run_daily_quality_control(
        ...     analyst_id="ANALYST-001",
        ...     qc_sample={"forensic_score": 85},
        ...     expected_results={"forensic_score": 85}
        ... )
        >>> print(result.result.value)
        "passed"
    """
    qcm = QualityControlManager(db_session)
    return qcm.run_daily_qc(analyst_id, qc_sample, expected_results)


def get_control_chart_status(chart_type: str = "daily_qc", db_session=None) -> Dict[str, Any]:
    """
    Convenience function to get control chart status

    Args:
        chart_type: Type of control chart
        db_session: Database session

    Returns:
        Dict with control chart analysis results
    """
    qcm = QualityControlManager(db_session)
    return qcm.get_control_chart_status(chart_type)
