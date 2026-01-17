"""
ISO/IEC 17025 compliance and quality control modules for forensic laboratory accreditation.

This package implements ISO/IEC 17025:2017 compliance requirements for:
- Methodology validation (Clause 7.2)
- Quality control procedures (Clause 7.7)
- Tool verification and calibration (Clause 6.4)

Modules:
    methodology_validator: Method validation protocols
    quality_control: QC procedures and control charts
    tool_calibration: Tool calibration and verification management

TAG: [FORENSIC-EVIDENCE-001]
Reference: SPEC-FORENSIC-EVIDENCE-001
Compliance: ISO/IEC 17025:2017
"""

# Methodology validation
from voice_man.forensics.compliance.methodology_validator import (
    MethodologyValidator,
    ValidationCriteria,
    ValidationParameters,
    ValidationResult,
    validate_audio_analysis_method,
    validate_crime_language_detection,
)

# Quality control
from voice_man.forensics.compliance.quality_control import (
    QualityControlManager,
    QCRecord,
    QCStatus,
    WestgardRule,
    run_daily_quality_control,
    get_control_chart_status,
)

# Tool calibration
from voice_man.forensics.compliance.tool_calibration import (
    ToolCalibrationManager,
    ToolInfo,
    CalibrationRecord,
    CalibrationStatus,
    register_forensic_tool,
    check_tool_calibration_status,
    get_tools_due_for_calibration,
)

__all__ = [
    # Methodology validation
    "MethodologyValidator",
    "ValidationCriteria",
    "ValidationParameters",
    "ValidationResult",
    "validate_audio_analysis_method",
    "validate_crime_language_detection",
    # Quality control
    "QualityControlManager",
    "QCRecord",
    "QCStatus",
    "WestgardRule",
    "run_daily_quality_control",
    "get_control_chart_status",
    # Tool calibration
    "ToolCalibrationManager",
    "ToolInfo",
    "CalibrationRecord",
    "CalibrationStatus",
    "register_forensic_tool",
    "check_tool_calibration_status",
    "get_tools_due_for_calibration",
]
