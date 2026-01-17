"""
포렌식 증거 관리 ORM 모델

Chain of Custody, Method Validation, Tool Verification 모델을 정의합니다.

TAG: [FORENSIC-EVIDENCE-001]
Reference: SPEC-FORENSIC-EVIDENCE-001
Compliance: ISO/IEC 27037, ISO/IEC 17025, Korean Criminal Procedure Law Article 313(2)(3)
"""

from datetime import datetime
from typing import Optional
from enum import Enum

from sqlalchemy import String, Float, Integer, DateTime, ForeignKey, Text, JSON, Boolean
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from voice_man.models.database import Base


class CustodyEventType(Enum):
    """Chain of Custody 이벤트 유형"""

    COLLECTION = "collection"  # 증거 수집
    TRANSFER = "transfer"  # 증거 이관
    ANALYSIS = "analysis"  # 분석 수행
    STORAGE = "storage"  # 보관
    SUBMISSION = "submission"  # 법정 제출
    VERIFICATION = "verification"  # 무결성 검증
    ARCHIVAL = "archival"  # 보존


class CustodyLog(Base):
    """
    Chain of Custody 로그 모델

    증거 자산의 모든 이동과 처리 이력을 기록합니다.
    해시 체인을 통해 무결성을 보장합니다.

    Compliance: ISO/IEC 27037 Section 7 (Evidence handling)
    Reference: NIST SP 800-86 Chain of Custody requirements
    """

    __tablename__ = "custody_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    asset_uuid: Mapped[str] = mapped_column(String(36), index=True, nullable=False)

    # 이벤트 정보
    event_type: Mapped[str] = mapped_column(String(50), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    # RFC 3161 타임스탬프 (선택)
    timestamp_rfc3161: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # 담당자 정보
    custodian_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    custodian_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    custodian_role: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # 위치 정보
    location: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    facility: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    room: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # 무결성 보장 (해시 체인)
    file_hash: Mapped[str] = mapped_column(String(64), nullable=False)  # SHA-256
    file_size: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    previous_hash: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True
    )  # 이전 이벤트와 연결
    current_hash: Mapped[str] = mapped_column(String(64), nullable=False)  # 이벤트 자체의 해시

    # 전자서명
    digital_signature: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # 이관 정보 (선택)
    transfer_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    transfer_method: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    received_by: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # 법적 메타데이터
    case_number: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, index=True)
    evidence_number: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, index=True)

    # 추가 메타데이터 (JSON)
    event_metadata: Mapped[Optional[dict]] = mapped_column("metadata", JSON, nullable=True)

    # 증인 (선택)
    witnesses: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)

    # 타임스탬프
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class MethodValidation(Base):
    """
    분석 방법론 검증 모델

    ISO/IEC 17025 Clause 7.2: Validation of methods 요구사항을 충족합니다.

    Compliance: ISO/IEC 17025:2017 Clause 7.2
    Reference: UK Government Method Validation in Digital Forensics
    """

    __tablename__ = "method_validation"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # 방법론 식별자
    method_name: Mapped[str] = mapped_column(String(255), nullable=False)
    method_version: Mapped[str] = mapped_column(String(50), nullable=False)
    method_type: Mapped[str] = mapped_column(
        String(100), nullable=False
    )  # e.g., "audio_analysis", "speaker_identification"

    # 검증 정보
    validation_date: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    validator_id: Mapped[str] = mapped_column(String(100), nullable=False)
    validator_name: Mapped[str] = mapped_column(String(255), nullable=False)

    # 검증 항목 (ISO/IEC 17025 요구사항)
    documented_in_sop: Mapped[bool] = mapped_column(Boolean, default=False)
    validated_with_reference: Mapped[bool] = mapped_column(Boolean, default=False)
    uncertainty_calculated: Mapped[bool] = mapped_column(Boolean, default=False)
    detection_limit_determined: Mapped[bool] = mapped_column(Boolean, default=False)
    selectivity_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    robustness_tested: Mapped[bool] = mapped_column(Boolean, default=False)
    bias_verified: Mapped[bool] = mapped_column(Boolean, default=False)

    # 성능 메트릭
    precision: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    recall: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    f1_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    uncertainty: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # 통계적 신뢰구간 (Bootstrap CI)
    confidence_level: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # e.g., 0.95
    confidence_interval_lower: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    confidence_interval_upper: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # 검증 상태
    validation_status: Mapped[str] = mapped_column(
        String(50), default="pending"
    )  # pending, passed, failed

    # 첨부 파일 경로 (참고 자료)
    reference_document_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)

    # 메모
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # 시정 조치 (검증 실패 시)
    corrective_actions: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)

    # 타임스탬프
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )


class ToolVerification(Base):
    """
    도구 검증 모델

    ISO/IEC 17025 Clause 6.4: Equipment 요구사항을 충족합니다.
    포렌식 분석에 사용된 모든 도구의 검증 기록을 관리합니다.

    Compliance: ISO/IEC 17025:2017 Clause 6.4, 6.5
    Reference: ASCLD/LAB Forensic Scope of Accreditation
    """

    __tablename__ = "tool_verification"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # 도구 식별자
    tool_name: Mapped[str] = mapped_column(String(255), nullable=False)
    tool_version: Mapped[str] = mapped_column(String(100), nullable=False)
    tool_vendor: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    tool_type: Mapped[str] = mapped_column(
        String(100), nullable=False
    )  # e.g., "audio_analysis", "transcription"

    # 검증 정보
    verification_date: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    verifier_id: Mapped[str] = mapped_column(String(100), nullable=False)
    verifier_name: Mapped[str] = mapped_column(String(255), nullable=False)

    # 교정 정보 (Calibration)
    calibration_date: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    next_calibration_date: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    calibration_certificate_number: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True
    )
    calibration_organization: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # 검증 항목
    version_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    functionality_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    accuracy_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    reproducibility_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    license_valid: Mapped[bool] = mapped_column(Boolean, default=False)

    # 검증 결과
    test_case_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    passed_test_cases: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    success_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # 검증 상태
    verification_status: Mapped[str] = mapped_column(
        String(50), default="pending"
    )  # pending, passed, failed, expired

    # 첨부 파일
    verification_report_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)

    # 알려진 문제점 및 제한사항
    known_issues: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    limitations: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # 메모
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # 타임스탬프
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )


class ForensicEvidence(Base):
    """
    포렌식 증거 메인 모델

    증거 자산의 기본 정보와 Chain of Custody를 연결합니다.

    Reference: SWGDE Digital Evidence Guidelines
    Compliance: ISO/IEC 27037 Section 6 (Data acquisition)
    """

    __tablename__ = "forensic_evidence"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    asset_uuid: Mapped[str] = mapped_column(String(36), unique=True, index=True, nullable=False)

    # 파일 정보
    original_filename: Mapped[str] = mapped_column(String(512), nullable=False)
    file_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)
    file_path: Mapped[str] = mapped_column(String(512), nullable=False)

    # 증거 유형
    evidence_type: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # e.g., "audio", "video", "document"
    media_type: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)  # MIME type

    # 법적 메타데이터
    case_number: Mapped[str] = mapped_column(String(100), index=True, nullable=False)
    evidence_number: Mapped[str] = mapped_column(
        String(100), unique=True, index=True, nullable=False
    )

    # 수집 정보
    collection_date: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    collector_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    collector_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    collection_method: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # 현재 상태
    current_status: Mapped[str] = mapped_column(
        String(50), default="collected"
    )  # collected, analyzing, verified, submitted
    current_location: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    current_custodian: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # 무결성
    digital_signature: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    timestamp_rfc3161: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # 분석 결과 (연결)
    analysis_results: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # 타임스탬프
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # 관계
    custody_events: Mapped[list["CustodyLog"]] = relationship(
        "CustodyLog",
        foreign_keys=[CustodyLog.asset_uuid],
        order_by=CustodyLog.timestamp,
        cascade="all, delete-orphan",
    )
