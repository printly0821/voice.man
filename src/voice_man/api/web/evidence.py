"""
Forensic Evidence API Routes

포렌식 증거 관리 API 엔드포인트를 제공합니다.

TAG: [FORENSIC-EVIDENCE-001]
Reference: SPEC-FORENSIC-EVIDENCE-001
Compliance: ISO/IEC 27037, Korean Criminal Procedure Law Article 313(2)(3)
"""

from datetime import datetime
from typing import Annotated, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from voice_man.forensics.evidence import (
    DigitalSignatureService,
    ImmutableAuditLogger,
    TimestampService,
)
from voice_man.models.forensic.evidence_models import (
    CustodyEventType,
    CustodyLog,
    ForensicEvidence,
    MethodValidation,
    ToolVerification,
)
from voice_man.models.database import get_db

router = APIRouter(prefix="/api/v1/evidence", tags=["Forensic Evidence"])


# ============================================================================
# Request/Response Models
# ============================================================================


class EvidenceUploadRequest(BaseModel):
    """증거 업로드 요청 모델"""

    case_number: str = Field(..., description="사건번호", min_length=1, max_length=100)
    evidence_number: Optional[str] = Field(
        None, description="증거번호 (비어있으면 자동 생성)", max_length=100
    )
    collector_id: Optional[str] = Field(None, description="수집자 ID", max_length=100)
    collector_name: Optional[str] = Field(None, description="수집자 성명", max_length=255)
    collection_method: Optional[str] = Field(None, description="수집 방법", max_length=100)
    location: Optional[str] = Field(None, description="수집 장소", max_length=255)
    facility: Optional[str] = Field(None, description="시설명", max_length=255)


class EvidenceUploadResponse(BaseModel):
    """증거 업로드 응답 모델"""

    asset_uuid: str = Field(..., description="자산 UUID")
    evidence_number: str = Field(..., description="증거번호")
    file_hash: str = Field(..., description="SHA-256 해시")
    file_size: int = Field(..., description="파일 크기 (bytes)")
    digital_signature: str = Field(..., description="전자서명 (Base64)")
    timestamp_rfc3161: Optional[str] = Field(None, description="RFC 3161 타임스탬프")
    chain_of_custody_id: int = Field(..., description="Chain of Custody 이벤트 ID")
    created_at: datetime = Field(..., description="생성 시각")


class CustodyChainResponse(BaseModel):
    """Chain of Custody 조회 응답 모델"""

    asset_uuid: str
    evidence_number: str
    case_number: str
    events: list["CustodyEventResponse"]
    total_events: int
    verification_status: str  # "verified", "broken", "incomplete"


class CustodyEventResponse(BaseModel):
    """Chain of Custody 이벤트 모델"""

    id: int
    event_type: str
    timestamp: datetime
    timestamp_rfc3161: Optional[str]
    custodian_id: Optional[str]
    custodian_name: Optional[str]
    custodian_role: Optional[str]
    location: Optional[str]
    facility: Optional[str]
    file_hash: str
    previous_hash: Optional[str]
    current_hash: str
    digital_signature: Optional[str]
    transfer_reason: Optional[str]
    received_by: Optional[str]


class EvidenceVerificationRequest(BaseModel):
    """증거 무결성 검증 요청 모델"""

    asset_uuid: str = Field(..., description="자산 UUID")
    expected_hash: Optional[str] = Field(None, description="예상 해시값")


class EvidenceVerificationResponse(BaseModel):
    """증거 무결성 검증 응답 모델"""

    asset_uuid: str
    is_valid: bool
    hash_match: bool
    signature_valid: bool
    chain_intact: bool
    total_events: int
    first_event_time: Optional[datetime]
    last_event_time: Optional[datetime]
    issues: list[str]


class MethodValidationResponse(BaseModel):
    """방법론 검증 응답 모델"""

    id: int
    method_name: str
    method_version: str
    validation_date: datetime
    validation_status: str
    precision: Optional[float]
    recall: Optional[float]
    f1_score: Optional[float]
    uncertainty: Optional[float]
    confidence_level: Optional[float]


class ToolVerificationResponse(BaseModel):
    """도구 검증 응답 모델"""

    id: int
    tool_name: str
    tool_version: str
    verification_date: datetime
    verification_status: str
    calibration_date: Optional[datetime]
    next_calibration_date: Optional[datetime]
    success_rate: Optional[float]


# ============================================================================
# API Endpoints
# ============================================================================


@router.post("/upload", response_model=EvidenceUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_evidence(
    file: Annotated[UploadFile, File(description="증거 파일")],
    case_number: str,
    collector_id: Optional[str] = None,
    collector_name: Optional[str] = None,
    collection_method: Optional[str] = None,
    location: Optional[str] = None,
    facility: Optional[str] = None,
    db: Session = Depends(get_db),
) -> EvidenceUploadResponse:
    """
    포렌식 증거 업로드 엔드포인트

    파일 업로드 시 자동으로:
    1. SHA-256 해시 계산
    2. RSA-2048 전자서명 생성
    3. RFC 3161 타임스탬프 발급
    4. Chain of Custody 이벤트 기록
    5. 데이터베이스 저장

    Compliance: ISO/IEC 27037 Section 6 (Data acquisition)
    """
    # 파일 내용 읽기
    content = await file.read()
    file_size = len(content)

    # 서비스 초기화
    sig_service = DigitalSignatureService()
    ts_service = TimestampService()
    audit_logger = ImmutableAuditLogger()

    # 해시 계산
    import hashlib

    file_hash = hashlib.sha256(content).hexdigest()

    # 전자서명 생성
    signature_result = sig_service.sign_data(file_hash)
    digital_signature = signature_result["signature_base64"]

    # RFC 3161 타임스탬프 발급
    try:
        timestamp_result = ts_service.generate_timestamp(file_hash)
        timestamp_rfc3161 = timestamp_result.get("timestamp_token")
    except Exception as e:
        # TSA 실패 시 로컬 타임스탬프로 대체
        timestamp_rfc3161 = None
        # 감사 로그에 기록
        audit_logger.log_event(
            event_type="tsa_fallback",
            asset_uuid=str(uuid4()),
            details={"error": str(e), "file_hash": file_hash},
        )

    # 자산 UUID 생성
    asset_uuid = str(uuid4())

    # 증거번호 생성 (자동)
    from datetime import datetime

    evidence_number = f"EVIDENCE-{datetime.now().strftime('%Y%m%d')}-{uuid4().hex[:8].upper()}"

    # Chain of Custody 초기 이벤트 생성
    custody_event = CustodyLog(
        asset_uuid=asset_uuid,
        event_type=CustodyEventType.COLLECTION.value,
        timestamp=datetime.utcnow(),
        timestamp_rfc3161=timestamp_rfc3161,
        custodian_id=collector_id,
        custodian_name=collector_name,
        custodian_role="evidence_collector",
        location=location,
        facility=facility,
        file_hash=file_hash,
        file_size=file_size,
        previous_hash=None,
        current_hash=signature_result["current_hash"],
        digital_signature=digital_signature,
        case_number=case_number,
        evidence_number=evidence_number,
        event_metadata={
            "original_filename": file.filename,
            "collection_method": collection_method,
        },
    )

    # 데이터베이스 저장
    try:
        db.add(custody_event)

        # ForensicEvidence 메인 레코드 생성
        evidence = ForensicEvidence(
            asset_uuid=asset_uuid,
            original_filename=file.filename,
            file_hash=file_hash,
            file_size=file_size,
            file_path="",  # 실제 저장 경로는 별도 설정 필요
            evidence_type="audio",  # 자동 감지 가능
            media_type=file.content_type,
            case_number=case_number,
            evidence_number=evidence_number,
            collection_date=datetime.utcnow(),
            collector_id=collector_id,
            collector_name=collector_name,
            collection_method=collection_method,
            current_status="collected",
            current_location=facility,
            current_custodian=collector_name,
            digital_signature=digital_signature,
            timestamp_rfc3161=timestamp_rfc3161,
        )
        db.add(evidence)
        db.commit()
        db.refresh(custody_event)
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"데이터베이스 저장 실패: {str(e)}",
        )

    return EvidenceUploadResponse(
        asset_uuid=asset_uuid,
        evidence_number=evidence_number,
        file_hash=file_hash,
        file_size=file_size,
        digital_signature=digital_signature,
        timestamp_rfc3161=timestamp_rfc3161,
        chain_of_custody_id=custody_event.id,
        created_at=custody_event.timestamp,
    )


@router.get("/{asset_uuid}/chain", response_model=CustodyChainResponse)
async def get_custody_chain(
    asset_uuid: str,
    db: Session = Depends(get_db),
) -> CustodyChainResponse:
    """
    Chain of Custody 조회 엔드포인트

    지정된 자산의 전체 Chain of Custody를 반환합니다.
    해시 체인 무결성을 검증합니다.

    Compliance: ISO/IEC 27037 Section 7 (Evidence handling)
    """
    # 증거 조회
    evidence = db.query(ForensicEvidence).filter(ForensicEvidence.asset_uuid == asset_uuid).first()

    if not evidence:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"자산을 찾을 수 없습니다: {asset_uuid}",
        )

    # Chain of Custody 이벤트 조회
    events = (
        db.query(CustodyLog)
        .filter(CustodyLog.asset_uuid == asset_uuid)
        .order_by(CustodyLog.timestamp)
        .all()
    )

    # 해시 체인 검증
    chain_intact = True
    issues = []

    for i, event in enumerate(events):
        if i > 0:
            prev_event = events[i - 1]
            if event.previous_hash != prev_event.current_hash:
                chain_intact = False
                issues.append(
                    f"이벤트 {event.id}: 해시 체인 불일치 "
                    f"(previous_hash: {event.previous_hash} != "
                    f"expected: {prev_event.current_hash})"
                )

    verification_status = "verified" if chain_intact else "broken"

    return CustodyChainResponse(
        asset_uuid=asset_uuid,
        evidence_number=evidence.evidence_number,
        case_number=evidence.case_number,
        events=[
            CustodyEventResponse(
                id=e.id,
                event_type=e.event_type,
                timestamp=e.timestamp,
                timestamp_rfc3161=e.timestamp_rfc3161,
                custodian_id=e.custodian_id,
                custodian_name=e.custodian_name,
                custodian_role=e.custodian_role,
                location=e.location,
                facility=e.facility,
                file_hash=e.file_hash,
                previous_hash=e.previous_hash,
                current_hash=e.current_hash,
                digital_signature=e.digital_signature,
                transfer_reason=e.transfer_reason,
                received_by=e.received_by,
            )
            for e in events
        ],
        total_events=len(events),
        verification_status=verification_status,
    )


@router.post("/verify", response_model=EvidenceVerificationResponse)
async def verify_evidence(
    request: EvidenceVerificationRequest,
    db: Session = Depends(get_db),
) -> EvidenceVerificationResponse:
    """
    증거 무결성 검증 엔드포인트

    해시, 전자서명, Chain of Custody 무결성을 검증합니다.

    Compliance: ISO/IEC 27037 Section 8 (Data reporting)
    """
    # 증거 조회
    evidence = (
        db.query(ForensicEvidence).filter(ForensicEvidence.asset_uuid == request.asset_uuid).first()
    )

    if not evidence:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"자산을 찾을 수 없습니다: {request.asset_uuid}",
        )

    # Chain of Custody 이벤트 조회
    events = (
        db.query(CustodyLog)
        .filter(CustodyLog.asset_uuid == request.asset_uuid)
        .order_by(CustodyLog.timestamp)
        .all()
    )

    # 해시 검증
    hash_match = True
    if request.expected_hash and evidence.file_hash != request.expected_hash:
        hash_match = False

    # 전자서명 검증
    sig_service = DigitalSignatureService()
    signature_valid = sig_service.verify_signature(evidence.file_hash, evidence.digital_signature)

    # 해시 체인 검증
    chain_intact = True
    issues = []

    for i, event in enumerate(events):
        if i > 0:
            prev_event = events[i - 1]
            if event.previous_hash != prev_event.current_hash:
                chain_intact = False
                issues.append(
                    f"이벤트 {event.id}: 해시 체인 불일치 "
                    f"(previous_hash: {event.previous_hash} != "
                    f"expected: {prev_event.current_hash})"
                )

    if not hash_match:
        issues.append("파일 해시 불일치")

    if not signature_valid:
        issues.append("전자서명 검증 실패")

    if not chain_intact:
        issues.append("Chain of Custody 해시 체인 손상")

    is_valid = hash_match and signature_valid and chain_intact

    first_event_time = events[0].timestamp if events else None
    last_event_time = events[-1].timestamp if events else None

    return EvidenceVerificationResponse(
        asset_uuid=request.asset_uuid,
        is_valid=is_valid,
        hash_match=hash_match,
        signature_valid=signature_valid,
        chain_intact=chain_intact,
        total_events=len(events),
        first_event_time=first_event_time,
        last_event_time=last_event_time,
        issues=issues,
    )


@router.get("/{asset_uuid}/audit")
async def get_audit_trail(
    asset_uuid: str,
    db: Session = Depends(get_db),
):
    """
    감사 로그 조회 엔드포인트

    지정된 자산의 전체 감사 로그를 JSONL 형식으로 반환합니다.

    Compliance: ISO/IEC 27037 Section 7.2 (Audit trail)
    """
    # 증거 존재 확인
    evidence = db.query(ForensicEvidence).filter(ForensicEvidence.asset_uuid == asset_uuid).first()

    if not evidence:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"자산을 찾을 수 없습니다: {asset_uuid}",
        )

    # Chain of Custody 이벤트를 감사 로그로 반환
    events = (
        db.query(CustodyLog)
        .filter(CustodyLog.asset_uuid == asset_uuid)
        .order_by(CustodyLog.timestamp)
        .all()
    )

    audit_trail = []
    for event in events:
        audit_trail.append(
            {
                "event_id": event.id,
                "asset_uuid": event.asset_uuid,
                "event_type": event.event_type,
                "timestamp": event.timestamp.isoformat(),
                "timestamp_rfc3161": event.timestamp_rfc3161,
                "custodian_id": event.custodian_id,
                "custodian_name": event.custodian_name,
                "custodian_role": event.custodian_role,
                "location": event.location,
                "facility": event.facility,
                "file_hash": event.file_hash,
                "previous_hash": event.previous_hash,
                "current_hash": event.current_hash,
                "digital_signature": event.digital_signature,
                "case_number": event.case_number,
                "evidence_number": event.evidence_number,
                "metadata": event.event_metadata,
                "witnesses": event.witnesses,
            }
        )

    return {
        "asset_uuid": asset_uuid,
        "evidence_number": evidence.evidence_number,
        "case_number": evidence.case_number,
        "audit_trail": audit_trail,
        "total_events": len(audit_trail),
    }


@router.get("/method-validation", response_model=list[MethodValidationResponse])
async def list_method_validations(
    db: Session = Depends(get_db),
) -> list[MethodValidationResponse]:
    """
    방법론 검증 목록 조회 엔드포인트

    ISO/IEC 17025 Clause 7.2: Validation of methods

    Returns:
        MethodValidationResponse 목록
    """
    validations = db.query(MethodValidation).order_by(MethodValidation.validation_date.desc()).all()

    return [
        MethodValidationResponse(
            id=v.id,
            method_name=v.method_name,
            method_version=v.method_version,
            validation_date=v.validation_date,
            validation_status=v.validation_status,
            precision=v.precision,
            recall=v.recall,
            f1_score=v.f1_score,
            uncertainty=v.uncertainty,
            confidence_level=v.confidence_level,
        )
        for v in validations
    ]


@router.get("/tool-verification", response_model=list[ToolVerificationResponse])
async def list_tool_verifications(
    db: Session = Depends(get_db),
) -> list[ToolVerificationResponse]:
    """
    도구 검증 목록 조회 엔드포인트

    ISO/IEC 17025 Clause 6.4: Equipment

    Returns:
        ToolVerificationResponse 목록
    """
    verifications = (
        db.query(ToolVerification).order_by(ToolVerification.verification_date.desc()).all()
    )

    return [
        ToolVerificationResponse(
            id=v.id,
            tool_name=v.tool_name,
            tool_version=v.tool_version,
            verification_date=v.verification_date,
            verification_status=v.verification_status,
            calibration_date=v.calibration_date,
            next_calibration_date=v.next_calibration_date,
            success_rate=v.success_rate,
        )
        for v in verifications
    ]


# 라우터 등록은 main.py에서 수행합니다
# app.include_router(router)
