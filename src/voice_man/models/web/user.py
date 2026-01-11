"""
# TAG(SPEC-FORENSIC-WEB-001): 사용자 모델 및 RBAC 구현
# Phase: GREEN - 최소 구현

사용자 모델, 역할 기반 접근 제어(RBAC), 비밀번호 해싱을 구현합니다.
"""

from enum import Enum
from datetime import datetime, timezone
from uuid import UUID, uuid4
from dataclasses import dataclass, field
import bcrypt
import base64


# =============================================================================
# UserRole 열거형
# =============================================================================


class UserRole(str, Enum):
    """사용자 역할 열거형"""

    LAWYER = "lawyer"  # 변호사: 모든 권한
    CLIENT = "client"  # 의뢰인: 읽기 전용
    EXPERT = "expert"  # 포렌식 전문가: 포렌식 분석 전용


# =============================================================================
# Permission 열거형
# =============================================================================


class Permission(str, Enum):
    """시스템 권한 열거형"""

    # 고소장 권한
    CREATE_COMPLAINT = "create_complaint"
    READ_COMPLAINT = "read_complaint"
    UPDATE_COMPLAINT = "update_complaint"
    DELETE_COMPLAINT = "delete_complaint"

    # 증거 권한
    CREATE_EVIDENCE = "create_evidence"
    READ_EVIDENCE = "read_evidence"
    DELETE_EVIDENCE = "delete_evidence"

    # 포렌식 분석 권한
    RUN_FORENSIC_ANALYSIS = "run_forensic_analysis"

    # 감사 로그 권한
    READ_AUDIT_LOG = "read_audit_log"


# =============================================================================
# RBAC 권한 매트릭스
# =============================================================================

# 역할별 권한 매핑
ROLE_PERMISSIONS: dict[UserRole, set[Permission]] = {
    UserRole.LAWYER: {
        # 고소장 모든 권한
        Permission.CREATE_COMPLAINT,
        Permission.READ_COMPLAINT,
        Permission.UPDATE_COMPLAINT,
        Permission.DELETE_COMPLAINT,
        # 증거 모든 권한
        Permission.CREATE_EVIDENCE,
        Permission.READ_EVIDENCE,
        Permission.DELETE_EVIDENCE,
        # 포렌식 분석
        Permission.RUN_FORENSIC_ANALYSIS,
        # 감사 로그
        Permission.READ_AUDIT_LOG,
    },
    UserRole.CLIENT: {
        # 고소장 읽기만
        Permission.READ_COMPLAINT,
        # 증거 읽기만
        Permission.READ_EVIDENCE,
        # 포렌식 분석 (자신의 증거 확인용)
        Permission.RUN_FORENSIC_ANALYSIS,
    },
    UserRole.EXPERT: {
        # 증거 읽기
        Permission.READ_EVIDENCE,
        # 포렌식 분석
        Permission.RUN_FORENSIC_ANALYSIS,
    },
}


# =============================================================================
# PasswordHasher: 비밀번호 해싱
# =============================================================================


class PasswordHasher:
    """
    비밀번호 해싱 및 검증 클래스

    bcrypt 알고리즘을 사용하여 비밀번호를 안전하게 해싱합니다.
    """

    # bcrypt의 72바이트 제한을 해결하기 위해 base64 인코딩 사용
    def _encode_password(self, password: str) -> bytes:
        """비밀번호를 base64로 인코딩하여 72바이트 제한을 우회합니다."""
        return base64.b64encode(password.encode("utf-8"))

    def _decode_password(self, encoded: bytes) -> str:
        """base64로 인코딩된 비밀번호를 디코딩합니다."""
        return base64.b64decode(encoded).decode("utf-8")

    def hash_password(self, password: str) -> str:
        """
        평문 비밀번호를 해싱합니다.

        Args:
            password: 평문 비밀번호

        Returns:
            bcrypt 해시된 비밀번호
        """
        # bcrypt는 72바이트 제한이 있으므로 base64 인코딩 사용
        encoded = self._encode_password(password)
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(encoded, salt)
        # base64 인코딩된 해시를 다시 base64로 저장하여 디코딩 가능하도록 함
        return base64.b64encode(hashed).decode("ascii")

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """
        평문 비밀번호와 해시된 비밀번호를 비교합니다.

        Args:
            plain_password: 평문 비밀번호
            hashed_password: 해시된 비밀번호

        Returns:
            일치하면 True, 아니면 False
        """
        try:
            # 저장된 해시를 디코딩
            hashed_bytes = base64.b64decode(hashed_password.encode("ascii"))
            # 입력 비밀번호를 인코딩
            encoded = self._encode_password(plain_password)
            return bcrypt.checkpw(encoded, hashed_bytes)
        except Exception:
            return False


# =============================================================================
# RBACChecker: 역할 기반 접근 제어
# =============================================================================


class RBACChecker:
    """
    역할 기반 접근 제어 검사기

    사용자 역할에 따른 권한을 확인합니다.
    """

    def has_permission(self, role: UserRole, permission: Permission) -> bool:
        """
        역할이 특정 권한을 가지고 있는지 확인합니다.

        Args:
            role: 사용자 역할
            permission: 확인할 권한

        Returns:
            권한이 있으면 True, 없으면 False
        """
        return permission in ROLE_PERMISSIONS.get(role, set())


# =============================================================================
# User 모델
# =============================================================================


@dataclass
class User:
    """
    사용자 모델

    Attributes:
        id: 사용자 고유 ID
        username: 사용자명 (고유)
        email: 이메일 주소 (고유)
        password_hash: 해시된 비밀번호
        full_name: 전체 이름 (선택적)
        role: 사용자 역할
        is_active: 활성 상태
        created_at: 생성 일시
        updated_at: 수정 일시
    """

    id: UUID = field(default_factory=uuid4)
    username: str = ""
    email: str = ""
    password_hash: str = ""
    full_name: str | None = None
    role: UserRole = UserRole.CLIENT
    is_active: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def verify_password(self, plain_password: str, hasher: PasswordHasher) -> bool:
        """
        평문 비밀번호를 검증합니다.

        Args:
            plain_password: 검증할 평문 비밀번호
            hasher: PasswordHasher 인스턴스

        Returns:
            비밀번호가 일치하면 True, 아니면 False
        """
        return hasher.verify_password(plain_password, self.password_hash)

    def has_permission(self, permission: Permission) -> bool:
        """
        사용자가 특정 권한을 가지고 있는지 확인합니다.

        Args:
            permission: 확인할 권한

        Returns:
            권한이 있으면 True, 없으면 False
        """
        checker = RBACChecker()
        return checker.has_permission(self.role, permission)

    def __repr__(self) -> str:
        """문자열 표현"""
        return f"User(id={self.id}, username={self.username}, email={self.email}, role={self.role.value})"
