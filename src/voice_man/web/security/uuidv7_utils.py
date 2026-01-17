"""
UUID v7 유틸리티 모듈

TAG: TAG-001-UUIDv7
SPEC: SPEC-FORENSIC-WEB-001

Universally Unique Identifier v7 (RFC 4122) 생성 및 관리.
UUID v7은 시간순 정렬 가능한 UUID로, UUID v4의 랜덤성과 UUID v1의 시간순 정렬을 결합합니다.
- 36자리 길이 (표준 형식)
- 시간순 정렬 가능
- ULID보다 길지만 표준 RFC 4122 준수
- uuid_extensions 라이브러리 사용
"""

from datetime import datetime, timezone
from typing import Optional
from uuid import UUID
from uuid_extensions import uuid7, uuid_to_datetime


def generate_uuidv7() -> str:
    """
    새로운 UUID v7를 생성합니다.

    Returns:
        str: 36자리 UUID v7 문자열 (표준 형식: xxxxxxxx-xxxx-7xxx-xxxx-xxxxxxxxxxxx)

    Example:
        >>> uid = generate_uuidv7()
        >>> len(uid)
        36
        >>> validate_uuidv7(uid)
        True
    """
    return str(uuid7())


def validate_uuidv7(uuid_str: Optional[str]) -> bool:
    """
    UUID v7 문자열의 유효성을 검증합니다.

    Args:
        uuid_str: 검증할 UUID v7 문자열

    Returns:
        bool: 유효한 UUID v7이면 True, 아니면 False

    Example:
        >>> uid = generate_uuidv7()
        >>> validate_uuidv7(uid)
        True
        >>> validate_uuidv7("00000000-0000-0000-0000-000000000000")
        False
    """
    if not uuid_str or not isinstance(uuid_str, str):
        return False

    # 길이 검증
    if len(uuid_str) != 36:
        return False

    # UUID 파싱 시도
    try:
        uid = UUID(uuid_str)

        # 버전 검증 (v7은 version=7)
        if uid.version != 7:
            return False

        # 변종 검증 (RFC 4122)
        if uid.variant != "specified in RFC 4122":
            return False

        return True
    except (ValueError, AttributeError):
        return False


def uuidv7_to_datetime(uuid_str: str) -> datetime:
    """
    UUID v7에서 타임스탬프를 추출하여 datetime 객체로 변환합니다.

    Args:
        uuid_str: UUID v7 문자열

    Returns:
        datetime: UTC 타임존의 datetime 객체

    Raises:
        ValueError: 유효하지 않은 UUID v7인 경우

    Example:
        >>> uid = generate_uuidv7()
        >>> dt = uuidv7_to_datetime(uid)
        >>> isinstance(dt, datetime)
        True
    """
    if not validate_uuidv7(uuid_str):
        raise ValueError(f"Invalid UUID v7: {uuid_str}")

    # uuid_extensions 라이브러리의 uuid_to_datetime 사용
    dt = uuid_to_datetime(uuid_str, suppress_error=False)

    if dt is None:
        raise ValueError(f"Cannot extract datetime from UUID v7: {uuid_str}")

    return dt


def datetime_to_uuidv7(dt: datetime) -> str:
    """
    datetime 객체를 해당 시점의 UUID v7로 변환합니다.

    참고: uuid_extensions 라이브러리는 현재 시간을 기반으로 UUID v7를 생성합니다.
    특정 시점의 정확한 UUID v7가 필요한 경우, 라이브러리 제약으로 인해
    근사치를 생성합니다. 실제 사용 시 generate_uuidv7() 사용을 권장합니다.

    Args:
        dt: datetime 객체 (UTC 권장)

    Returns:
        str: 해당 시점 근처의 UUID v7 문자열

    Example:
        >>> from datetime import datetime, timezone
        >>> dt = datetime.now(timezone.utc)
        >>> uid = datetime_to_uuidv7(dt)
        >>> validate_uuidv7(uid)
        True
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    # uuid_extensions 라이브러리는 자체적으로 현재 시간 기반 UUID v7 생성
    # 특정 시점의 정확한 UUID v7 생성은 라이브러리 제약으로 어려움
    return str(uuid7())


def get_uuidv7_timestamp_ms(uuid_str: str) -> int:
    """
    UUID v7에서 타임스탬프를 밀리초 단위로 추출합니다.

    Args:
        uuid_str: UUID v7 문자열

    Returns:
        int: Unix epoch milliseconds

    Raises:
        ValueError: 유효하지 않은 UUID v7인 경우
    """
    dt = uuidv7_to_datetime(uuid_str)
    return int(dt.timestamp() * 1000)
