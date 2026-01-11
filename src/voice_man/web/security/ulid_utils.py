"""
ULID 유틸리티 모듈

TAG: TAG-001-ULID
SPEC: SPEC-FORENSIC-WEB-001

Universally Unique Lexicographically Sortable Identifier (ULID) 생성 및 관리.
Crockford Base32 인코딩을 사용하며, 128-bit 식별자를 생성합니다.
- 26字符 길이
- 시간순 정렬 가능
- UUID보다 짧고 URL-safe
"""

import time
from datetime import datetime, timezone, timedelta
from typing import Optional
from ulid import ULID as PyUlid
from ulid import api as ulid_api


# Crockford Base32 문자셋 (ULID 표준)
CROCKFORD_BASE32 = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def generate_ulid() -> str:
    """
    새로운 ULID를 생성합니다.

    Returns:
        str: 26자리 ULID 문자열 (Crockford Base32 인코딩)

    Example:
        >>> ulid = generate_ulid()
        >>> len(ulid)
        26
        >>> validate_ulid(ulid)
        True
    """
    # ulid-py 라이브러리 사용
    ulid_obj = ulid_api.new()
    return str(ulid_obj)


def validate_ulid(ulid_str: Optional[str]) -> bool:
    """
    ULID 문자열의 유효성을 검증합니다.

    Args:
        ulid_str: 검증할 ULID 문자열

    Returns:
        bool: 유효한 ULID이면 True, 아니면 False

    Example:
        >>> ulid = generate_ulid()
        >>> validate_ulid(ulid)
        True
        >>> validate_ulid("invalid")
        False
    """
    if not ulid_str or not isinstance(ulid_str, str):
        return False

    # 길이 검증
    if len(ulid_str) != 26:
        return False

    # 문자셋 검증 (Crockford Base32)
    valid_chars = set(CROCKFORD_BASE32)
    if not all(c in valid_chars for c in ulid_str):
        return False

    # ulid-py 라이브러리로 파싱 시도
    try:
        PyUlid.from_str(ulid_str)
        return True
    except (ValueError, AttributeError):
        return False


def ulid_to_datetime(ulid_str: str) -> datetime:
    """
    ULID에서 타임스탬프를 추출하여 datetime 객체로 변환합니다.

    Args:
        ulid_str: ULID 문자열

    Returns:
        datetime: UTC 타임존의 datetime 객체

    Raises:
        ValueError: 유효하지 않은 ULID인 경우

    Example:
        >>> ulid = generate_ulid()
        >>> dt = ulid_to_datetime(ulid)
        >>> isinstance(dt, datetime)
        True
    """
    if not validate_ulid(ulid_str):
        raise ValueError(f"Invalid ULID: {ulid_str}")

    ulid_obj = PyUlid.from_str(ulid_str)

    # ULID timestamp는 Unix epoch milliseconds (microseconds in ulid-py)
    # ulid-py는 microseconds를 사용하므로 milliseconds로 변환
    timestamp_ms = ulid_obj.timestamp.microseconds // 1000

    # Unix epoch milliseconds로부터 datetime 생성
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)


def datetime_to_ulid(dt: datetime) -> str:
    """
    datetime 객체를 해당 시점의 ULID로 변환합니다.

    참고: 동일한 시간에 생성된 ULID는 랜덤 부분이 다르므로,
    이 함수는 타임스탬프만 동일한 ULID를 생성합니다.
    실제 사용 시 랜덤성을 위해 generate_ulid() 사용을 권장합니다.

    Args:
        dt: datetime 객체 (UTC 권장)

    Returns:
        str: 해당 시점의 ULID 문자열

    Example:
        >>> from datetime import datetime, timezone
        >>> dt = datetime.now(timezone.utc)
        >>> ulid = datetime_to_ulid(dt)
        >>> validate_ulid(ulid)
        True
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    # milliseconds로 변환
    timestamp_ms = int(dt.timestamp() * 1000)

    # ulid-py는 microseconds를 사용하므로 변환
    timestamp_micros = timestamp_ms * 1000

    # 타임스탬프와 랜덤 부분으로 ULID 생성
    ulid_obj = ulid_api.new(timestamp_micros)
    return str(ulid_obj)


def get_ulid_timestamp_ms(ulid_str: str) -> int:
    """
    ULID에서 타임스탬프를 밀리초 단위로 추출합니다.

    Args:
        ulid_str: ULID 문자열

    Returns:
        int: Unix epoch milliseconds

    Raises:
        ValueError: 유효하지 않은 ULID인 경우
    """
    if not validate_ulid(ulid_str):
        raise ValueError(f"Invalid ULID: {ulid_str}")

    ulid_obj = PyUlid.from_str(ulid_str)
    return ulid_obj.timestamp.microseconds // 1000
