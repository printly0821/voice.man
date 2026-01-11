"""
JWT Authentication Module

Provides JWT token creation, validation, and refresh functionality for
SPEC-FORENSIC-WEB-001 web authentication system.

TAG: SPEC-FORENSIC-WEB-001/AUTH/JWT
"""

import os
from datetime import datetime, timedelta, UTC
from typing import Optional, Dict, Any
from dataclasses import dataclass

import jwt
from jwt import PyJWTError


# ============================================================================
# Configuration
# ============================================================================


class JWTConfig:
    """JWT configuration settings"""

    SECRET_KEY: str = os.getenv(
        "JWT_SECRET_KEY", "your-secret-key-change-in-production-min-32-chars"
    )
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 240  # 4 hours
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7


# ============================================================================
# Exceptions
# ============================================================================


class AuthenticationError(Exception):
    """Base exception for authentication errors"""

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class TokenExpiredError(AuthenticationError):
    """Raised when token has expired"""

    pass


class InvalidTokenError(AuthenticationError):
    """Raised when token is invalid or malformed"""

    pass


# ============================================================================
# Data Models
# ============================================================================


@dataclass
class TokenPayload:
    """JWT token payload structure"""

    user_id: str
    role: str
    exp: datetime
    iat: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JWT encoding"""
        return {
            "sub": self.user_id,
            "role": self.role,
            "exp": int(self.exp.timestamp()),
            "iat": int(self.iat.timestamp()),
        }


# ============================================================================
# JWT Functions
# ============================================================================


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """
    Create JWT access token.

    Args:
        data: Payload data to include in token (sub, email, role, etc.)
        expires_delta: Custom expiration time (default: 4 hours)

    Returns:
        Encoded JWT string

    Example:
        >>> token = create_access_token(
        ...     data={"sub": "user-123", "email": "user@example.com", "role": "lawyer"}
        ... )
    """
    # Copy data to avoid mutating original
    to_encode = data.copy()

    # Set expiration time
    if expires_delta:
        expire = datetime.now(UTC) + expires_delta
    else:
        expire = datetime.now(UTC) + timedelta(minutes=JWTConfig.ACCESS_TOKEN_EXPIRE_MINUTES)

    # Add standard claims
    to_encode.update(
        {
            "exp": expire,
            "iat": datetime.now(UTC),
        }
    )

    # Encode JWT
    encoded_jwt = jwt.encode(to_encode, JWTConfig.SECRET_KEY, algorithm=JWTConfig.ALGORITHM)

    return encoded_jwt


def create_refresh_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """
    Create JWT refresh token.

    Args:
        data: Payload data (typically just user_id/sub)
        expires_delta: Custom expiration time (default: 7 days)

    Returns:
        Encoded JWT string

    Example:
        >>> token = create_refresh_token(data={"sub": "user-123"})
    """
    to_encode = data.copy()

    # Set expiration time (default 7 days)
    if expires_delta:
        expire = datetime.now(UTC) + expires_delta
    else:
        expire = datetime.now(UTC) + timedelta(days=JWTConfig.REFRESH_TOKEN_EXPIRE_DAYS)

    to_encode.update(
        {
            "exp": expire,
            "iat": datetime.now(UTC),
            "type": "refresh",  # Mark as refresh token
        }
    )

    encoded_jwt = jwt.encode(to_encode, JWTConfig.SECRET_KEY, algorithm=JWTConfig.ALGORITHM)

    return encoded_jwt


def decode_token(token: str, validate_exp: bool = True) -> Dict[str, Any]:
    """
    Decode JWT token without full validation.

    Args:
        token: JWT string to decode
        validate_exp: Whether to validate expiration (default: True)

    Returns:
        Decoded token payload as dictionary

    Raises:
        InvalidTokenError: If token is malformed or invalid
        TokenExpiredError: If token is expired and validate_exp=True
    """
    try:
        # Decode with optional expiration validation
        options = {"verify_exp": validate_exp}
        payload = jwt.decode(
            token, JWTConfig.SECRET_KEY, algorithms=[JWTConfig.ALGORITHM], options=options
        )
        return payload

    except jwt.ExpiredSignatureError as e:
        raise TokenExpiredError("Token has expired") from e
    except jwt.InvalidSignatureError as e:
        raise InvalidTokenError("Invalid token signature") from e
    except jwt.DecodeError as e:
        raise InvalidTokenError("Malformed token") from e
    except PyJWTError as e:
        raise InvalidTokenError(f"Invalid token: {str(e)}") from e


def validate_token(token: str) -> TokenPayload:
    """
    Validate JWT token and return payload.

    Args:
        token: JWT string to validate

    Returns:
        TokenPayload with user data

    Raises:
        TokenExpiredError: If token has expired
        InvalidTokenError: If token is invalid or malformed
    """
    try:
        payload = decode_token(token, validate_exp=True)

        # Extract required fields
        user_id = payload.get("sub")
        role = payload.get("role")
        exp = payload.get("exp")
        iat = payload.get("iat")

        # Validate required fields
        if not user_id:
            raise InvalidTokenError("Missing 'sub' claim in token")
        if not role:
            raise InvalidTokenError("Missing 'role' claim in token")
        if not exp:
            raise InvalidTokenError("Missing 'exp' claim in token")
        if not iat:
            raise InvalidTokenError("Missing 'iat' claim in token")

        # Convert timestamps to datetime
        exp_datetime = datetime.fromtimestamp(exp, tz=UTC)
        iat_datetime = datetime.fromtimestamp(iat, tz=UTC)

        return TokenPayload(
            user_id=user_id,
            role=role,
            exp=exp_datetime,
            iat=iat_datetime,
        )

    except TokenExpiredError:
        raise
    except InvalidTokenError:
        raise
    except Exception as e:
        raise InvalidTokenError(f"Token validation failed: {str(e)}") from e


def refresh_access_token(refresh_token: str) -> str:
    """
    Create new access token from refresh token.

    Args:
        refresh_token: Valid refresh token

    Returns:
        New access token

    Raises:
        TokenExpiredError: If refresh token has expired
        InvalidTokenError: If refresh token is invalid
    """
    # Validate refresh token
    payload = validate_token(refresh_token)

    # Create new access token with user data from refresh token
    new_access_token = create_access_token(
        data={
            "sub": payload.user_id,
            "role": payload.role,
        }
    )

    return new_access_token


# ============================================================================
# Utility Functions
# ============================================================================


def get_token_expiry(token: str) -> Optional[datetime]:
    """
    Get expiration time from token without validation.

    Args:
        token: JWT string

    Returns:
        Expiration datetime or None if not found
    """
    try:
        payload = decode_token(token, validate_exp=False)
        exp_timestamp = payload.get("exp")
        if exp_timestamp:
            return datetime.fromtimestamp(exp_timestamp, tz=UTC)
    except Exception:
        pass
    return None


def is_token_expired(token: str) -> bool:
    """
    Check if token is expired without raising exception.

    Args:
        token: JWT string

    Returns:
        True if token is expired, False otherwise
    """
    try:
        validate_token(token)
        return False
    except TokenExpiredError:
        return True
    except Exception:
        return True


def extract_user_id(token: str) -> Optional[str]:
    """
    Extract user_id from token without full validation.

    Args:
        token: JWT string

    Returns:
        User ID (sub claim) or None if not found
    """
    try:
        payload = decode_token(token, validate_exp=False)
        return payload.get("sub")
    except Exception:
        return None


def extract_role(token: str) -> Optional[str]:
    """
    Extract role from token without full validation.

    Args:
        token: JWT string

    Returns:
        Role claim or None if not found
    """
    try:
        payload = decode_token(token, validate_exp=False)
        return payload.get("role")
    except Exception:
        return None


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    # Configuration
    "JWTConfig",
    # Exceptions
    "AuthenticationError",
    "TokenExpiredError",
    "InvalidTokenError",
    # Data Models
    "TokenPayload",
    # Functions
    "create_access_token",
    "create_refresh_token",
    "decode_token",
    "validate_token",
    "refresh_access_token",
    "get_token_expiry",
    "is_token_expired",
    "extract_user_id",
    "extract_role",
]
