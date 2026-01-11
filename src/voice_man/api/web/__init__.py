"""
Web API Package for SPEC-FORENSIC-WEB-001

Contains authentication, complaint, evidence, and forensic APIs.
TAG: SPEC-FORENSIC-WEB-001/API
"""

from voice_man.api.web.auth import (
    router as auth_router,
    LoginRequest,
    LoginResponse,
    RefreshTokenRequest,
    RefreshTokenResponse,
    UserResponse,
)

__all__ = [
    "auth_router",
    "LoginRequest",
    "LoginResponse",
    "RefreshTokenRequest",
    "RefreshTokenResponse",
    "UserResponse",
]
