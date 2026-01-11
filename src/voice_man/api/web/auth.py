"""
Authentication API Module

Provides login, logout, and token refresh endpoints for
SPEC-FORENSIC-WEB-001 web authentication system.

TAG: SPEC-FORENSIC-WEB-001/AUTH/API
"""

from datetime import timedelta
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, EmailStr

from voice_man.models.web.user import User, UserRole, PasswordHasher, RBACChecker
from voice_man.security.jwt import (
    create_access_token,
    create_refresh_token,
    refresh_access_token,
    validate_token,
    TokenExpiredError,
    InvalidTokenError,
    JWTConfig,
)


# ============================================================================
# Router Configuration
# ============================================================================


router = APIRouter(prefix="/api/v1/auth", tags=["Authentication"])
security = HTTPBearer(auto_error=False)


# ============================================================================
# Request/Response Models
# ============================================================================


class LoginRequest(BaseModel):
    """Login request payload"""

    email: EmailStr
    password: str


class UserResponse(BaseModel):
    """User information response"""

    id: str
    email: str
    username: str
    full_name: str | None = None
    role: UserRole
    is_active: bool

    @classmethod
    def model_validate(cls, obj):
        """Validate with UUID to string conversion"""
        from uuid import UUID

        if hasattr(obj, "id"):
            obj_dict = obj.__dict__.copy()
            if isinstance(obj_dict.get("id"), UUID):
                obj_dict["id"] = str(obj_dict["id"])
            return super().model_validate(obj_dict)
        return super().model_validate(obj)


class LoginResponse(BaseModel):
    """Login response with tokens"""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds
    user: UserResponse


class RefreshTokenRequest(BaseModel):
    """Refresh token request"""

    refresh_token: str


class RefreshTokenResponse(BaseModel):
    """Refresh token response"""

    access_token: str
    token_type: str = "bearer"
    expires_in: int


class LogoutResponse(BaseModel):
    """Logout response"""

    message: str


# ============================================================================
# Token Blacklist (In-Memory for Development)
# ============================================================================


class TokenBlacklist:
    """Simple in-memory token blacklist for development"""

    def __init__(self):
        self._blacklisted: set[str] = set()

    async def add(self, token: str, expires_in: timedelta) -> None:
        """Add token to blacklist"""
        self._blacklisted.add(token)

    async def exists(self, token: str) -> bool:
        """Check if token is blacklisted"""
        return token in self._blacklisted


# Global blacklist instance (replace with Redis in production)
_token_blacklist = TokenBlacklist()


# ============================================================================
# Mock User Service (Replace with actual service)
# ============================================================================


class MockUserService:
    """Mock user service for development"""

    def __init__(self):
        from uuid import uuid4

        self._users: dict[str, User] = {}
        self._password_hasher = PasswordHasher()
        # Create default test user
        default_password = "TestPassword123!"
        self._users["test@example.com"] = User(
            id=uuid4(),
            username="testuser",
            email="test@example.com",
            password_hash=self._password_hasher.hash_password(default_password),
            role=UserRole.LAWYER,
            full_name="Test User",
            is_active=True,
        )

    async def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        return self._users.get(email)

    async def get_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        for user in self._users.values():
            if user.id == user_id:
                return user
        return None

    async def update_last_login(self, user_id: str) -> None:
        """Update user's last login time"""
        pass  # Implement in production


_user_service = MockUserService()


def get_user_service():
    """Get user service instance"""
    return _user_service


def get_token_blacklist():
    """Get token blacklist instance"""
    return _token_blacklist


# ============================================================================
# Authentication Dependencies
# ============================================================================


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
    user_service=Depends(get_user_service),
    token_blacklist=Depends(get_token_blacklist),
) -> User:
    """
    Get current authenticated user from JWT token.

    Args:
        credentials: HTTP Bearer credentials
        user_service: User service instance
        token_blacklist: Token blacklist instance

    Returns:
        Authenticated user

    Raises:
        HTTPException: If token is invalid or user not found
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials

    # Check if token is blacklisted
    if await token_blacklist.exists(token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has been revoked",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Validate token
    try:
        payload = validate_token(token)
    except TokenExpiredError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        ) from e
    except InvalidTokenError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        ) from e

    # Get user
    user = await user_service.get_by_id(payload.user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user


async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """
    Get current active user.

    Args:
        current_user: Current authenticated user

    Returns:
        Active user

    Raises:
        HTTPException: If user is inactive
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive",
        )
    return current_user


# ============================================================================
# Endpoints
# ============================================================================


@router.post("/login", response_model=LoginResponse, status_code=status.HTTP_200_OK)
async def login(
    request: LoginRequest,
    user_service=Depends(get_user_service),
) -> LoginResponse:
    """
    Authenticate user and return JWT tokens.

    Args:
        request: Login request with email and password
        user_service: User service instance

    Returns:
        Login response with access token, refresh token, and user info

    Raises:
        HTTPException: If authentication fails
    """
    # Get user by email
    user = await user_service.get_by_email(request.email)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    # Verify password
    password_hasher = PasswordHasher()
    if not password_hasher.verify_password(request.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    # Check if user is active
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive",
        )

    # Create tokens
    access_token = create_access_token(
        data={
            "sub": user.id,
            "email": user.email,
            "role": user.role.value,
        }
    )

    refresh_token = create_refresh_token(
        data={
            "sub": user.id,
        }
    )

    # Update last login
    await user_service.update_last_login(user.id)

    return LoginResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=JWTConfig.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user=UserResponse.model_validate(user),
    )


@router.post("/logout", response_model=LogoutResponse)
async def logout(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
    token_blacklist=Depends(get_token_blacklist),
) -> LogoutResponse:
    """
    Logout user by blacklisting the access token.

    Args:
        credentials: HTTP Bearer credentials
        token_blacklist: Token blacklist instance

    Returns:
        Logout confirmation message
    """
    if credentials is None:
        return LogoutResponse(message="Successfully logged out")

    token = credentials.credentials

    # Add to blacklist (only if not already blacklisted)
    if not await token_blacklist.exists(token):
        await token_blacklist.add(token, timedelta(minutes=JWTConfig.ACCESS_TOKEN_EXPIRE_MINUTES))

    return LogoutResponse(message="Successfully logged out")


@router.post("/refresh", response_model=RefreshTokenResponse)
async def refresh_token(
    request: RefreshTokenRequest,
    token_blacklist=Depends(get_token_blacklist),
) -> RefreshTokenResponse:
    """
    Refresh access token using refresh token.

    Args:
        request: Refresh token request
        token_blacklist: Token blacklist instance

    Returns:
        New access token

    Raises:
        HTTPException: If refresh token is invalid or expired
    """
    refresh_token_str = request.refresh_token

    # Check if refresh token is blacklisted
    if await token_blacklist.exists(refresh_token_str):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has been revoked",
        )

    # Validate refresh token and create new access token
    try:
        new_access_token = refresh_access_token(refresh_token_str)
    except TokenExpiredError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
        ) from e
    except InvalidTokenError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
        ) from e

    return RefreshTokenResponse(
        access_token=new_access_token,
        token_type="bearer",
        expires_in=JWTConfig.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_active_user),
) -> UserResponse:
    """
    Get current authenticated user information.

    Args:
        current_user: Current authenticated user

    Returns:
        User information
    """
    return UserResponse.model_validate(current_user)


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    "router",
    "LoginRequest",
    "LoginResponse",
    "RefreshTokenRequest",
    "RefreshTokenResponse",
    "UserResponse",
    "get_current_user",
    "get_current_active_user",
    "get_user_service",
    "get_token_blacklist",
]
