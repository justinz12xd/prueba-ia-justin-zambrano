"""Utilidades de seguridad: hashing de contraseñas, JWT (access + refresh) y roles."""
from __future__ import annotations

import datetime as dt
from typing import Literal

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext

from app.core.config import get_settings

settings = get_settings()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")

Role = Literal["admin", "agent", "customer"]


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def _create_token(subject: str, role: str, expires_delta: dt.timedelta,
                   token_type: str) -> str:
    now = dt.datetime.now(dt.timezone.utc)
    payload = {
        "sub": subject,
        "role": role,
        "type": token_type,
        "iat": now,
        "exp": now + expires_delta,
    }
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)


def create_access_token(subject: str, role: str,
                         expires_minutes: int | None = None) -> str:
    minutes = expires_minutes or settings.ACCESS_TOKEN_EXPIRE_MINUTES
    return _create_token(subject, role, dt.timedelta(minutes=minutes), "access")


def create_refresh_token(subject: str, role: str) -> str:
    return _create_token(subject, role, dt.timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS),
                          "refresh")


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM])
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token inválido o expirado",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


class TokenPayload:
    def __init__(self, sub: str, role: str, token_type: str):
        self.sub = sub
        self.role = role
        self.token_type = token_type


def get_current_user(token: str = Depends(oauth2_scheme)) -> TokenPayload:
    payload = decode_token(token)
    if payload.get("type") != "access":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                             detail="Se requiere un access token, no un refresh token")
    return TokenPayload(sub=payload["sub"], role=payload["role"], token_type=payload["type"])


def require_roles(*allowed_roles: Role):
    """Dependency factory: restringe un endpoint a ciertos roles."""

    def _checker(user: TokenPayload = Depends(get_current_user)) -> TokenPayload:
        if user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Rol '{user.role}' no autorizado. Roles permitidos: {allowed_roles}",
            )
        return user

    return _checker
