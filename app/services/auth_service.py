"""Lógica de negocio de autenticación (login, refresh, registro)."""
from __future__ import annotations

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.core.security import (create_access_token, create_refresh_token, decode_token,
                                hash_password, verify_password)
from app.repositories.user_repository import UserRepository
from app.schemas.auth import LoginRequest, RefreshRequest, RegisterRequest, TokenResponse
from app.core.config import get_settings

settings = get_settings()


class AuthService:
    def __init__(self, db: Session):
        self.db = db
        self.users = UserRepository(db)

    def _issue_tokens(self, email: str, role: str) -> TokenResponse:
        return TokenResponse(
            access_token=create_access_token(email, role),
            refresh_token=create_refresh_token(email, role),
            expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            role=role,
        )

    def login(self, data: LoginRequest) -> TokenResponse:
        user = self.users.get_by_email(data.email)
        if user is None or not verify_password(data.password, user.hashed_password):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                                 detail="Credenciales inválidas")
        return self._issue_tokens(user.email, user.role)

    def refresh(self, data: RefreshRequest) -> TokenResponse:
        payload = decode_token(data.refresh_token)
        if payload.get("type") != "refresh":
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                                 detail="Se requiere un refresh token válido")
        user = self.users.get_by_email(payload["sub"])
        if user is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                                 detail="Usuario no encontrado o inactivo")
        return self._issue_tokens(user.email, user.role)

    def register(self, data: RegisterRequest):
        if self.users.get_by_email(data.email) is not None:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                                 detail="Ya existe un usuario con ese email")
        return self.users.create(
            email=data.email,
            hashed_password=hash_password(data.password),
            role=data.role,
            customer_id=data.customer_id,
        )
