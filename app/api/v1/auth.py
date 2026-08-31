from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.schemas.auth import LoginRequest, RefreshRequest, TokenResponse
from app.services.auth_service import AuthService

router = APIRouter(prefix="/api/v1/auth", tags=["Autenticación"])


@router.post("/login", response_model=TokenResponse,
             summary="Iniciar sesión",
             description="Autentica con email/contraseña y retorna access + refresh token JWT.")
def login(data: LoginRequest, db: Session = Depends(get_db)) -> TokenResponse:
    return AuthService(db).login(data)


@router.post("/refresh", response_model=TokenResponse,
             summary="Refrescar access token",
             description="Genera un nuevo access token a partir de un refresh token válido.")
def refresh(data: RefreshRequest, db: Session = Depends(get_db)) -> TokenResponse:
    return AuthService(db).refresh(data)
