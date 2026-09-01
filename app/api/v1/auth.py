from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import TokenPayload, get_current_user
from app.schemas.auth import LoginRequest, MeResponse, RefreshRequest, TokenResponse
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


@router.get("/me", response_model=MeResponse, summary="Usuario autenticado",
            description="Devuelve email, rol y cliente asociado a la sesión actual. "
                        "El portal la usa para saber de qué cliente son los tickets.")
def me(user: TokenPayload = Depends(get_current_user),
       db: Session = Depends(get_db)) -> MeResponse:
    return AuthService(db).me(user.sub)
