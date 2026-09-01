from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, EmailStr, Field


class LoginRequest(BaseModel):
    email: EmailStr = Field(..., examples=["admin@telecom.com"])
    password: str = Field(..., min_length=6, examples=["Admin123!"])


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in: int = Field(..., description="Segundos de validez del access token")
    role: str

    model_config = {"json_schema_extra": {"examples": [{
        "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
        "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
        "token_type": "bearer",
        "expires_in": 1800,
        "role": "admin",
    }]}}


class RefreshRequest(BaseModel):
    refresh_token: str


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=6)
    role: Literal["admin", "agent", "customer"] = "customer"
    customer_id: int | None = None


class MeResponse(BaseModel):
    """Identidad del usuario autenticado.

    El portal la necesita para saber a qué cliente pertenece la sesión: el JWT solo
    lleva el email y el rol, no el customer_id.
    """

    email: EmailStr
    role: Literal["admin", "agent", "customer"]
    customer_id: int | None = Field(
        None, description="Cliente asociado a la cuenta, si el usuario es un cliente"
    )
    customer_name: str | None = None
