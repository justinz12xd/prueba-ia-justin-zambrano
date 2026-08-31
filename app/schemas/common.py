"""Schemas comunes: respuesta de error estandarizada y paginación."""
from __future__ import annotations

from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class ErrorDetail(BaseModel):
    code: str = Field(..., description="Código interno del error, p.ej. 'VALIDATION_ERROR'")
    message: str = Field(..., description="Mensaje descriptivo y accionable")
    details: Any | None = Field(default=None, description="Información adicional del error")


class ErrorResponse(BaseModel):
    error: ErrorDetail


class PaginatedResponse(BaseModel, Generic[T]):
    items: list[T]
    total: int
    page: int
    page_size: int
