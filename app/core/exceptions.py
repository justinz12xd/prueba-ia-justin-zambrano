"""Manejadores de excepciones globales -> respuestas de error estandarizadas."""
from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.schemas.common import ErrorDetail, ErrorResponse

CODE_BY_STATUS = {
    400: "BAD_REQUEST",
    401: "UNAUTHORIZED",
    403: "FORBIDDEN",
    404: "NOT_FOUND",
    409: "CONFLICT",
    422: "VALIDATION_ERROR",
    503: "SERVICE_UNAVAILABLE",
}


def _error_body(code: str, message: str, details=None) -> dict:
    return ErrorResponse(error=ErrorDetail(code=code, message=message, details=details)).model_dump()


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        code = CODE_BY_STATUS.get(exc.status_code, "ERROR")
        return JSONResponse(status_code=exc.status_code,
                             content=_error_body(code, str(exc.detail)))

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        # pydantic v2 puede incluir excepciones "crudas" (no serializables) en
        # error["ctx"] cuando el error viene de un @field_validator (p.ej. ValueError
        # de validación de teléfono). Se normalizan a texto antes de responder.
        safe_errors = []
        for error in exc.errors():
            error = dict(error)
            if "ctx" in error and isinstance(error["ctx"], dict):
                error["ctx"] = {k: str(v) for k, v in error["ctx"].items()}
            safe_errors.append(error)
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=_error_body("VALIDATION_ERROR", "Error de validación en la solicitud",
                                 details=safe_errors),
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                             content=_error_body("VALIDATION_ERROR", str(exc)))

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_error_body("INTERNAL_ERROR", "Ocurrió un error interno inesperado"),
        )
