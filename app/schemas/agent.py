from __future__ import annotations

import datetime as dt

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, examples=["Hola, mi internet no funciona"])
    session_id: str | None = Field(None, description="Si se omite, se crea una nueva sesión")
    customer_id: int | None = Field(None, description="Si se conoce, permite personalizar la atención")


class ChatTicket(BaseModel):
    """Ticket abierto (o reutilizado) por el agente durante la conversación."""

    ticket_id: int
    category: str
    priority: str
    status: str
    is_new: bool = Field(..., description="False si se reutilizó un ticket abierto del mismo tipo")
    estimated_hours: float | None = Field(
        None, description="Estimación de la red de tiempo de resolución, si está disponible"
    )


class ChatResponse(BaseModel):
    session_id: str
    response: str
    intent: str | None = None
    escalate: bool = False
    ticket: ChatTicket | None = Field(
        None, description="Presente si la consulta ameritó registrar una solicitud de soporte"
    )
    customer_context: dict | None = None


class SessionMessage(BaseModel):
    role: str
    content: str


class SessionResponse(BaseModel):
    session_id: str
    customer_id: int | None
    conversation: list[SessionMessage]
    tokens_used: int
    started_at: dt.datetime
    ended_at: dt.datetime | None
