from __future__ import annotations

import datetime as dt

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, examples=["Hola, mi internet no funciona"])
    session_id: str | None = Field(None, description="Si se omite, se crea una nueva sesión")
    customer_id: int | None = Field(None, description="Si se conoce, permite personalizar la atención")


class ChatResponse(BaseModel):
    session_id: str
    response: str
    intent: str | None = None
    escalate: bool = False
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
