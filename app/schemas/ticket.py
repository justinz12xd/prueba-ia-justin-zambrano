from __future__ import annotations

import datetime as dt
from typing import Literal

from pydantic import BaseModel, Field

TicketCategoryLiteral = Literal["TECH", "BILL", "PLAN", "CNCL", "OTHR"]
Priority = Literal["low", "medium", "high"]
TicketStatus = Literal["open", "in_progress", "resolved", "closed"]


class TicketBase(BaseModel):
    description: str = Field(
        ..., min_length=20, max_length=500,
        examples=["El internet está muy lento desde hace dos días, ya reinicié el router"],
    )
    priority: Priority = "medium"


class TicketCreate(TicketBase):
    customer_id: int
    category: TicketCategoryLiteral | None = Field(
        None, description="Si no se envía, se infiere automáticamente con el clasificador ML"
    )


class TicketUpdate(BaseModel):
    description: str | None = Field(None, min_length=20, max_length=500)
    priority: Priority | None = None
    status: TicketStatus | None = None
    category: TicketCategoryLiteral | None = None
    satisfaction: int | None = Field(None, ge=1, le=5)


class TicketResponse(TicketBase):
    ticket_id: int
    customer_id: int
    category: str
    status: TicketStatus
    satisfaction: int | None = None
    is_active: bool
    created_at: dt.datetime
    resolved_at: dt.datetime | None = None

    model_config = {"from_attributes": True}


class TicketClassifyRequest(BaseModel):
    description: str = Field(..., min_length=10, examples=["No tengo señal de internet"])


class TicketClassifyResponse(BaseModel):
    predicted_category: TicketCategoryLiteral
    probabilities: dict[str, float]


class TicketQueueItem(BaseModel):
    """Ticket enriquecido con las señales de los modelos, para la consola de soporte.

    Se arma en el backend (y no con N llamadas desde el navegador) para que la
    consola se dibuje con una sola petición.
    """

    ticket_id: int
    customer_id: int
    customer_name: str | None = None
    category: TicketCategoryLiteral
    priority: Priority
    status: TicketStatus
    description: str
    created_at: dt.datetime

    sentiment: str | None = Field(None, description="Sentimiento detectado en la descripción")
    is_frustrated: bool = False
    churn_probability: float | None = None
    churn_risk: str | None = None
    estimated_hours: float | None = None
