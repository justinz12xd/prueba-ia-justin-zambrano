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

    model_config = {
        "from_attributes": True,
        "json_schema_extra": {"examples": [{
        "ticket_id": 12,
        "customer_id": 1,
        "description": "El internet se corta cada media hora desde el lunes",
        "category": "TECH",
        "priority": "high",
        "status": "open",
        "satisfaction": None,
        "is_active": True,
        "created_at": "2026-08-31T18:35:00Z",
        "resolved_at": None,
    }]},
    }


class TicketClassifyRequest(BaseModel):
    description: str = Field(..., min_length=10, examples=["No tengo señal de internet"])


class TicketClassifyResponse(BaseModel):
    predicted_category: TicketCategoryLiteral
    probabilities: dict[str, float]

    model_config = {"json_schema_extra": {"examples": [{
        "predicted_category": "TECH",
        "probabilities": {
            "TECH": 0.8087, "PLAN": 0.0593, "BILL": 0.0448,
            "CNCL": 0.0444, "OTHR": 0.0427,
        },
    }]}}


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

    sentiment: str | None = Field(
        None, description="Sentimiento del último mensaje del cliente en la conversación "
                          "(o de la descripción, si el ticket no tiene interacciones)")
    sentiment_source: Literal["last_message", "description"] = Field(
        "description", description="Sobre qué texto se evaluó el sentimiento")
    is_frustrated: bool = False
    churn_probability: float | None = None
    churn_risk: str | None = None
    estimated_hours: float | None = None
