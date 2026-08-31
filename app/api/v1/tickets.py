from __future__ import annotations

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import require_roles
from app.schemas.ticket import (TicketClassifyRequest, TicketClassifyResponse, TicketCreate,
                                 TicketResponse, TicketUpdate)
from app.services.ticket_service import TicketService

router = APIRouter(prefix="/api/v1/tickets", tags=["Tickets"])


@router.get("", response_model=list[TicketResponse], summary="Listar tickets",
            dependencies=[Depends(require_roles("admin", "agent", "customer"))])
def list_tickets(skip: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=200),
                  customer_id: int | None = None,
                  db: Session = Depends(get_db)) -> list[TicketResponse]:
    return TicketService(db).list_tickets(skip, limit, customer_id)


@router.get("/{ticket_id}", response_model=TicketResponse, summary="Obtener un ticket",
            dependencies=[Depends(require_roles("admin", "agent", "customer"))])
def get_ticket(ticket_id: int, db: Session = Depends(get_db)) -> TicketResponse:
    return TicketService(db).get_ticket_or_404(ticket_id)


@router.post("", response_model=TicketResponse, status_code=status.HTTP_201_CREATED,
             summary="Crear ticket",
             description="Si no se envía 'category', se infiere automáticamente con el "
                         "clasificador ML a partir de la descripción.",
             dependencies=[Depends(require_roles("admin", "agent", "customer"))])
def create_ticket(data: TicketCreate, db: Session = Depends(get_db)) -> TicketResponse:
    return TicketService(db).create(data)


@router.put("/{ticket_id}", response_model=TicketResponse, summary="Actualizar ticket",
            dependencies=[Depends(require_roles("admin", "agent"))])
def update_ticket(ticket_id: int, data: TicketUpdate,
                   db: Session = Depends(get_db)) -> TicketResponse:
    return TicketService(db).update(ticket_id, data)


@router.post("/classify", response_model=TicketClassifyResponse,
             summary="Clasificar descripción de ticket",
             description="Clasifica un texto libre en una de las categorías "
                         "TECH/BILL/PLAN/CNCL/OTHR sin crear un ticket.",
             dependencies=[Depends(require_roles("admin", "agent", "customer"))])
def classify_ticket(data: TicketClassifyRequest, db: Session = Depends(get_db)) -> TicketClassifyResponse:
    return TicketService(db).classify(data.description)
