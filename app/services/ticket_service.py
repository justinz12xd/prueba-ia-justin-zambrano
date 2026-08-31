"""Lógica de negocio de tickets, incluyendo la clasificación automática."""
from __future__ import annotations

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.ml_runtime import ticket_classifier
from app.models.ticket import Ticket
from app.repositories.customer_repository import CustomerRepository
from app.repositories.ticket_repository import TicketRepository
from app.schemas.ticket import TicketClassifyResponse, TicketCreate, TicketUpdate


class TicketService:
    def __init__(self, db: Session):
        self.db = db
        self.repo = TicketRepository(db)
        self.customers = CustomerRepository(db)

    def list_tickets(self, skip: int, limit: int, customer_id: int | None) -> list[Ticket]:
        return self.repo.list(skip, limit, customer_id)

    def get_ticket_or_404(self, ticket_id: int) -> Ticket:
        ticket = self.repo.get(ticket_id)
        if ticket is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                 detail=f"Ticket {ticket_id} no encontrado")
        return ticket

    def classify(self, description: str) -> TicketClassifyResponse:
        try:
            category, probabilities = ticket_classifier.classify_ticket(description)
        except ticket_classifier.TicketClassifierUnavailable as exc:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                                 detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                                 detail=str(exc)) from exc
        return TicketClassifyResponse(predicted_category=category, probabilities=probabilities)

    def create(self, data: TicketCreate) -> Ticket:
        if self.customers.get(data.customer_id) is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                 detail=f"Cliente {data.customer_id} no encontrado")
        category = data.category
        if category is None:
            category, _ = ticket_classifier.classify_ticket(data.description)
        return self.repo.create(data.customer_id, category, data.description, data.priority)

    def update(self, ticket_id: int, data: TicketUpdate) -> Ticket:
        ticket = self.get_ticket_or_404(ticket_id)
        return self.repo.update(ticket, data)
