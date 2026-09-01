"""Lógica de negocio de tickets, incluyendo la clasificación automática."""
from __future__ import annotations

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.ml_runtime import (churn_model, resolution_time_model, sentiment_model,
                            ticket_classifier)
from app.models.ticket import Ticket
from app.repositories.customer_repository import CustomerRepository
from app.repositories.ticket_repository import TicketRepository
from app.schemas.ticket import (TicketClassifyResponse, TicketCreate, TicketQueueItem,
                                TicketUpdate)


class TicketService:
    def __init__(self, db: Session):
        self.db = db
        self.repo = TicketRepository(db)
        self.customers = CustomerRepository(db)

    def list_tickets(self, skip: int, limit: int, customer_id: int | None) -> list[Ticket]:
        return self.repo.list(skip, limit, customer_id)

    def queue(self, limit: int = 20, only_open: bool = True) -> list[TicketQueueItem]:
        """Bandeja de trabajo del agente: tickets + señales de los tres modelos.

        Cada modelo se consulta de forma tolerante a fallos: si uno no está
        disponible, el ticket igual aparece en la bandeja, solo que sin ese dato.
        """
        tickets = self.repo.list(0, limit, None)
        if only_open:
            tickets = [t for t in tickets if t.status in ("open", "in_progress")]

        items: list[TicketQueueItem] = []
        for ticket in tickets:
            item = TicketQueueItem(
                ticket_id=ticket.ticket_id,
                customer_id=ticket.customer_id,
                customer_name=ticket.customer.name if ticket.customer else None,
                category=ticket.category,
                priority=ticket.priority,
                status=ticket.status,
                description=ticket.description,
                created_at=ticket.created_at,
            )
            try:
                sentiment, _, is_frustrated = sentiment_model.analyze_sentiment(ticket.description)
                item.sentiment, item.is_frustrated = sentiment, is_frustrated
            except Exception:  # noqa: BLE001
                pass
            try:
                item.estimated_hours = round(resolution_time_model.predict_resolution_time(
                    ticket.category, ticket.priority, ticket.description,
                    ticket.created_at.hour, ticket.created_at.weekday()), 1)
            except Exception:  # noqa: BLE001
                pass
            if ticket.customer is not None:
                try:
                    prob, risk = churn_model.predict_churn({
                        "tenure_months": ticket.customer.tenure_months,
                        "monthly_charge": ticket.customer.monthly_charge,
                        "total_charges": ticket.customer.total_charges,
                        "contract_type": ticket.customer.contract_type,
                        "payment_method": ticket.customer.payment_method,
                        "num_tickets": len(ticket.customer.tickets or []),
                        "avg_satisfaction": 3.5,
                    })
                    item.churn_probability, item.churn_risk = round(prob, 4), risk
                except Exception:  # noqa: BLE001
                    pass
            items.append(item)
        return items

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
