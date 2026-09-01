"""Capa de acceso a datos para Ticket."""
from __future__ import annotations

import datetime as dt

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.ticket import Ticket
from app.schemas.ticket import TicketUpdate


class TicketRepository:
    def __init__(self, db: Session):
        self.db = db

    def list(self, skip: int = 0, limit: int = 50, customer_id: int | None = None,
              only_active: bool = True) -> list[Ticket]:
        stmt = select(Ticket)
        if only_active:
            stmt = stmt.where(Ticket.is_active.is_(True))
        if customer_id is not None:
            stmt = stmt.where(Ticket.customer_id == customer_id)
        stmt = stmt.offset(skip).limit(limit).order_by(Ticket.ticket_id.desc())
        return list(self.db.scalars(stmt))

    def get(self, ticket_id: int, only_active: bool = True) -> Ticket | None:
        ticket = self.db.get(Ticket, ticket_id)
        if ticket is None:
            return None
        if only_active and not ticket.is_active:
            return None
        return ticket

    def find_open_by_category(self, customer_id: int, category: str) -> Ticket | None:
        """Ticket activo y sin resolver del mismo tipo para ese cliente.

        Lo usa el agente para no abrir un ticket nuevo cada vez que el cliente
        insiste sobre el mismo problema en la conversación.
        """
        stmt = (
            select(Ticket)
            .where(Ticket.customer_id == customer_id,
                   Ticket.category == category,
                   Ticket.is_active.is_(True),
                   Ticket.status.in_(("open", "in_progress")))
            .order_by(Ticket.ticket_id.desc())
        )
        return self.db.scalars(stmt).first()

    def create(self, customer_id: int, category: str, description: str, priority: str) -> Ticket:
        ticket = Ticket(customer_id=customer_id, category=category, description=description,
                         priority=priority, status="open")
        self.db.add(ticket)
        self.db.commit()
        self.db.refresh(ticket)
        return ticket

    def update(self, ticket: Ticket, data: TicketUpdate) -> Ticket:
        payload = data.model_dump(exclude_unset=True)
        if payload.get("status") in {"resolved", "closed"} and ticket.resolved_at is None:
            ticket.resolved_at = dt.datetime.now(dt.timezone.utc)
        for field, value in payload.items():
            setattr(ticket, field, value)
        self.db.commit()
        self.db.refresh(ticket)
        return ticket
