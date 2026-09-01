from __future__ import annotations

import datetime as dt

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class TicketCategory(Base):
    __tablename__ = "ticket_category"

    category_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    category_name: Mapped[str] = mapped_column(String(10), unique=True, nullable=False)
    description: Mapped[str] = mapped_column(String(255), nullable=True)
    avg_resolution: Mapped[float] = mapped_column(Integer, nullable=True)


class Ticket(Base):
    __tablename__ = "ticket"

    ticket_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    customer_id: Mapped[int] = mapped_column(ForeignKey("customer.customer_id"), nullable=False)
    category: Mapped[str] = mapped_column(String(10), nullable=False)  # TECH/BILL/PLAN/CNCL/OTHR
    description: Mapped[str] = mapped_column(Text, nullable=False)
    priority: Mapped[str] = mapped_column(String(10), nullable=False, default="medium")
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="open")
    satisfaction: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Conversación de la que nació el ticket. Permite que un asesor humano entre al
    # chat desde la bandeja y siga hablando con el cliente en el mismo hilo.
    agent_session_id: Mapped[str | None] = mapped_column(
        ForeignKey("agent_session.session_id"), nullable=True)

    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    deleted_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True),
                                                      server_default=func.now())
    resolved_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    customer: Mapped["Customer"] = relationship(back_populates="tickets")
    interactions: Mapped[list["Interaction"]] = relationship(back_populates="ticket")
