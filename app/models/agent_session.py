from __future__ import annotations

import datetime as dt

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class AgentSession(Base):
    __tablename__ = "agent_session"

    session_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    customer_id: Mapped[int | None] = mapped_column(ForeignKey("customer.customer_id"),
                                                      nullable=True)
    conversation: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    tokens_used: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    started_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True),
                                                      server_default=func.now())
    ended_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    customer: Mapped["Customer | None"] = relationship(back_populates="agent_sessions")
