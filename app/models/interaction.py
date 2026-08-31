from __future__ import annotations

import datetime as dt

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class Interaction(Base):
    __tablename__ = "interaction"

    interaction_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticket_id: Mapped[int] = mapped_column(ForeignKey("ticket.ticket_id"), nullable=False)
    agent_response: Mapped[str] = mapped_column(Text, nullable=True)
    customer_msg: Mapped[str] = mapped_column(Text, nullable=False)
    sentiment: Mapped[str | None] = mapped_column(String(10), nullable=True)  # positive/neutral/negative
    resolution_time: Mapped[float | None] = mapped_column(Float, nullable=True)
    timestamp: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True),
                                                     server_default=func.now())

    ticket: Mapped["Ticket"] = relationship(back_populates="interactions")
