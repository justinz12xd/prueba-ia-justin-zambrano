from __future__ import annotations

import datetime as dt

from sqlalchemy import Boolean, DateTime, Float, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class Customer(Base):
    __tablename__ = "customer"

    customer_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    email: Mapped[str] = mapped_column(String(150), unique=True, nullable=False, index=True)
    phone: Mapped[str] = mapped_column(String(15), nullable=False)
    plan_type: Mapped[str] = mapped_column(String(60), nullable=False)
    monthly_charge: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    tenure_months: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_charges: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    contract_type: Mapped[str] = mapped_column(String(30), nullable=False,
                                                default="month-to-month")
    payment_method: Mapped[str] = mapped_column(String(30), nullable=False,
                                                 default="credit_card")
    churn_status: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Eliminación lógica
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    deleted_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True),
                                                             nullable=True)

    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True),
                                                      server_default=func.now())
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True),
                                                      server_default=func.now(),
                                                      onupdate=func.now())

    tickets: Mapped[list["Ticket"]] = relationship(back_populates="customer")
    predictions: Mapped[list["Prediction"]] = relationship(back_populates="customer")
    agent_sessions: Mapped[list["AgentSession"]] = relationship(back_populates="customer")
    user_account: Mapped["UserAccount | None"] = relationship(back_populates="customer")
