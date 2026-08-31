"""Cuenta de usuario para autenticación (no forma parte del esquema original del
enunciado, pero es necesaria para JWT + roles admin/agent/customer)."""
from __future__ import annotations

import datetime as dt

from sqlalchemy import Boolean, DateTime, ForeignKey, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class UserAccount(Base):
    __tablename__ = "user_account"

    user_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(150), unique=True, nullable=False, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(20), nullable=False, default="customer")
    customer_id: Mapped[int | None] = mapped_column(ForeignKey("customer.customer_id"),
                                                      nullable=True)

    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True),
                                                      server_default=func.now())

    customer: Mapped["Customer | None"] = relationship(back_populates="user_account")
