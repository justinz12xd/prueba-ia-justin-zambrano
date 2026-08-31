"""Capa de acceso a datos para Customer (patrón Repository)."""
from __future__ import annotations

import datetime as dt

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.customer import Customer
from app.schemas.customer import CustomerCreate, CustomerUpdate


class CustomerRepository:
    def __init__(self, db: Session):
        self.db = db

    def list(self, skip: int = 0, limit: int = 50, only_active: bool = True) -> list[Customer]:
        stmt = select(Customer)
        if only_active:
            stmt = stmt.where(Customer.is_active.is_(True))
        stmt = stmt.offset(skip).limit(limit).order_by(Customer.customer_id)
        return list(self.db.scalars(stmt))

    def count(self, only_active: bool = True) -> int:
        stmt = select(Customer)
        if only_active:
            stmt = stmt.where(Customer.is_active.is_(True))
        return len(list(self.db.scalars(stmt)))

    def get(self, customer_id: int, only_active: bool = True) -> Customer | None:
        customer = self.db.get(Customer, customer_id)
        if customer is None:
            return None
        if only_active and not customer.is_active:
            return None
        return customer

    def get_by_email(self, email: str) -> Customer | None:
        stmt = select(Customer).where(Customer.email == email)
        return self.db.scalars(stmt).first()

    def create(self, data: CustomerCreate) -> Customer:
        customer = Customer(**data.model_dump())
        self.db.add(customer)
        self.db.commit()
        self.db.refresh(customer)
        return customer

    def update(self, customer: Customer, data: CustomerUpdate) -> Customer:
        for field, value in data.model_dump(exclude_unset=True).items():
            setattr(customer, field, value)
        self.db.commit()
        self.db.refresh(customer)
        return customer

    def soft_delete(self, customer: Customer) -> Customer:
        customer.is_active = False
        customer.deleted_at = dt.datetime.now(dt.timezone.utc)
        self.db.commit()
        self.db.refresh(customer)
        return customer
