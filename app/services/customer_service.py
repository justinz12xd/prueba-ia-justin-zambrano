"""Lógica de negocio de clientes, incluyendo la predicción de churn persistida."""
from __future__ import annotations

import datetime as dt
import logging

from fastapi import HTTPException, status
from sqlalchemy.exc import DatabaseError
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from app.ml_runtime import churn_model
from app.models.customer import Customer
from app.models.prediction import Prediction
from app.models.ticket import Ticket
from app.repositories.customer_repository import CustomerRepository
from app.schemas.customer import ChurnPredictionResponse, CustomerCreate, CustomerUpdate

logger = logging.getLogger("customers")


class CustomerService:
    def __init__(self, db: Session):
        self.db = db
        self.repo = CustomerRepository(db)

    def list_customers(self, skip: int, limit: int) -> tuple[list[Customer], int]:
        return self.repo.list(skip, limit), self.repo.count()

    def get_customer_or_404(self, customer_id: int) -> Customer:
        customer = self.repo.get(customer_id)
        if customer is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                 detail=f"Cliente {customer_id} no encontrado")
        return customer

    def create(self, data: CustomerCreate) -> Customer:
        if self.repo.get_by_email(data.email):
            raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                                 detail="Ya existe un cliente con ese email")
        return self.repo.create(data)

    def update(self, customer_id: int, data: CustomerUpdate) -> Customer:
        customer = self.get_customer_or_404(customer_id)
        return self.repo.update(customer, data)

    def soft_delete(self, customer_id: int) -> Customer:
        customer = self.get_customer_or_404(customer_id)
        return self.repo.soft_delete(customer)

    def ticket_stats(self, customer_id: int) -> tuple[int, float]:
        """Nº de tickets sin resolver y satisfacción promedio real del cliente.

        En PostgreSQL se resuelve con la función `fn_customer_churn_summary` de
        `sql/init.sql` (una sola llamada). En SQLite —que es lo que usan los tests—
        esa función PL/pgSQL no existe, así que se calcula el equivalente con el ORM.
        """
        if self.db.bind is not None and self.db.bind.dialect.name == "postgresql":
            try:
                row = self.db.execute(
                    text("SELECT open_tickets, avg_satisfaction "
                         "FROM fn_customer_churn_summary(:customer_id)"),
                    {"customer_id": customer_id},
                ).mappings().first()
                if row is not None:
                    avg = row["avg_satisfaction"]
                    return int(row["open_tickets"] or 0), float(avg) if avg is not None else 3.5
            except DatabaseError as exc:
                # La función vive en sql/init.sql, que docker-compose carga al crear el
                # volumen. En un Postgres gestionado (Railway, Render, Supabase) ese
                # script puede no haberse ejecutado nunca: en vez de romper el endpoint,
                # se calcula lo mismo con el ORM.
                logger.warning("fn_customer_churn_summary no disponible, uso el ORM: %s", exc)
                # En PostgreSQL una sentencia fallida invalida la transacción entera:
                # sin este rollback, las consultas siguientes también fallarían.
                self.db.rollback()

        # El fallback debe devolver exactamente lo mismo que la función SQL: tickets
        # SIN RESOLVER (el enunciado define num_tickets como "cantidad de tickets
        # abiertos") y la satisfacción promedio sobre todos los tickets del cliente.
        num_tickets = self.db.execute(
            select(func.count(Ticket.ticket_id)).where(
                Ticket.customer_id == customer_id,
                Ticket.is_active.is_(True),
                Ticket.status.notin_(("resolved", "closed")),
            )
        ).scalar_one()
        avg_satisfaction = self.db.execute(
            select(func.avg(Ticket.satisfaction)).where(
                Ticket.customer_id == customer_id, Ticket.is_active.is_(True))
        ).scalar_one()
        return int(num_tickets or 0), float(avg_satisfaction) if avg_satisfaction is not None else 3.5

    def predict_churn_for_customer(self, customer_id: int) -> ChurnPredictionResponse:
        customer = self.get_customer_or_404(customer_id)
        num_tickets, avg_satisfaction = self.ticket_stats(customer_id)
        features = {
            "tenure_months": customer.tenure_months,
            "monthly_charge": customer.monthly_charge,
            "total_charges": customer.total_charges,
            "contract_type": customer.contract_type,
            "payment_method": customer.payment_method,
            "num_tickets": num_tickets,
            "avg_satisfaction": avg_satisfaction,
        }
        prob, risk = churn_model.predict_churn(features)

        record = Prediction(customer_id=customer_id, churn_prob=prob, risk_level=risk,
                             model_version="1.0.0")
        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)

        return ChurnPredictionResponse(
            customer_id=customer_id,
            churn_probability=prob,
            risk_level=risk,
            model_version="1.0.0",
            generated_at=record.created_at or dt.datetime.now(dt.timezone.utc),
        )
