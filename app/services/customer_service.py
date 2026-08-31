"""Lógica de negocio de clientes, incluyendo la predicción de churn persistida."""
from __future__ import annotations

import datetime as dt

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.ml_runtime import churn_model
from app.models.customer import Customer
from app.models.prediction import Prediction
from app.models.ticket import Ticket
from app.repositories.customer_repository import CustomerRepository
from app.schemas.customer import ChurnPredictionResponse, CustomerCreate, CustomerUpdate


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

    def predict_churn_for_customer(self, customer_id: int) -> ChurnPredictionResponse:
        customer = self.get_customer_or_404(customer_id)
        num_tickets = (
            self.db.query(Ticket)
            .filter(Ticket.customer_id == customer_id, Ticket.is_active.is_(True))
            .count()
        )
        features = {
            "tenure_months": customer.tenure_months,
            "monthly_charge": customer.monthly_charge,
            "total_charges": customer.total_charges,
            "contract_type": customer.contract_type,
            "payment_method": customer.payment_method,
            "num_tickets": num_tickets,
            "avg_satisfaction": 3.5,
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
