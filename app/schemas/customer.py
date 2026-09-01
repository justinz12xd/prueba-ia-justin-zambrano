from __future__ import annotations

import datetime as dt
import re
from typing import Literal

from pydantic import BaseModel, EmailStr, Field, field_validator

PHONE_REGEX = re.compile(r"^09\d{8,}$")  # empieza con 09, solo dígitos, mínimo 10 en total
ContractType = Literal["month-to-month", "one-year", "two-year"]
PaymentMethod = Literal["credit_card", "bank_transfer", "cash"]


class CustomerBase(BaseModel):
    name: str = Field(..., min_length=2, max_length=120, examples=["María Pérez"])
    email: EmailStr = Field(..., examples=["maria.perez@example.com"])
    phone: str = Field(..., examples=["0991234567"])
    plan_type: str = Field(..., examples=["Fibra 200MB"])
    monthly_charge: float = Field(..., ge=0, examples=[35.5])
    tenure_months: int = Field(0, ge=0)
    total_charges: float = Field(0, ge=0)
    contract_type: ContractType = "month-to-month"
    payment_method: PaymentMethod = "credit_card"

    @field_validator("phone")
    @classmethod
    def validate_phone(cls, v: str) -> str:
        if not PHONE_REGEX.match(v):
            raise ValueError(
                "El teléfono debe contener solo números, empezar con '09' y tener "
                "mínimo 10 dígitos en total"
            )
        return v


class CustomerCreate(CustomerBase):
    pass


class CustomerUpdate(BaseModel):
    name: str | None = Field(None, min_length=2, max_length=120)
    email: EmailStr | None = None
    phone: str | None = None
    plan_type: str | None = None
    monthly_charge: float | None = Field(None, ge=0)
    tenure_months: int | None = Field(None, ge=0)
    total_charges: float | None = Field(None, ge=0)
    contract_type: ContractType | None = None
    payment_method: PaymentMethod | None = None

    @field_validator("phone")
    @classmethod
    def validate_phone(cls, v: str | None) -> str | None:
        if v is not None and not PHONE_REGEX.match(v):
            raise ValueError(
                "El teléfono debe contener solo números, empezar con '09' y tener "
                "mínimo 10 dígitos en total"
            )
        return v


class CustomerResponse(CustomerBase):
    customer_id: int
    churn_status: int
    is_active: bool
    created_at: dt.datetime

    model_config = {
        "from_attributes": True,
        "json_schema_extra": {"examples": [{
        "customer_id": 1,
        "name": "María Pérez",
        "email": "maria.perez@example.com",
        "phone": "0991234567",
        "plan_type": "Fibra 200MB",
        "monthly_charge": 35.5,
        "tenure_months": 18,
        "total_charges": 639.0,
        "contract_type": "month-to-month",
        "payment_method": "credit_card",
        "churn_status": 0,
        "is_active": True,
        "created_at": "2026-03-14T10:25:00Z",
    }]},
    }


class ChurnPredictionResponse(BaseModel):
    customer_id: int
    churn_probability: float = Field(..., ge=0, le=1)
    risk_level: Literal["low", "medium", "high"]
    model_version: str
    generated_at: dt.datetime

    model_config = {"json_schema_extra": {"examples": [{
        "customer_id": 1,
        "churn_probability": 0.6831,
        "risk_level": "high",
        "model_version": "1.0.0",
        "generated_at": "2026-08-31T18:40:12Z",
    }]}}
