from __future__ import annotations

import datetime as dt
from typing import Literal

from pydantic import BaseModel, Field


class ChurnFeaturesInput(BaseModel):
    tenure_months: int = Field(..., ge=0, examples=[14])
    monthly_charge: float = Field(..., ge=0, examples=[42.5])
    total_charges: float = Field(..., ge=0, examples=[595.0])
    contract_type: Literal["month-to-month", "one-year", "two-year"] = "month-to-month"
    payment_method: Literal["credit_card", "bank_transfer", "cash"] = "credit_card"
    num_tickets: int = Field(0, ge=0, examples=[3])
    avg_satisfaction: float = Field(3.5, ge=1, le=5, examples=[2.8])


class ChurnPredictionOutput(BaseModel):
    churn_probability: float
    risk_level: Literal["low", "medium", "high"]
    model_version: str


class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=3, examples=["Estoy muy molesto con el servicio"])


class SentimentResponse(BaseModel):
    sentiment: Literal["positive", "neutral", "negative"]
    probabilities: dict[str, float]
    is_frustrated: bool = Field(
        ..., description="True si sentiment == negative con alta confianza (usado para escalar)"
    )


class ResolutionTimeRequest(BaseModel):
    """Entrada de la red multi-input de la Parte 2.2 (texto + categóricas + temporales)."""

    description: str = Field(
        ..., min_length=10, max_length=500,
        examples=["El router pierde la señal cada media hora desde el lunes"],
    )
    category: Literal["TECH", "BILL", "PLAN", "CNCL", "OTHR"] = Field(..., examples=["TECH"])
    priority: Literal["low", "medium", "high"] = Field("medium", examples=["high"])
    hour_of_day: int | None = Field(
        None, ge=0, le=23,
        description="Hora del día (0-23). Si se omite, se usa la hora actual UTC.",
        examples=[14],
    )
    day_of_week: int | None = Field(
        None, ge=0, le=6,
        description="Día de la semana (0=lunes ... 6=domingo). Si se omite, se usa el día actual.",
        examples=[2],
    )


class ResolutionTimeResponse(BaseModel):
    estimated_hours: float = Field(..., description="Tiempo estimado de resolución, en horas")
    estimated_resolution_at: dt.datetime = Field(
        ..., description="Marca temporal estimada de resolución (ahora + estimated_hours)"
    )
    category: str
    priority: str
    hour_of_day: int
    day_of_week: int
    model_version: str


class ModelsInfoResponse(BaseModel):
    ticket_classifier: dict
    churn_model: dict
    sentiment_model: dict
    resolution_time_model: dict
