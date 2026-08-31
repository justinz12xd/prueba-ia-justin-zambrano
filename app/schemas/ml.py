from __future__ import annotations

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


class ModelsInfoResponse(BaseModel):
    ticket_classifier: dict
    churn_model: dict
    sentiment_model: dict
    resolution_time_model: dict
