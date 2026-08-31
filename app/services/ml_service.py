"""Fachada de servicios ML/DL para los endpoints /api/v1/ml/* y el protocolo MCP."""
from __future__ import annotations

from fastapi import HTTPException, status

from app.ml_runtime import churn_model, resolution_time_model, sentiment_model, ticket_classifier
from app.schemas.ml import (ChurnFeaturesInput, ChurnPredictionOutput, ModelsInfoResponse,
                             SentimentResponse)
from app.schemas.ticket import TicketClassifyResponse


class MLService:
    def classify_ticket(self, description: str) -> TicketClassifyResponse:
        try:
            category, probabilities = ticket_classifier.classify_ticket(description)
        except ticket_classifier.TicketClassifierUnavailable as exc:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
        return TicketClassifyResponse(predicted_category=category, probabilities=probabilities)

    def predict_churn(self, features: ChurnFeaturesInput) -> ChurnPredictionOutput:
        try:
            prob, risk = churn_model.predict_churn(features.model_dump())
        except churn_model.ChurnModelUnavailable as exc:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
        return ChurnPredictionOutput(churn_probability=prob, risk_level=risk,
                                      model_version="1.0.0")

    def analyze_sentiment(self, text: str) -> SentimentResponse:
        try:
            sentiment, probabilities, is_frustrated = sentiment_model.analyze_sentiment(text)
        except sentiment_model.SentimentModelUnavailable as exc:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
        return SentimentResponse(sentiment=sentiment, probabilities=probabilities,
                                  is_frustrated=is_frustrated)

    def predict_resolution_time(self, category: str, priority: str, description: str,
                                 hour_of_day: int, day_of_week: int) -> float:
        try:
            return resolution_time_model.predict_resolution_time(
                category, priority, description, hour_of_day, day_of_week)
        except resolution_time_model.ResolutionModelUnavailable as exc:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc

    def models_info(self) -> ModelsInfoResponse:
        return ModelsInfoResponse(
            ticket_classifier=ticket_classifier.get_model_info(),
            churn_model=churn_model.get_model_info(),
            sentiment_model=sentiment_model.get_model_info(),
            resolution_time_model=resolution_time_model.get_model_info(),
        )
