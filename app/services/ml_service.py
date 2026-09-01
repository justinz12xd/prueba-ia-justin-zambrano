"""Fachada de servicios ML/DL para los endpoints /api/v1/ml/* y el protocolo MCP."""
from __future__ import annotations

import datetime as dt

from fastapi import HTTPException, status

from app.ml_runtime import churn_model, resolution_time_model, sentiment_model, ticket_classifier
from app.schemas.ml import (ChurnFeaturesInput, ChurnPredictionOutput, ModelsInfoResponse,
                             ResolutionTimeRequest, ResolutionTimeResponse, SentimentResponse)
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

    def predict_resolution_time(self, data: ResolutionTimeRequest) -> ResolutionTimeResponse:
        """Estima el tiempo de resolución de un ticket (Parte 2.2).

        `hour_of_day` y `day_of_week` son opcionales: si no se envían se toman del
        momento actual, que es el caso de uso real (estimar al crear el ticket).
        """
        now = dt.datetime.now(dt.timezone.utc)
        hour_of_day = data.hour_of_day if data.hour_of_day is not None else now.hour
        day_of_week = data.day_of_week if data.day_of_week is not None else now.weekday()

        try:
            hours = resolution_time_model.predict_resolution_time(
                data.category, data.priority, data.description, hour_of_day, day_of_week)
        except resolution_time_model.ResolutionModelUnavailable as exc:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc

        return ResolutionTimeResponse(
            estimated_hours=round(hours, 2),
            estimated_resolution_at=now + dt.timedelta(hours=hours),
            category=data.category,
            priority=data.priority,
            hour_of_day=hour_of_day,
            day_of_week=day_of_week,
            model_version="1.0.0",
        )

    def models_info(self) -> ModelsInfoResponse:
        return ModelsInfoResponse(
            ticket_classifier=ticket_classifier.get_model_info(),
            churn_model=churn_model.get_model_info(),
            sentiment_model=sentiment_model.get_model_info(),
            resolution_time_model=resolution_time_model.get_model_info(),
        )
