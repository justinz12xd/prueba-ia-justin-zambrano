from __future__ import annotations

from fastapi import APIRouter, Depends

from app.core.security import require_roles
from app.schemas.ml import (ChurnFeaturesInput, ChurnPredictionOutput, ModelsInfoResponse,
                             ResolutionTimeRequest, ResolutionTimeResponse, SentimentRequest,
                             SentimentResponse)
from app.schemas.ticket import TicketClassifyRequest, TicketClassifyResponse
from app.services.ml_service import MLService

router = APIRouter(prefix="/api/v1/ml", tags=["Modelos ML"])
ml_service = MLService()


@router.post("/predict-churn", response_model=ChurnPredictionOutput,
             summary="Predecir churn a partir de features",
             description="No requiere que el cliente exista en BD; útil para simulaciones.",
             dependencies=[Depends(require_roles("admin", "agent"))])
def predict_churn(data: ChurnFeaturesInput) -> ChurnPredictionOutput:
    return ml_service.predict_churn(data)


@router.post("/classify-ticket", response_model=TicketClassifyResponse,
            description="Igual que /tickets/classify, expuesto bajo /ml para consumir el modelo de forma independiente del recurso ticket. Devuelve la categoría y la probabilidad de cada clase.",
             summary="Clasificar ticket (alias ML)",
             dependencies=[Depends(require_roles("admin", "agent", "customer"))])
def classify_ticket(data: TicketClassifyRequest) -> TicketClassifyResponse:
    return ml_service.classify_ticket(data.description)


@router.post("/analyze-sentiment", response_model=SentimentResponse,
            description="Clasifica un texto como positive/neutral/negative con la red LSTM. El campo is_frustrated (negative con confianza > 0.6) es la señal que el agente usa para escalar.",
             summary="Analizar sentimiento de un texto",
             dependencies=[Depends(require_roles("admin", "agent"))])
def analyze_sentiment(data: SentimentRequest) -> SentimentResponse:
    return ml_service.analyze_sentiment(data.text)


@router.post("/predict-resolution-time", response_model=ResolutionTimeResponse,
             summary="Estimar tiempo de resolución de un ticket",
             description="Red neuronal multi-input de la Parte 2.2: combina la descripción "
                         "(embeddings), la categoría y prioridad (one-hot) y la hora/día "
                         "(codificación cíclica) para estimar las horas hasta la resolución. "
                         "Si no se envían hora ni día, se usan los del momento actual. "
                         "Abierto al rol customer: el portal muestra al cliente el tiempo "
                         "estimado de respuesta de su propia solicitud.",
             dependencies=[Depends(require_roles("admin", "agent", "customer"))])
def predict_resolution_time(data: ResolutionTimeRequest) -> ResolutionTimeResponse:
    return ml_service.predict_resolution_time(data)


@router.get("/models/info", response_model=ModelsInfoResponse,
            summary="Información de los modelos cargados",
            description="Metadatos de los 4 modelos entrenados (ML y DL): si están "
                        "cargados, versión, clases, número de parámetros, etc.")
def models_info() -> ModelsInfoResponse:
    return ml_service.models_info()
