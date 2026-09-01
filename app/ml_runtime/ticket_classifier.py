"""Wrapper de inferencia para el clasificador de tickets (Parte 1.1)."""
from __future__ import annotations

from functools import lru_cache

import joblib

from app.core.config import get_settings
# Import necesario aunque no se referencie explícitamente: `normalize_text` es el
# `preprocessor` serializado dentro del TfidfVectorizer del pipeline entrenado, y
# joblib debe poder resolverlo por esta misma ruta de módulo al deserializar.
from app.ml_runtime.text_preprocessing import normalize_text, validate_min_length  # noqa: F401

settings = get_settings()
CATEGORIES = ["TECH", "BILL", "PLAN", "CNCL", "OTHR"]


class TicketClassifierUnavailable(Exception):
    pass


@lru_cache
def _load_pipeline():
    path = settings.ML_MODELS_DIR / "ticket_classifier.joblib"
    if not path.exists():
        raise TicketClassifierUnavailable(
            f"No se encontró el modelo entrenado en {path}. "
            "Ejecute: python ml_training/train_ticket_classifier.py"
        )
    return joblib.load(path)


def get_pipeline():
    """Pipeline entrenado (TF-IDF + clasificador), cacheado por proceso.

    Público para que scripts de evaluación puedan usar `predict` en lote en vez de
    llamar a `classify_ticket` texto por texto.
    """
    return _load_pipeline()


def classify_ticket(description: str) -> tuple[str, dict[str, float]]:
    """Retorna (categoría_predicha, {categoria: probabilidad})."""
    validate_min_length(description)
    pipeline = _load_pipeline()
    predicted = pipeline.predict([description])[0]
    proba = pipeline.predict_proba([description])[0]
    classes = pipeline.classes_
    probabilities = {cls: float(p) for cls, p in zip(classes, proba)}
    # asegura que todas las categorías conocidas estén presentes
    for cat in CATEGORIES:
        probabilities.setdefault(cat, 0.0)
    return predicted, probabilities


def get_model_info() -> dict:
    try:
        pipeline = _load_pipeline()
        return {
            "loaded": True,
            "categories": list(pipeline.classes_),
            "steps": [name for name, _ in pipeline.steps],
        }
    except TicketClassifierUnavailable as exc:
        return {"loaded": False, "error": str(exc)}
