"""Wrapper de inferencia para el modelo de churn (Parte 1.2)."""
from __future__ import annotations

from functools import lru_cache

import joblib
import pandas as pd

from app.core.config import get_settings

settings = get_settings()
MODEL_VERSION = "1.0.0"


class ChurnModelUnavailable(Exception):
    pass


@lru_cache
def _load_pipeline():
    path = settings.ML_MODELS_DIR / "churn_model.joblib"
    if not path.exists():
        raise ChurnModelUnavailable(
            f"No se encontró el modelo entrenado en {path}. "
            "Ejecute: python ml_training/train_churn_model.py"
        )
    return joblib.load(path)


def _risk_level(prob: float) -> str:
    if prob < 0.33:
        return "low"
    if prob < 0.66:
        return "medium"
    return "high"


def predict_churn(features: dict) -> tuple[float, str]:
    """features debe incluir: tenure_months, monthly_charge, total_charges,
    contract_type, payment_method, num_tickets, avg_satisfaction.
    Aplica el mismo feature engineering usado en el entrenamiento."""
    pipeline = _load_pipeline()

    tenure = max(features.get("tenure_months", 0), 0)
    total_charges = features.get("total_charges", 0.0)
    num_tickets = features.get("num_tickets", 0)

    row = {
        "tenure_months": tenure,
        "monthly_charge": features.get("monthly_charge", 0.0),
        "total_charges": total_charges,
        "num_tickets": num_tickets,
        "avg_satisfaction": features.get("avg_satisfaction", 3.5),
        "charge_per_tenure": total_charges / (tenure or 1),
        "tickets_per_tenure": num_tickets / ((tenure or 1) / 12),
        "contract_type": features.get("contract_type", "month-to-month"),
        "payment_method": features.get("payment_method", "credit_card"),
    }
    df = pd.DataFrame([row])
    prob = float(pipeline.predict_proba(df)[0, 1])
    return prob, _risk_level(prob)


def get_model_info() -> dict:
    try:
        pipeline = _load_pipeline()
        return {
            "loaded": True,
            "model_version": MODEL_VERSION,
            "steps": [name for name, _ in pipeline.steps],
        }
    except ChurnModelUnavailable as exc:
        return {"loaded": False, "error": str(exc)}
