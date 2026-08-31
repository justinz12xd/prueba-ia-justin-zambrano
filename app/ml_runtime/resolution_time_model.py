"""Wrapper de inferencia para el modelo de tiempo de resolución (Parte 2.2)."""
from __future__ import annotations

from functools import lru_cache

import joblib
import numpy as np

from app.core.config import get_settings

settings = get_settings()


class ResolutionModelUnavailable(Exception):
    pass


@lru_cache
def _load_artifacts():
    import tensorflow as tf

    model_path = settings.DL_MODELS_DIR / "resolution_time_model.keras"
    tokenizer_path = settings.DL_MODELS_DIR / "resolution_time_tokenizer.joblib"
    encoders_path = settings.DL_MODELS_DIR / "resolution_time_encoders.joblib"

    if not (model_path.exists() and tokenizer_path.exists() and encoders_path.exists()):
        raise ResolutionModelUnavailable(
            f"No se encontró el modelo en {settings.DL_MODELS_DIR}. "
            "Ejecute: python dl_training/train_resolution_time_model.py"
        )
    model = tf.keras.models.load_model(model_path)
    tokenizer = joblib.load(tokenizer_path)
    encoders = joblib.load(encoders_path)
    return model, tokenizer, encoders


def _cyclical(value: int, period: int) -> np.ndarray:
    radians = 2 * np.pi * value / period
    return np.array([[np.sin(radians), np.cos(radians)]])


def _one_hot(value: str, categories: list[str]) -> np.ndarray:
    idx = categories.index(value) if value in categories else 0
    return np.eye(len(categories))[idx][None, :]


def predict_resolution_time(category: str, priority: str, description: str,
                             hour_of_day: int, day_of_week: int) -> float:
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    model, tokenizer, encoders = _load_artifacts()
    max_len = encoders["max_len"]

    seq = tokenizer.texts_to_sequences([description])
    text_input = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")

    inputs = {
        "text_input": text_input,
        "category_input": _one_hot(category, encoders["categories"]),
        "priority_input": _one_hot(priority, encoders["priorities"]),
        "hour_input": _cyclical(hour_of_day, 24),
        "day_input": _cyclical(day_of_week, 7),
    }
    pred = model.predict(inputs, verbose=0)[0, 0]
    return max(0.25, float(pred))


def get_model_info() -> dict:
    try:
        model, _, encoders = _load_artifacts()
        return {
            "loaded": True,
            "categories": encoders["categories"],
            "priorities": encoders["priorities"],
            "params": int(model.count_params()),
        }
    except ResolutionModelUnavailable as exc:
        return {"loaded": False, "error": str(exc)}
