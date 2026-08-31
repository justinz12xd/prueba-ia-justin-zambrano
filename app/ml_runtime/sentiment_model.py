"""Wrapper de inferencia para el modelo de sentimiento LSTM (Parte 2.1)."""
from __future__ import annotations

from functools import lru_cache

import joblib
import numpy as np

from app.core.config import get_settings

settings = get_settings()

FRUSTRATION_THRESHOLD = 0.6  # confianza mínima en "negative" para considerar frustración


class SentimentModelUnavailable(Exception):
    pass


@lru_cache
def _load_artifacts():
    import tensorflow as tf  # import perezoso: TF es pesado y solo se necesita aquí

    model_path = settings.DL_MODELS_DIR / "sentiment_model.keras"
    tokenizer_path = settings.DL_MODELS_DIR / "sentiment_tokenizer.joblib"
    encoder_path = settings.DL_MODELS_DIR / "sentiment_label_encoder.joblib"

    if not (model_path.exists() and tokenizer_path.exists() and encoder_path.exists()):
        raise SentimentModelUnavailable(
            f"No se encontró el modelo de sentimiento en {settings.DL_MODELS_DIR}. "
            "Ejecute: python dl_training/train_sentiment_model.py"
        )
    model = tf.keras.models.load_model(model_path)
    tokenizer = joblib.load(tokenizer_path)
    label_encoder = joblib.load(encoder_path)
    return model, tokenizer, label_encoder


def analyze_sentiment(text: str) -> tuple[str, dict[str, float], bool]:
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    model, tokenizer, label_encoder = _load_artifacts()

    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=200, padding="post", truncating="post")
    proba = model.predict(padded, verbose=0)[0]

    classes = label_encoder.classes_
    probabilities = {cls: float(p) for cls, p in zip(classes, proba)}
    predicted_idx = int(np.argmax(proba))
    predicted_label = classes[predicted_idx]

    is_frustrated = (
        predicted_label == "negative" and probabilities["negative"] >= FRUSTRATION_THRESHOLD
    )
    return predicted_label, probabilities, is_frustrated


def get_model_info() -> dict:
    try:
        model, _, label_encoder = _load_artifacts()
        return {
            "loaded": True,
            "classes": list(label_encoder.classes_),
            "params": int(model.count_params()),
        }
    except SentimentModelUnavailable as exc:
        return {"loaded": False, "error": str(exc)}
