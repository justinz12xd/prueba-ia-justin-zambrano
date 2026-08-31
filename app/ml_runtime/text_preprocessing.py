"""Utilidades de preprocesamiento de texto compartidas entre el entrenamiento
(ml_training/train_ticket_classifier.py) y la inferencia (app/ml_runtime/ticket_classifier.py).

IMPORTANTE: `normalize_text` se usa como `preprocessor` dentro de un
`TfidfVectorizer` que luego se serializa con joblib. Debe vivir en un módulo
"real" e importable (no en un script ejecutado como __main__), de lo
contrario joblib/pickle no puede reconstruir la función al cargar el modelo
desde otro proceso (p. ej. la API).
"""
from __future__ import annotations

import re
import unicodedata

SPANISH_STOPWORDS = [
    "de", "la", "que", "el", "en", "y", "a", "los", "del", "se", "las", "por",
    "un", "para", "con", "no", "una", "su", "al", "lo", "como", "más", "pero",
    "sus", "le", "ya", "o", "este", "sí", "porque", "esta", "entre", "cuando",
    "muy", "sin", "sobre", "también", "me", "hasta", "hay", "donde", "quien",
    "desde", "todo", "nos", "durante", "todos", "uno", "les", "ni", "contra",
    "otros", "ese", "eso", "ante", "ellos", "e", "esto", "mí", "antes", "algunos",
    "qué", "unos", "yo", "otro", "otras", "otra", "él", "tanto", "esa", "estos",
    "mucho", "quienes", "nada", "muchos", "cual", "poco", "ella", "estar", "estas",
    "algunas", "algo", "nosotros", "mi", "mis", "tú", "te", "ti", "tu", "tus",
]


def normalize_text(text: str) -> str:
    """Limpieza de texto: minúsculas y quita caracteres no alfanuméricos,
    CONSERVANDO tildes y 'ñ' (importante para español)."""
    text = str(text).strip().lower()
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[^a-záéíóúüñ0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def validate_min_length(text: str, min_len: int = 10) -> str:
    if text is None or len(text.strip()) < min_len:
        raise ValueError(f"El texto de entrada debe tener mínimo {min_len} caracteres")
    return text
