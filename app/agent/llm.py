"""Wrapper del LLM (Google Gemini) usado por el nodo generate_response.

Si no hay GOOGLE_API_KEY configurada, el agente sigue funcionando con
respuestas basadas en plantillas (modo degradado), para no depender de una
llamada externa en tests/CI.
"""
from __future__ import annotations

from functools import lru_cache

from app.core.config import get_settings

settings = get_settings()


@lru_cache
def get_llm():
    if not settings.GOOGLE_API_KEY:
        return None
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=settings.GEMINI_MODEL,
        google_api_key=settings.GOOGLE_API_KEY,
        temperature=0.4,
        max_output_tokens=400,
    )


def is_llm_available() -> bool:
    return get_llm() is not None
