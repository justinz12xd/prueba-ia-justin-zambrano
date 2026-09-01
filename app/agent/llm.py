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

    # gemini-2.5-flash es un modelo de razonamiento: los tokens de "pensamiento"
    # se descuentan de max_output_tokens, así que con un presupuesto ajustado la
    # respuesta visible salía cortada a media frase. Aquí no hace falta razonar
    # (son respuestas cortas de atención al cliente), de modo que se desactiva el
    # pensamiento y se deja holgura para el texto.
    return ChatGoogleGenerativeAI(
        model=settings.GEMINI_MODEL,
        google_api_key=settings.GOOGLE_API_KEY,
        temperature=0.4,
        max_output_tokens=512,
        thinking_budget=0,
    )


def is_llm_available() -> bool:
    return get_llm() is not None
