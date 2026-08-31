"""Estado del agente conversacional (LangGraph)."""
from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    messages: list[dict]          # historial de conversación: [{"role": "user"|"assistant", "content": str}, ...]
    customer_id: int | None       # identificador del cliente, si se conoce
    intent: str | None            # "greeting" | "farewell" | "account_query" |
                                   # "technical_support" | "general_info"
    context: dict[str, Any]       # info del cliente (plan, churn risk, etc.) + señales internas
    escalate: bool                # True si se debe escalar a un agente humano
    response: str | None          # respuesta final generada
    error: str | None             # mensaje de error interno, si algún nodo falló
