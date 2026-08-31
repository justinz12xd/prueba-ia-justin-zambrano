"""Nodos del grafo del agente conversacional.

Cada nodo recibe el AgentState y (opcionalmente) el RunnableConfig de LangGraph,
desde el cual se obtiene la sesión de base de datos inyectada por
`app.services.agent_service` en `config["configurable"]["db"]`.

Todos los nodos están envueltos en try/except: si algo falla, se registra el
error en `state["error"]` y se degrada con valores por defecto seguros en vez
de abortar el grafo completo (manejo de errores y estados inválidos).
"""
from __future__ import annotations

import logging
import re

from langchain_core.runnables import RunnableConfig

from app.agent.llm import get_llm
from app.agent.state import AgentState
from app.ml_runtime import churn_model, sentiment_model, ticket_classifier

logger = logging.getLogger("agent")

GREETING_PATTERNS = re.compile(
    r"^\s*(hola|buenas|buenos d[ií]as|buenas tardes|buenas noches|hey|qu[eé] tal)\b",
    re.IGNORECASE,
)
FAREWELL_PATTERNS = re.compile(
    r"\b(adi[oó]s|hasta luego|gracias.*(nada m[aá]s|eso es todo)|chao|nos vemos|"
    r"eso ser[ií]a todo|hasta pronto)\b",
    re.IGNORECASE,
)
HUMAN_REQUEST_PATTERNS = re.compile(
    r"\b(hablar con (un |una )?(humano|persona|agente|supervisor)|"
    r"quiero un (agente|supervisor) real|no quiero hablar con un bot)\b",
    re.IGNORECASE,
)

CATEGORY_TO_INTENT = {
    "TECH": "technical_support",
    "BILL": "account_query",
    "PLAN": "account_query",
    "CNCL": "account_query",
    "OTHR": "general_info",
}


def _last_user_message(state: AgentState) -> str:
    messages = state.get("messages") or []
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""


# ---------------------------------------------------------------------------
# Nodo: classify_intent
# ---------------------------------------------------------------------------

def classify_intent(state: AgentState, config: RunnableConfig | None = None) -> AgentState:
    text = _last_user_message(state)
    context = dict(state.get("context") or {})

    try:
        if GREETING_PATTERNS.search(text or ""):
            intent = "greeting"
        elif FAREWELL_PATTERNS.search(text or ""):
            intent = "farewell"
        else:
            if len(text.strip()) < 10:
                # Muy corto para el clasificador ML -> se trata como info general
                intent = "general_info"
            else:
                category, probabilities = ticket_classifier.classify_ticket(text)
                intent = CATEGORY_TO_INTENT.get(category, "general_info")
                context["predicted_category"] = category
                context["category_probabilities"] = probabilities

        # Señal de frustración vía modelo de sentimiento (Parte 2.1)
        try:
            sentiment, sent_proba, is_frustrated = sentiment_model.analyze_sentiment(text or "")
            context["sentiment"] = sentiment
            context["sentiment_probabilities"] = sent_proba
            context["is_frustrated"] = is_frustrated
        except sentiment_model.SentimentModelUnavailable as exc:
            logger.warning("Modelo de sentimiento no disponible: %s", exc)
            context["sentiment"] = "unknown"
            context["is_frustrated"] = False

        context["wants_human"] = bool(HUMAN_REQUEST_PATTERNS.search(text or ""))

        return {**state, "intent": intent, "context": context, "error": None}
    except Exception as exc:  # noqa: BLE001 - nodo robusto ante fallos de ML
        logger.exception("Error en classify_intent")
        return {**state, "intent": "general_info", "context": context, "error": str(exc)}


# ---------------------------------------------------------------------------
# Nodo: get_customer_info
# ---------------------------------------------------------------------------

def get_customer_info(state: AgentState, config: RunnableConfig | None = None) -> AgentState:
    context = dict(state.get("context") or {})
    customer_id = state.get("customer_id")

    if not customer_id:
        context["customer_found"] = False
        return {**state, "context": context}

    db = (config or {}).get("configurable", {}).get("db") if config else None
    if db is None:
        context["customer_found"] = False
        context["customer_info_error"] = "Sin sesión de base de datos disponible"
        return {**state, "context": context}

    try:
        from app.repositories.customer_repository import CustomerRepository

        customer = CustomerRepository(db).get(customer_id)
        if customer is None:
            context["customer_found"] = False
            return {**state, "context": context}

        context["customer_found"] = True
        context["customer_name"] = customer.name
        context["plan_type"] = customer.plan_type
        context["monthly_charge"] = customer.monthly_charge
        context["tenure_months"] = customer.tenure_months
        context["contract_type"] = customer.contract_type

        # Churn de alto riesgo -> se usa luego en check_escalation
        try:
            prob, risk = churn_model.predict_churn({
                "tenure_months": customer.tenure_months,
                "monthly_charge": customer.monthly_charge,
                "total_charges": customer.total_charges,
                "contract_type": customer.contract_type,
                "payment_method": customer.payment_method,
                "num_tickets": len(customer.tickets or []),
                "avg_satisfaction": 3.5,
            })
            context["churn_probability"] = prob
            context["churn_risk"] = risk
        except churn_model.ChurnModelUnavailable as exc:
            logger.warning("Modelo de churn no disponible: %s", exc)

        return {**state, "context": context, "error": None}
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error en get_customer_info")
        context["customer_found"] = False
        return {**state, "context": context, "error": str(exc)}


# ---------------------------------------------------------------------------
# Nodos de manejo por intención
# ---------------------------------------------------------------------------

def handle_account_query(state: AgentState, config: RunnableConfig | None = None) -> AgentState:
    context = dict(state.get("context") or {})
    context["handler"] = "account_query"
    if context.get("predicted_category") == "CNCL":
        context["is_cancellation"] = True
    return {**state, "context": context}


def handle_technical_support(state: AgentState, config: RunnableConfig | None = None) -> AgentState:
    context = dict(state.get("context") or {})
    context["handler"] = "technical_support"
    return {**state, "context": context}


def handle_general_info(state: AgentState, config: RunnableConfig | None = None) -> AgentState:
    context = dict(state.get("context") or {})
    context["handler"] = "general_info"
    return {**state, "context": context}


# ---------------------------------------------------------------------------
# Nodo: check_escalation
# ---------------------------------------------------------------------------

def check_escalation(state: AgentState, config: RunnableConfig | None = None) -> AgentState:
    context = dict(state.get("context") or {})

    escalate = bool(
        context.get("wants_human")
        or context.get("is_frustrated")
        or context.get("is_cancellation")
        or context.get("churn_risk") == "high"
    )
    if escalate:
        reasons = []
        if context.get("wants_human"):
            reasons.append("el cliente solicitó explícitamente hablar con una persona")
        if context.get("is_frustrated"):
            reasons.append("se detectó frustración en el mensaje")
        if context.get("is_cancellation"):
            reasons.append("es una solicitud de cancelación de servicio")
        if context.get("churn_risk") == "high":
            reasons.append("el cliente tiene alto riesgo de abandono (churn)")
        context["escalation_reason"] = "; ".join(reasons)

    return {**state, "context": context, "escalate": escalate}


# ---------------------------------------------------------------------------
# Nodo: generate_response
# ---------------------------------------------------------------------------

TEMPLATE_RESPONSES = {
    "greeting": "¡Hola! Soy el asistente virtual de atención al cliente. ¿En qué puedo ayudarte hoy?",
    "farewell": "¡Gracias por contactarnos! Que tengas un excelente día. 👋",
}


def _template_response(state: AgentState) -> str:
    intent = state.get("intent")
    context = state.get("context") or {}

    if intent in TEMPLATE_RESPONSES:
        return TEMPLATE_RESPONSES[intent]

    if state.get("escalate"):
        base = ("Entiendo tu situación y voy a escalar tu caso a un agente humano "
                "para que te ayude de inmediato. ")
        if context.get("is_cancellation"):
            base += "Un especialista de retención se comunicará contigo en breve."
        return base

    if intent == "technical_support":
        return ("Lamento el inconveniente técnico. He registrado tu caso; un especialista "
                "de soporte revisará tu conexión y te contactaremos con una solución.")
    if intent == "account_query":
        name = context.get("customer_name")
        saludo = f"{name}, " if name else ""
        return (f"{saludo}he registrado tu consulta de cuenta/facturación. "
                "En breve un agente confirmará los detalles contigo.")
    return "Gracias por tu mensaje, en breve un agente revisará tu consulta."


def generate_response(state: AgentState, config: RunnableConfig | None = None) -> AgentState:
    llm = get_llm()
    if llm is None:
        return {**state, "response": _template_response(state)}

    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        context = state.get("context") or {}
        system_prompt = (
            "Eres un agente virtual de atención al cliente de una empresa de "
            "telecomunicaciones. Responde en español, de forma breve (máximo 4 líneas), "
            "cordial y profesional.\n"
            f"Intención detectada: {state.get('intent')}\n"
            f"¿Debe escalarse a un humano?: {state.get('escalate')}"
            + (f" (motivo: {context.get('escalation_reason')})" if state.get("escalate") else "")
            + "\n"
            + (f"Cliente: {context.get('customer_name')}, plan {context.get('plan_type')}, "
               f"{context.get('tenure_months')} meses de antigüedad.\n"
               if context.get("customer_found") else "Cliente no identificado.\n")
            + "Si se debe escalar, indícalo amablemente en la respuesta. "
            "Si es un saludo, saluda y pregunta en qué puedes ayudar. "
            "Si es una despedida, despídete cordialmente."
        )
        history = state.get("messages") or []
        recent = history[-6:]
        lc_messages = [SystemMessage(content=system_prompt)]
        for m in recent:
            lc_messages.append(HumanMessage(content=f"[{m.get('role')}] {m.get('content')}"))

        result = llm.invoke(lc_messages)
        response_text = result.content if hasattr(result, "content") else str(result)
        return {**state, "response": response_text}
    except Exception as exc:  # noqa: BLE001 - nunca debe tumbar el endpoint /agent/chat
        logger.exception("Error llamando al LLM, uso respuesta de fallback")
        return {**state, "response": _template_response(state), "error": str(exc)}
