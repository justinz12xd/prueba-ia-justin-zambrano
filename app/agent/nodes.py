"""Nodos del grafo del agente conversacional.

Cada nodo recibe el AgentState y (opcionalmente) el RunnableConfig de LangGraph,
desde el cual se obtiene la sesión de base de datos inyectada por
`app.services.agent_service` en `config["configurable"]["db"]`.

IMPORTANTE: el parámetro debe anotarse exactamente como `RunnableConfig`, sin
unión ni valor por defecto. LangGraph inspecciona la firma para decidir si
inyecta el config, y con `RunnableConfig | None = None` no lo hace: los nodos
recibirían `config=None` y se quedarían sin acceso a la base de datos.

Todos los nodos están envueltos en try/except: si algo falla, se registra el
error en `state["error"]` y se degrada con valores por defecto seguros en vez
de abortar el grafo completo (manejo de errores y estados inválidos).
"""
from __future__ import annotations

import datetime as dt
import logging
import re

from langchain_core.runnables import RunnableConfig

from app.agent.llm import get_llm
from app.agent.state import AgentState
from app.ml_runtime import (churn_model, resolution_time_model, sentiment_model,
                            ticket_classifier)

logger = logging.getLogger("agent")


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)

# El saludo se busca SOLO al principio del mensaje y se consume junto con la
# puntuación que lo sigue, para poder separarlo del contenido real. Las variantes
# largas van dentro de un grupo opcional ("buenas" + "tardes") en vez de como
# alternativas sueltas: con alternativas, "buenas" ganaría y dejaría "tardes,"
# pegado al resto del mensaje.
GREETING_PATTERNS = re.compile(
    r"^\s*(?:hola|hey|qu[eé] tal|buen(?:os|as)(?:\s+(?:d[ií]as|tardes|noches))?)"
    r"[\s,;:.!¡¿?-]*",
    re.IGNORECASE,
)
FAREWELL_PATTERNS = re.compile(
    r"\b(adi[oó]s|hasta luego|gracias.*(nada m[aá]s|eso es todo)|chao|nos vemos|"
    r"eso ser[ií]a todo|hasta pronto)\b",
    re.IGNORECASE,
)

# Mínimo de caracteres que el clasificador de tickets acepta (ver text_preprocessing).
MIN_CLASIFICABLE = 10
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


def _separar_saludo(texto: str) -> tuple[str, str]:
    """Divide el mensaje en (saludo inicial, resto).

    Hace falta porque la gente saluda antes de contar su problema: "Buenas tardes,
    el internet no me funciona" es un ticket técnico, no un saludo. Si se tratara
    como saludo, el grafo saltaría los handlers y no se registraría nada.
    """
    resto = (texto or "").strip()
    saludos: list[str] = []
    # En bucle, porque la gente encadena saludos: "Hola, buenas tardes, ...".
    # Sin esto quedaría "buenas tardes" como si fuera el contenido del mensaje.
    while (match := GREETING_PATTERNS.match(resto)) and match.end() > 0:
        saludos.append(match.group(0).strip(" ,;:.!¡¿?-"))
        resto = resto[match.end():].strip()
    return " ".join(s for s in saludos if s), resto


def _last_user_message(state: AgentState) -> str:
    messages = state.get("messages") or []
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""


# ---------------------------------------------------------------------------
# Nodo: classify_intent
# ---------------------------------------------------------------------------

def classify_intent(state: AgentState, config: RunnableConfig) -> AgentState:
    text = _last_user_message(state)
    context = dict(state.get("context") or {})

    try:
        saludo, resto = _separar_saludo(text)
        despedida = FAREWELL_PATTERNS.search(text or "")

        if saludo and len(resto) < MIN_CLASIFICABLE:
            # El mensaje es SOLO un saludo ("hola", "buenas tardes").
            intent = "greeting"
        elif despedida and len(text.strip()) < 40:
            # Despedida breve. En un mensaje largo, "gracias" o "adiós" suelen ser
            # cortesía dentro de una consulta real, no el final de la conversación.
            intent = "farewell"
        elif len(resto) < MIN_CLASIFICABLE:
            # Demasiado corto para el clasificador -> se trata como info general
            intent = "general_info"
        else:
            # Se clasifica el CONTENIDO, sin el saludo: así "Buenas tardes, no tengo
            # internet" llega al clasificador como "no tengo internet".
            category, probabilities = ticket_classifier.classify_ticket(resto)
            intent = CATEGORY_TO_INTENT.get(category, "general_info")
            context["predicted_category"] = category
            context["category_probabilities"] = probabilities
            if saludo:
                context["greeting_prefix"] = saludo

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

def get_customer_info(state: AgentState, config: RunnableConfig) -> AgentState:
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

        # Churn de alto riesgo -> se usa luego en check_escalation.
        # Los tickets y la satisfacción salen de CustomerService.ticket_stats, que en
        # PostgreSQL resuelve con la función SQL fn_customer_churn_summary.
        try:
            from app.services.customer_service import CustomerService

            num_tickets, avg_satisfaction = CustomerService(db).ticket_stats(customer_id)
            prob, risk = churn_model.predict_churn({
                "tenure_months": customer.tenure_months,
                "monthly_charge": customer.monthly_charge,
                "total_charges": customer.total_charges,
                "contract_type": customer.contract_type,
                "payment_method": customer.payment_method,
                "num_tickets": num_tickets,
                "avg_satisfaction": avg_satisfaction,
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

MIN_TICKET_DESCRIPTION = 20  # mismo mínimo que exige el schema TicketCreate


def _priority_from_context(context: dict) -> str:
    """Prioridad sugerida a partir de las señales del mensaje."""
    if context.get("is_frustrated") or context.get("churn_risk") == "high":
        return "high"
    if context.get("is_cancellation"):
        return "high"
    return "medium"


def _open_ticket(state: AgentState, config: RunnableConfig | None, context: dict) -> dict:
    """Registra la solicitud del cliente como ticket, si corresponde.

    Reglas:
      - hace falta un cliente identificado y una sesión de BD;
      - el mensaje debe tener al menos 20 caracteres (mismo mínimo que la API);
      - si el cliente ya tiene un ticket abierto de esa categoría, se reutiliza en
        vez de duplicarlo: insistir sobre el mismo problema no abre tickets nuevos.

    Es best-effort: cualquier fallo queda en el contexto y la conversación sigue.
    """
    db = (config or {}).get("configurable", {}).get("db") if config else None
    customer_id = state.get("customer_id")
    text = _last_user_message(state).strip()
    category = context.get("predicted_category")

    if db is None or not customer_id or not category or len(text) < MIN_TICKET_DESCRIPTION:
        return context

    try:
        from app.repositories.ticket_repository import TicketRepository

        repo = TicketRepository(db)
        existing = repo.find_open_by_category(customer_id, category)
        if existing is not None:
            ticket, is_new = existing, False
        else:
            ticket = repo.create(customer_id, category, text[:500],
                                 _priority_from_context(context))
            is_new = True

        estimated_hours = None
        try:
            estimated_hours = round(resolution_time_model.predict_resolution_time(
                category, ticket.priority, text, _now().hour, _now().weekday()), 2)
        except Exception as exc:  # noqa: BLE001 - la estimación es opcional
            logger.warning("No se pudo estimar el tiempo de resolución: %s", exc)

        context["ticket"] = {
            "ticket_id": ticket.ticket_id,
            "category": ticket.category,
            "priority": ticket.priority,
            "status": ticket.status,
            "is_new": is_new,
            "estimated_hours": estimated_hours,
        }
    except Exception as exc:  # noqa: BLE001 - nunca romper la conversación por el ticket
        logger.exception("Error registrando el ticket desde el agente")
        context["ticket_error"] = str(exc)
    return context


def handle_account_query(state: AgentState, config: RunnableConfig) -> AgentState:
    context = dict(state.get("context") or {})
    context["handler"] = "account_query"
    if context.get("predicted_category") == "CNCL":
        context["is_cancellation"] = True
    context = _open_ticket(state, config, context)
    return {**state, "context": context}


def handle_technical_support(state: AgentState, config: RunnableConfig) -> AgentState:
    context = dict(state.get("context") or {})
    context["handler"] = "technical_support"
    context = _open_ticket(state, config, context)
    return {**state, "context": context}


def handle_general_info(state: AgentState, config: RunnableConfig) -> AgentState:
    context = dict(state.get("context") or {})
    context["handler"] = "general_info"
    return {**state, "context": context}


# ---------------------------------------------------------------------------
# Nodo: check_escalation
# ---------------------------------------------------------------------------

def check_escalation(state: AgentState, config: RunnableConfig) -> AgentState:
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
        ticket = context.get("ticket")
        if ticket:
            base += (f" Tu solicitud #{ticket['ticket_id']} ya está registrada."
                     if ticket["is_new"] else
                     f" Lo sumé a tu solicitud #{ticket['ticket_id']}.")
        return base

    ticket = context.get("ticket")
    referencia = ""
    if ticket:
        referencia = (f" Quedó registrada como la solicitud #{ticket['ticket_id']}."
                      if ticket["is_new"]
                      else f" La sumé a tu solicitud #{ticket['ticket_id']}, que sigue abierta.")

    if intent == "technical_support":
        return ("Lamento el inconveniente técnico. Un especialista revisará tu conexión y "
                f"te contactaremos con una solución.{referencia}")
    if intent == "account_query":
        name = context.get("customer_name")
        saludo = f"{name}, " if name else ""
        return (f"{saludo}he tomado tu consulta de cuenta/facturación. "
                f"En breve un agente confirmará los detalles contigo.{referencia}")
    return "Gracias por tu mensaje, en breve un agente revisará tu consulta."


def generate_response(state: AgentState, config: RunnableConfig) -> AgentState:
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
            + (f"Se registró la solicitud de soporte #{context['ticket']['ticket_id']} "
               f"para este caso; menciónala en la respuesta.\n"
               if context.get("ticket") and context["ticket"].get("is_new") else "")
            + (f"El caso se sumó a la solicitud #{context['ticket']['ticket_id']}, que "
               f"ya estaba abierta; menciónalo.\n"
               if context.get("ticket") and not context["ticket"].get("is_new") else "")
            + "Si se debe escalar, indícalo amablemente en la respuesta. "
            "Si es un saludo, saluda y pregunta en qué puedes ayudar. "
            "Si es una despedida, despídete cordialmente. "
            "No inventes datos concretos (precios, horarios, fechas) que no te hayan dado."
        )
        history = state.get("messages") or []
        recent = history[-6:]
        lc_messages = [SystemMessage(content=system_prompt)]
        for m in recent:
            lc_messages.append(HumanMessage(content=f"[{m.get('role')}] {m.get('content')}"))

        result = llm.invoke(lc_messages)
        response_text = (result.content if hasattr(result, "content") else str(result)) or ""
        if not response_text.strip():
            # El LLM puede devolver vacío si agota el presupuesto de tokens; en ese
            # caso es preferible una plantilla correcta a un mensaje en blanco.
            logger.warning("El LLM devolvió una respuesta vacía, uso plantilla")
            return {**state, "response": _template_response(state)}
        return {**state, "response": response_text.strip()}
    except Exception as exc:  # noqa: BLE001 - nunca debe tumbar el endpoint /agent/chat
        logger.exception("Error llamando al LLM, uso respuesta de fallback")
        return {**state, "response": _template_response(state), "error": str(exc)}
