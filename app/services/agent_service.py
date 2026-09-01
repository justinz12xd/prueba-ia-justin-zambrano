"""Orquesta el agente LangGraph: reconstruye el estado desde la sesión persistida
en BD, invoca el grafo, guarda los mensajes nuevos y —cuando la conversación quedó
ligada a un ticket— registra la interacción en la tabla `interaction` del esquema."""
from __future__ import annotations

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.agent.graph import get_compiled_graph
from app.models.interaction import Interaction
from app.repositories.agent_session_repository import AgentSessionRepository
from app.schemas.agent import ChatRequest, ChatResponse, ChatTicket, SessionResponse


class AgentService:
    def __init__(self, db: Session):
        self.db = db
        self.sessions = AgentSessionRepository(db)

    def chat(self, data: ChatRequest) -> ChatResponse:
        session = self.sessions.get_or_create(data.session_id, data.customer_id)
        history: list[dict] = list(session.conversation or [])
        history.append({"role": "user", "content": data.message})

        graph = get_compiled_graph()
        initial_state = {
            "messages": history,
            "customer_id": data.customer_id or session.customer_id,
            "intent": None,
            "context": {},
            "escalate": False,
            "response": None,
            "error": None,
        }
        result = graph.invoke(initial_state, config={"configurable": {"db": self.db}})

        assistant_message = {"role": "assistant", "content": result.get("response", "")}
        approx_tokens = len(data.message.split()) + len(assistant_message["content"].split())
        self.sessions.append_messages(
            session, [{"role": "user", "content": data.message}, assistant_message],
            tokens_used_delta=approx_tokens,
        )

        context = result.get("context") or {}
        ticket_data = context.get("ticket")

        # La tabla `interaction` del esquema guarda el par mensaje/respuesta con el
        # sentimiento detectado y el tiempo estimado. Solo tiene sentido cuando la
        # conversación derivó en un ticket, porque interaction.ticket_id es obligatorio.
        if ticket_data:
            self.db.add(Interaction(
                ticket_id=ticket_data["ticket_id"],
                customer_msg=data.message,
                agent_response=assistant_message["content"],
                sentiment=context.get("sentiment"),
                resolution_time=ticket_data.get("estimated_hours"),
            ))
            self.db.commit()

        return ChatResponse(
            session_id=session.session_id,
            response=result.get("response", ""),
            intent=result.get("intent"),
            escalate=result.get("escalate", False),
            ticket=ChatTicket(**ticket_data) if ticket_data else None,
            customer_context=context,
        )

    def get_session(self, session_id: str) -> SessionResponse:
        session = self.sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                 detail=f"Sesión {session_id} no encontrada")
        return SessionResponse(
            session_id=session.session_id,
            customer_id=session.customer_id,
            conversation=session.conversation,
            tokens_used=session.tokens_used,
            started_at=session.started_at,
            ended_at=session.ended_at,
        )

    def delete_session(self, session_id: str) -> None:
        session = self.sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                 detail=f"Sesión {session_id} no encontrada")
        self.sessions.delete(session)
