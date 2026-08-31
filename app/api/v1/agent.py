from __future__ import annotations

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.schemas.agent import ChatRequest, ChatResponse, SessionResponse
from app.services.agent_service import AgentService

router = APIRouter(prefix="/api/v1/agent", tags=["Agente Conversacional"])


@router.post("/chat", response_model=ChatResponse, summary="Conversar con el agente",
             description="Envía un mensaje al agente LangGraph. Si no se envía session_id, "
                         "se crea una nueva sesión persistida en agent_session.")
def chat(data: ChatRequest, db: Session = Depends(get_db)) -> ChatResponse:
    return AgentService(db).chat(data)


@router.get("/sessions/{session_id}", response_model=SessionResponse,
            summary="Obtener una sesión de conversación")
def get_session(session_id: str, db: Session = Depends(get_db)) -> SessionResponse:
    return AgentService(db).get_session(session_id)


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT,
               response_model=None, summary="Eliminar una sesión de conversación")
def delete_session(session_id: str, db: Session = Depends(get_db)) -> None:
    AgentService(db).delete_session(session_id)
