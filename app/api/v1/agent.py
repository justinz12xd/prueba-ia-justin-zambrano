from __future__ import annotations

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.api.deps import CustomerScope, ensure_owner
from app.core.security import require_roles
from app.schemas.agent import ChatRequest, ChatResponse, SessionResponse
from app.services.agent_service import AgentService

router = APIRouter(prefix="/api/v1/agent", tags=["Agente Conversacional"])


@router.post("/chat", response_model=ChatResponse, summary="Conversar con el agente",
             description="Envía un mensaje al agente LangGraph. Si no se envía session_id, "
                         "se crea una nueva sesión persistida en agent_session. "
                         "Requiere autenticación (cualquier rol).",
             dependencies=[Depends(require_roles("admin", "agent", "customer"))])
def chat(data: ChatRequest, scope: CustomerScope,
          db: Session = Depends(get_db)) -> ChatResponse:
    # Un cliente solo puede conversar en su propio nombre: si envía otro customer_id
    # se le sustituye por el suyo, en vez de dejarle consultar datos ajenos.
    if scope is not None:
        data = data.model_copy(update={"customer_id": scope})
    return AgentService(db).chat(data)


@router.get("/sessions/{session_id}", response_model=SessionResponse,
            summary="Obtener una sesión de conversación",
            description="Devuelve el historial completo de la conversación. "
                        "Requiere autenticación (cualquier rol).",
            dependencies=[Depends(require_roles("admin", "agent", "customer"))])
def get_session(session_id: str, scope: CustomerScope,
                 db: Session = Depends(get_db)) -> SessionResponse:
    sesion = AgentService(db).get_session(session_id)
    ensure_owner(scope, sesion.customer_id)
    return sesion


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT,
               response_model=None, summary="Eliminar una sesión de conversación",
               description="Elimina la sesión y su historial. Restringido a personal "
                           "interno (admin/agent), igual que el resto de eliminaciones.",
               dependencies=[Depends(require_roles("admin", "agent"))])
def delete_session(session_id: str, db: Session = Depends(get_db)) -> None:
    AgentService(db).delete_session(session_id)
