from __future__ import annotations

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import TokenPayload, get_current_user, require_roles
from app.schemas.ticket import (AgentReplyRequest, TicketClassifyRequest,
                                 TicketClassifyResponse, TicketConversation,
                                 TicketCreate, TicketQueueItem, TicketResponse,
                                 TicketUpdate)
from app.services.ticket_service import TicketService

router = APIRouter(prefix="/api/v1/tickets", tags=["Tickets"])


@router.get("", response_model=list[TicketResponse], summary="Listar tickets",
            description="Lista los tickets activos. Con el parámetro customer_id se filtran los de un cliente concreto.",
            dependencies=[Depends(require_roles("admin", "agent", "customer"))])
def list_tickets(skip: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=200),
                  customer_id: int | None = None,
                  db: Session = Depends(get_db)) -> list[TicketResponse]:
    return TicketService(db).list_tickets(skip, limit, customer_id)


@router.get("/queue", response_model=list[TicketQueueItem],
            summary="Bandeja de trabajo del agente",
            description="Tickets sin resolver enriquecidos con las señales de los modelos: "
                        "sentimiento del cliente, riesgo de churn y tiempo estimado de "
                        "resolución. Se declara antes de /{ticket_id} para que la ruta "
                        "literal gane sobre la paramétrica.",
            dependencies=[Depends(require_roles("admin", "agent"))])
def tickets_queue(limit: int = Query(20, ge=1, le=50), only_open: bool = True,
                   db: Session = Depends(get_db)) -> list[TicketQueueItem]:
    return TicketService(db).queue(limit, only_open)


@router.get("/{ticket_id}", response_model=TicketResponse, summary="Obtener un ticket",
            description="Obtiene un ticket por su identificador. Retorna 404 si no existe o fue eliminado lógicamente.",
            dependencies=[Depends(require_roles("admin", "agent", "customer"))])
def get_ticket(ticket_id: int, db: Session = Depends(get_db)) -> TicketResponse:
    return TicketService(db).get_ticket_or_404(ticket_id)


@router.post("", response_model=TicketResponse, status_code=status.HTTP_201_CREATED,
             summary="Crear ticket",
             description="Si no se envía 'category', se infiere automáticamente con el "
                         "clasificador ML a partir de la descripción.",
             dependencies=[Depends(require_roles("admin", "agent", "customer"))])
def create_ticket(data: TicketCreate, db: Session = Depends(get_db)) -> TicketResponse:
    return TicketService(db).create(data)


@router.put("/{ticket_id}", response_model=TicketResponse, summary="Actualizar ticket",
            description="Actualiza el estado, la prioridad, la categoría o la satisfacción. Al pasar a resolved/closed se registra automáticamente resolved_at.",
            dependencies=[Depends(require_roles("admin", "agent"))])
def update_ticket(ticket_id: int, data: TicketUpdate,
                   db: Session = Depends(get_db)) -> TicketResponse:
    return TicketService(db).update(ticket_id, data)


@router.post("/classify", response_model=TicketClassifyResponse,
             summary="Clasificar descripción de ticket",
             description="Clasifica un texto libre en una de las categorías "
                         "TECH/BILL/PLAN/CNCL/OTHR sin crear un ticket.",
             dependencies=[Depends(require_roles("admin", "agent", "customer"))])
def classify_ticket(data: TicketClassifyRequest, db: Session = Depends(get_db)) -> TicketClassifyResponse:
    return TicketService(db).classify(data.description)


@router.get("/{ticket_id}/conversation", response_model=TicketConversation,
            summary="Conversación que originó el ticket",
            description="Devuelve el hilo completo con el cliente para que un asesor pueda "
                        "retomarlo desde la bandeja. Los mensajes traen su rol: 'user' el "
                        "cliente, 'assistant' el agente virtual y 'agent_human' un asesor.",
            dependencies=[Depends(require_roles("admin", "agent"))])
def ticket_conversation(ticket_id: int, db: Session = Depends(get_db)) -> TicketConversation:
    return TicketService(db).conversation(ticket_id)


@router.post("/{ticket_id}/reply", response_model=TicketConversation,
             summary="Responder al cliente como asesor humano",
             description="Añade un mensaje del asesor a la conversación del cliente, lo "
                         "registra en `interaction` y pasa el ticket a 'en proceso'. "
                         "Devuelve 409 si el ticket no nació de una conversación.",
             dependencies=[Depends(require_roles("admin", "agent"))])
def ticket_reply(ticket_id: int, data: AgentReplyRequest,
                  user: TokenPayload = Depends(get_current_user),
                  db: Session = Depends(get_db)) -> TicketConversation:
    return TicketService(db).reply_as_human(ticket_id, data.message, autor=user.sub)
