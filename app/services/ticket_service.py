"""Lógica de negocio de tickets, incluyendo la clasificación automática."""
from __future__ import annotations

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.ml_runtime import (churn_model, resolution_time_model, sentiment_model,
                            ticket_classifier)
from app.models.agent_session import AgentSession
from app.models.interaction import Interaction
from app.models.ticket import Ticket
from app.repositories.customer_repository import CustomerRepository
from app.repositories.ticket_repository import TicketRepository
from app.schemas.ticket import (ConversationMessage, TicketClassifyResponse,
                                TicketConversation, TicketCreate, TicketQueueItem,
                                TicketUpdate)


class TicketService:
    def __init__(self, db: Session):
        self.db = db
        self.repo = TicketRepository(db)
        self.customers = CustomerRepository(db)

    def list_tickets(self, skip: int, limit: int, customer_id: int | None) -> list[Ticket]:
        return self.repo.list(skip, limit, customer_id)

    def _ultimo_mensaje_por_ticket(self, ticket_ids: list[int]) -> dict[int, str]:
        """Último `customer_msg` registrado de cada ticket, en una sola consulta."""
        if not ticket_ids:
            return {}
        filas = self.db.execute(
            select(Interaction.ticket_id, Interaction.customer_msg)
            .where(Interaction.ticket_id.in_(ticket_ids))
            .order_by(Interaction.ticket_id, Interaction.interaction_id)
        ).all()
        # Al ir ordenadas de forma ascendente, la última asignación de cada ticket gana.
        # Se descartan los mensajes vacíos a propósito: las respuestas de un asesor
        # humano se guardan con customer_msg="" y no deben contar como texto del
        # cliente al evaluar su estado de ánimo.
        return {ticket_id: mensaje for ticket_id, mensaje in filas if mensaje}

    def queue(self, limit: int = 20, only_open: bool = True) -> list[TicketQueueItem]:
        """Bandeja de trabajo del agente: tickets + señales de los tres modelos.

        Cada modelo se consulta de forma tolerante a fallos: si uno no está
        disponible, el ticket igual aparece en la bandeja, solo que sin ese dato.
        """
        tickets = self.repo.list(0, limit, None, only_open=only_open)

        ultimos_mensajes = self._ultimo_mensaje_por_ticket([t.ticket_id for t in tickets])

        items: list[TicketQueueItem] = []
        for ticket in tickets:
            item = TicketQueueItem(
                ticket_id=ticket.ticket_id,
                customer_id=ticket.customer_id,
                customer_name=ticket.customer.name if ticket.customer else None,
                category=ticket.category,
                priority=ticket.priority,
                status=ticket.status,
                description=ticket.description,
                created_at=ticket.created_at,
                agent_session_id=ticket.agent_session_id,
            )
            # El sentimiento se evalúa sobre el ÚLTIMO mensaje del cliente en esa
            # conversación, no sobre la descripción del ticket. La descripción es el
            # primer mensaje —normalmente el más calmado—, así que un cliente que se
            # enfada más adelante aparecía en la bandeja como si estuviera tranquilo.
            texto_sentimiento = ultimos_mensajes.get(ticket.ticket_id) or ticket.description
            item.sentiment_source = ("last_message" if ticket.ticket_id in ultimos_mensajes
                                      else "description")
            try:
                sentiment, _, is_frustrated = sentiment_model.analyze_sentiment(texto_sentimiento)
                item.sentiment, item.is_frustrated = sentiment, is_frustrated
            except Exception:  # noqa: BLE001
                pass
            try:
                item.estimated_hours = round(resolution_time_model.predict_resolution_time(
                    ticket.category, ticket.priority, ticket.description,
                    ticket.created_at.hour, ticket.created_at.weekday()), 1)
            except Exception:  # noqa: BLE001
                pass
            if ticket.customer is not None:
                try:
                    from app.services.customer_service import CustomerService

                    num_tickets, avg_satisfaction = CustomerService(self.db).ticket_stats(
                        ticket.customer_id)
                    prob, risk = churn_model.predict_churn({
                        "tenure_months": ticket.customer.tenure_months,
                        "monthly_charge": ticket.customer.monthly_charge,
                        "total_charges": ticket.customer.total_charges,
                        "contract_type": ticket.customer.contract_type,
                        "payment_method": ticket.customer.payment_method,
                        "num_tickets": num_tickets,
                        "avg_satisfaction": avg_satisfaction,
                    })
                    item.churn_probability, item.churn_risk = round(prob, 4), risk
                except Exception:  # noqa: BLE001
                    pass
            items.append(item)
        return items

    def get_ticket_or_404(self, ticket_id: int) -> Ticket:
        ticket = self.repo.get(ticket_id)
        if ticket is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                 detail=f"Ticket {ticket_id} no encontrado")
        return ticket

    def classify(self, description: str) -> TicketClassifyResponse:
        try:
            category, probabilities = ticket_classifier.classify_ticket(description)
        except ticket_classifier.TicketClassifierUnavailable as exc:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                                 detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                                 detail=str(exc)) from exc
        return TicketClassifyResponse(predicted_category=category, probabilities=probabilities)

    def create(self, data: TicketCreate) -> Ticket:
        if self.customers.get(data.customer_id) is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                 detail=f"Cliente {data.customer_id} no encontrado")
        category = data.category
        if category is None:
            category, _ = ticket_classifier.classify_ticket(data.description)
        return self.repo.create(data.customer_id, category, data.description, data.priority)

    # ------------------------------------------------------------------
    # Traspaso a un asesor humano
    # ------------------------------------------------------------------

    def conversation(self, ticket_id: int) -> TicketConversation:
        """Hilo del que nació el ticket, para que un asesor pueda retomarlo."""
        ticket = self.get_ticket_or_404(ticket_id)
        session = (self.db.get(AgentSession, ticket.agent_session_id)
                   if ticket.agent_session_id else None)
        mensajes = [ConversationMessage(**m) for m in (session.conversation or [])] if session else []
        return TicketConversation(
            ticket_id=ticket.ticket_id,
            customer_id=ticket.customer_id,
            customer_name=ticket.customer.name if ticket.customer else None,
            session_id=ticket.agent_session_id,
            status=ticket.status,
            category=ticket.category,
            messages=mensajes,
        )

    def reply_as_human(self, ticket_id: int, mensaje: str, autor: str) -> TicketConversation:
        """Añade la respuesta de un asesor a la conversación del cliente.

        El mensaje se guarda en `agent_session.conversation` con el rol
        `agent_human`, para que el portal del cliente lo distinga del bot, y queda
        registrado en `interaction`. El ticket pasa a "en proceso": alguien lo tomó.
        """
        ticket = self.get_ticket_or_404(ticket_id)
        if not ticket.agent_session_id:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Este ticket no nació de una conversación, no hay chat que retomar",
            )
        session = self.db.get(AgentSession, ticket.agent_session_id)
        if session is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                 detail="La conversación asociada ya no existe")

        session.conversation = [*(session.conversation or []),
                                {"role": "agent_human", "content": mensaje, "author": autor}]
        self.db.add(Interaction(ticket_id=ticket.ticket_id, customer_msg="",
                                 agent_response=mensaje))
        if ticket.status == "open":
            ticket.status = "in_progress"
        self.db.commit()
        return self.conversation(ticket_id)

    def update(self, ticket_id: int, data: TicketUpdate) -> Ticket:
        ticket = self.get_ticket_or_404(ticket_id)
        return self.repo.update(ticket, data)
