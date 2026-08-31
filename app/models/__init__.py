"""Importa todos los modelos para que se registren en Base.metadata
(necesario para Base.metadata.create_all)."""
from app.models.agent_session import AgentSession
from app.models.customer import Customer
from app.models.interaction import Interaction
from app.models.prediction import Prediction
from app.models.ticket import Ticket, TicketCategory
from app.models.user import UserAccount

__all__ = [
    "AgentSession",
    "Customer",
    "Interaction",
    "Prediction",
    "Ticket",
    "TicketCategory",
    "UserAccount",
]
