"""Endpoints que implementan el protocolo MCP (Model Context Protocol) para que
otros agentes de IA puedan interactuar con el sistema.

Convención seguida (estilo JSON-RPC 2.0, igual que el enunciado):
  - Los errores de ejecución de una tool se devuelven con HTTP 200 y
    `result.isError = true` (así lo hace el protocolo MCP real: el error es
    parte del resultado, no un fallo de transporte).
  - Un error de transporte real (payload inválido, tool inexistente) se
    reporta igual dentro del sobre JSON-RPC, con isError=true, para mantener
    el contrato de respuesta consistente para el agente que consume el MCP.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.ml_runtime import churn_model, ticket_classifier
from app.schemas.mcp import (MCPCapabilitiesResponse, MCPContentItem, MCPResourceDescriptor,
                              MCPResponse, MCPResult, MCPToolDescriptor, ToolExecuteRequest)
from app.services.agent_service import AgentService
from app.services.ticket_service import TicketService

router = APIRouter(prefix="/mcp", tags=["MCP"])

# ---------------------------------------------------------------------------
# Definición de tools disponibles
# ---------------------------------------------------------------------------

TOOL_DESCRIPTORS = [
    MCPToolDescriptor(
        name="predict_churn",
        description="Predice la probabilidad de abandono (churn) de un cliente. Acepta "
                    "'customer_id' (usa datos reales de BD) o features manuales.",
        input_schema={
            "type": "object",
            "properties": {
                "customer_id": {"type": "integer"},
                "tenure_months": {"type": "integer"},
                "monthly_charge": {"type": "number"},
                "total_charges": {"type": "number"},
                "contract_type": {"type": "string"},
                "payment_method": {"type": "string"},
                "num_tickets": {"type": "integer"},
                "avg_satisfaction": {"type": "number"},
            },
        },
    ),
    MCPToolDescriptor(
        name="classify_ticket",
        description="Clasifica la descripción de un ticket en TECH/BILL/PLAN/CNCL/OTHR.",
        input_schema={"type": "object", "properties": {"description": {"type": "string"}},
                      "required": ["description"]},
    ),
    MCPToolDescriptor(
        name="get_customer_info",
        description="Obtiene la información de un cliente por su customer_id.",
        input_schema={"type": "object", "properties": {"customer_id": {"type": "integer"}},
                      "required": ["customer_id"]},
    ),
    MCPToolDescriptor(
        name="create_ticket",
        description="Crea un nuevo ticket de soporte para un cliente.",
        input_schema={
            "type": "object",
            "properties": {
                "customer_id": {"type": "integer"},
                "description": {"type": "string"},
                "category": {"type": "string"},
                "priority": {"type": "string"},
            },
            "required": ["customer_id", "description"],
        },
    ),
    MCPToolDescriptor(
        name="chat_with_agent",
        description="Inicia o continúa una conversación con el agente conversacional (LangGraph).",
        input_schema={
            "type": "object",
            "properties": {
                "message": {"type": "string"},
                "session_id": {"type": "string"},
                "customer_id": {"type": "integer"},
            },
            "required": ["message"],
        },
    ),
]

RESOURCE_DESCRIPTORS = [
    MCPResourceDescriptor(id="ticket_categories", name="Categorías de tickets",
                          description="Lista de categorías soportadas por el clasificador."),
    MCPResourceDescriptor(id="models_info", name="Información de modelos",
                          description="Metadatos de los modelos ML/DL cargados en el servidor."),
]


@router.get("/capabilities", response_model=MCPCapabilitiesResponse,
            summary="Capacidades del servidor MCP")
def capabilities() -> MCPCapabilitiesResponse:
    return MCPCapabilitiesResponse(tools=TOOL_DESCRIPTORS, resources=RESOURCE_DESCRIPTORS)


@router.get("/resources", response_model=list[MCPResourceDescriptor],
            summary="Lista recursos disponibles")
def list_resources() -> list[MCPResourceDescriptor]:
    return RESOURCE_DESCRIPTORS


@router.get("/resources/{resource_id}", response_model=MCPResponse,
            summary="Obtiene un recurso específico")
def get_resource(resource_id: str) -> MCPResponse:
    if resource_id == "ticket_categories":
        data = {"categories": ["TECH", "BILL", "PLAN", "CNCL", "OTHR"]}
        return MCPResponse(id=resource_id, result=MCPResult(
            content=[MCPContentItem(type="json", data=data)], isError=False))
    if resource_id == "models_info":
        from app.services.ml_service import MLService
        data = MLService().models_info().model_dump()
        return MCPResponse(id=resource_id, result=MCPResult(
            content=[MCPContentItem(type="json", data=data)], isError=False))

    return MCPResponse(id=resource_id, result=MCPResult(
        content=[MCPContentItem(type="text", text=f"Recurso '{resource_id}' no encontrado")],
        isError=True))


# ---------------------------------------------------------------------------
# Ejecución de tools
# ---------------------------------------------------------------------------

def _tool_predict_churn(args: dict, db: Session) -> dict:
    if "customer_id" in args and args["customer_id"] is not None:
        from app.services.customer_service import CustomerService
        result = CustomerService(db).predict_churn_for_customer(int(args["customer_id"]))
        return result.model_dump(mode="json")
    prob, risk = churn_model.predict_churn(args)
    return {"churn_probability": prob, "risk_level": risk, "model_version": "1.0.0"}


def _tool_classify_ticket(args: dict, db: Session) -> dict:
    category, probabilities = ticket_classifier.classify_ticket(args["description"])
    return {"predicted_category": category, "probabilities": probabilities}


def _tool_get_customer_info(args: dict, db: Session) -> dict:
    from app.services.customer_service import CustomerService
    customer = CustomerService(db).get_customer_or_404(int(args["customer_id"]))
    return {
        "customer_id": customer.customer_id,
        "name": customer.name,
        "email": customer.email,
        "plan_type": customer.plan_type,
        "contract_type": customer.contract_type,
        "monthly_charge": customer.monthly_charge,
        "tenure_months": customer.tenure_months,
        "churn_status": customer.churn_status,
    }


def _tool_create_ticket(args: dict, db: Session) -> dict:
    from app.schemas.ticket import TicketCreate
    ticket_in = TicketCreate(
        customer_id=int(args["customer_id"]),
        description=args["description"],
        category=args.get("category"),
        priority=args.get("priority", "medium"),
    )
    ticket = TicketService(db).create(ticket_in)
    return {"ticket_id": ticket.ticket_id, "category": ticket.category, "status": ticket.status}


def _tool_chat_with_agent(args: dict, db: Session) -> dict:
    from app.schemas.agent import ChatRequest
    chat_result = AgentService(db).chat(ChatRequest(
        message=args["message"], session_id=args.get("session_id"),
        customer_id=args.get("customer_id"),
    ))
    return chat_result.model_dump(mode="json")


TOOL_HANDLERS = {
    "predict_churn": _tool_predict_churn,
    "classify_ticket": _tool_classify_ticket,
    "get_customer_info": _tool_get_customer_info,
    "create_ticket": _tool_create_ticket,
    "chat_with_agent": _tool_chat_with_agent,
}


@router.post("/tools/execute", response_model=MCPResponse, summary="Ejecuta una herramienta MCP")
def execute_tool(request: ToolExecuteRequest, db: Session = Depends(get_db)) -> MCPResponse:
    handler = TOOL_HANDLERS.get(request.tool)
    if handler is None:
        return MCPResponse(id=request.id, result=MCPResult(
            content=[MCPContentItem(type="text",
                                    text=f"Tool '{request.tool}' no reconocida. Disponibles: "
                                         f"{list(TOOL_HANDLERS)}")],
            isError=True))
    try:
        data = handler(request.arguments, db)
        return MCPResponse(id=request.id, result=MCPResult(
            content=[MCPContentItem(type="json", data=data)], isError=False))
    except Exception as exc:  # noqa: BLE001 - el error se reporta dentro del sobre MCP
        return MCPResponse(id=request.id, result=MCPResult(
            content=[MCPContentItem(type="text", text=str(exc))], isError=True))
