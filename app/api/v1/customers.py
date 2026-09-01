from __future__ import annotations

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import require_roles
from app.schemas.common import PaginatedResponse
from app.schemas.customer import (ChurnPredictionResponse, CustomerCreate, CustomerResponse,
                                   CustomerUpdate)
from app.services.customer_service import CustomerService

router = APIRouter(prefix="/api/v1/customers", tags=["Clientes"])


@router.get("", response_model=PaginatedResponse[CustomerResponse],
            description="Devuelve los clientes activos con paginación (skip/limit). Los clientes eliminados lógicamente (is_active=false) no aparecen.",
            summary="Listar clientes",
            dependencies=[Depends(require_roles("admin", "agent"))])
def list_customers(skip: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=200),
                    db: Session = Depends(get_db)) -> PaginatedResponse[CustomerResponse]:
    items, total = CustomerService(db).list_customers(skip, limit)
    return PaginatedResponse(items=items, total=total, page=skip // limit + 1, page_size=limit)


@router.get("/{customer_id}", response_model=CustomerResponse,
            description="Obtiene un cliente por su identificador. Retorna 404 si no existe o si fue eliminado lógicamente.",
            summary="Obtener un cliente",
            dependencies=[Depends(require_roles("admin", "agent", "customer"))])
def get_customer(customer_id: int, db: Session = Depends(get_db)) -> CustomerResponse:
    return CustomerService(db).get_customer_or_404(customer_id)


@router.post("", response_model=CustomerResponse, status_code=status.HTTP_201_CREATED,
            description="Crea un cliente. El email debe ser único (409 si ya existe) y el teléfono debe empezar con 09 y tener al menos 10 dígitos.",
             summary="Crear cliente", dependencies=[Depends(require_roles("admin", "agent"))])
def create_customer(data: CustomerCreate, db: Session = Depends(get_db)) -> CustomerResponse:
    return CustomerService(db).create(data)


@router.put("/{customer_id}", response_model=CustomerResponse,
            description="Actualiza parcialmente un cliente: solo se modifican los campos enviados.",
            summary="Actualizar cliente", dependencies=[Depends(require_roles("admin", "agent"))])
def update_customer(customer_id: int, data: CustomerUpdate,
                     db: Session = Depends(get_db)) -> CustomerResponse:
    return CustomerService(db).update(customer_id, data)


@router.delete("/{customer_id}", response_model=CustomerResponse,
               summary="Eliminar cliente (lógico)",
               description="Marca is_active=False y deleted_at; no borra el registro físicamente.",
               dependencies=[Depends(require_roles("admin"))])
def delete_customer(customer_id: int, db: Session = Depends(get_db)) -> CustomerResponse:
    return CustomerService(db).soft_delete(customer_id)


@router.get("/{customer_id}/churn-prediction", response_model=ChurnPredictionResponse,
            summary="Predicción de churn del cliente",
            description="Calcula la probabilidad de abandono con el modelo entrenado y "
                        "persiste el resultado en la tabla prediction.",
            dependencies=[Depends(require_roles("admin", "agent"))])
def churn_prediction(customer_id: int, db: Session = Depends(get_db)) -> ChurnPredictionResponse:
    return CustomerService(db).predict_churn_for_customer(customer_id)
