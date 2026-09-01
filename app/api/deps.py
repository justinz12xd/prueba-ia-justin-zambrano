"""Dependencias compartidas de la capa HTTP.

Aquí vive el control de **pertenencia** de los recursos, que es distinto del control
de rol. `require_roles(...)` responde a "¿puedes usar este endpoint?"; esto responde a
"¿son tuyos estos datos?". Sin lo segundo, cualquier cliente autenticado podía leer la
ficha, los tickets y las conversaciones de cualquier otro cliente con solo cambiar el
id de la URL.
"""
from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import TokenPayload, get_current_user
from app.repositories.user_repository import UserRepository

# Un cliente sin ficha asociada no debe poder ver los datos de nadie. Se le asigna un
# identificador que ningún cliente real puede tener, en vez de dejarlo sin restricción.
SIN_CLIENTE = -1


def get_customer_scope(user: TokenPayload = Depends(get_current_user),
                        db: Session = Depends(get_db)) -> int | None:
    """Alcance de datos del usuario autenticado.

    Devuelve `None` para el personal interno (admin/agent), que ve todo, y el
    `customer_id` para un cliente, que solo puede ver lo suyo.
    """
    if user.role in ("admin", "agent"):
        return None
    cuenta = UserRepository(db).get_by_email(user.sub)
    return cuenta.customer_id if cuenta and cuenta.customer_id else SIN_CLIENTE


CustomerScope = Annotated[int | None, Depends(get_customer_scope)]


def ensure_owner(scope: int | None, owner_id: int | None) -> None:
    """Corta el acceso si un cliente pide algo que no le pertenece.

    Se responde **404 y no 403** a propósito: un 403 confirmaría que ese cliente,
    ticket o sesión existe, que ya es información que no le corresponde.
    """
    if scope is not None and owner_id != scope:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                             detail="Recurso no encontrado")


def scoped_customer_id(scope: int | None, solicitado: int | None) -> int | None:
    """Fuerza el filtro por cliente cuando quien consulta es un cliente.

    Así `GET /tickets?customer_id=<ajeno>` devuelve los del propio solicitante en vez
    de los de otro, y un cliente no puede abrir tickets a nombre de terceros.
    """
    return scope if scope is not None else solicitado
