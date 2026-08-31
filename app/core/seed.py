"""Seed de datos mínimos al arrancar la app: categorías de ticket y usuarios
de prueba (uno por rol) para poder probar la API/Swagger de inmediato."""
from __future__ import annotations

import logging

from sqlalchemy.orm import Session

from app.core.security import hash_password
from app.models.customer import Customer
from app.models.ticket import TicketCategory
from app.models.user import UserAccount

logger = logging.getLogger("seed")

CATEGORY_SEED = [
    ("TECH", "Problemas técnicos", "Internet lento, sin conexión, fallas de router", 8),
    ("BILL", "Facturación", "Consultas y reclamos sobre facturas y cobros", 3),
    ("PLAN", "Planes y servicios", "Cambios de plan, upgrades, servicios adicionales", 4),
    ("CNCL", "Cancelación", "Solicitudes de baja o cancelación de servicio", 5),
    ("OTHR", "Otros", "Consultas generales no categorizadas", 2),
]

TEST_USERS = [
    ("admin@telecom.com", "Admin123!", "admin", None),
    ("agente@telecom.com", "Agente123!", "agent", None),
    ("cliente@telecom.com", "Cliente123!", "customer", 1),
]

DEMO_CUSTOMER = dict(
    name="Cliente Demo", email="cliente@telecom.com", phone="0991234567",
    plan_type="Fibra 200MB", monthly_charge=39.9, tenure_months=18, total_charges=718.2,
    contract_type="one-year", payment_method="credit_card", churn_status=0,
)


def run_seed(db: Session) -> None:
    if db.query(TicketCategory).count() == 0:
        for code, name, desc, avg_res in CATEGORY_SEED:
            db.add(TicketCategory(category_name=code, description=f"{name}: {desc}",
                                   avg_resolution=avg_res))
        logger.info("Categorías de ticket sembradas")

    if db.query(Customer).count() == 0:
        db.add(Customer(**DEMO_CUSTOMER))
        logger.info("Cliente demo sembrado")
    db.commit()

    if db.query(UserAccount).count() == 0:
        for email, password, role, customer_id in TEST_USERS:
            db.add(UserAccount(email=email, hashed_password=hash_password(password),
                                role=role, customer_id=customer_id))
        db.commit()
        logger.info("Usuarios de prueba sembrados: %s", [u[0] for u in TEST_USERS])
