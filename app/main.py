"""Punto de entrada de la API — Sistema Inteligente de Atención al Cliente."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import app.models  # noqa: F401  registra todos los modelos en Base.metadata
from app.api.v1 import agent, auth, customers, ml, tickets
from app.core.config import get_settings
from app.core.database import Base, SessionLocal, engine
from app.core.exceptions import register_exception_handlers
from app.core.seed import run_seed
from app.mcp.router import router as mcp_router

logging.basicConfig(level=logging.INFO)
settings = get_settings()


def _aplicar_sql_init() -> None:
    """Instala la función y el procedimiento PL/pgSQL de `sql/init.sql`.

    Con docker-compose el script lo carga Postgres al crear el volumen, pero en una
    base gestionada (Railway, Render, Supabase) nadie lo ejecuta. Se aplica aquí de
    forma idempotente —el script usa IF NOT EXISTS y CREATE OR REPLACE— y sin
    interrumpir el arranque si algo falla: la app funciona igual sin ellos.
    """
    if engine.dialect.name != "postgresql":
        return
    ruta = Path(__file__).resolve().parent.parent / "sql" / "init.sql"
    if not ruta.is_file():
        return
    try:
        with engine.begin() as conexion:
            conexion.exec_driver_sql(ruta.read_text(encoding="utf-8"))
        logging.getLogger("startup").info("sql/init.sql aplicado (funciones SQL disponibles)")
    except Exception as exc:  # noqa: BLE001 - nunca debe impedir que la API arranque
        logging.getLogger("startup").warning("No se pudo aplicar sql/init.sql: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    _aplicar_sql_init()
    db = SessionLocal()
    try:
        run_seed(db)
    finally:
        db.close()
    yield


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "API REST para clasificación de tickets, predicción de churn, análisis de "
        "sentimiento, un agente conversacional (LangGraph + Gemini) y un servidor "
        "MCP para que otros agentes de IA interactúen con el sistema."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

register_exception_handlers(app)

app.include_router(auth.router)
app.include_router(customers.router)
app.include_router(tickets.router)
app.include_router(ml.router)
app.include_router(agent.router)
app.include_router(mcp_router)


@app.get("/", tags=["Health"], summary="Health check")
def health():
    return {"status": "ok", "app": settings.APP_NAME, "version": settings.APP_VERSION,
            "portal": "/portal", "demo": "/demo", "docs": "/docs"}


# UIs estáticas (opcionales): consumen la propia API. Se registran solo si la carpeta
# existe, para que la app arranque igual en entornos que no la copien.
#   /portal -> centro de ayuda para clientes (simulación del producto)
#   /demo   -> panel técnico con los 4 modelos y el MCP
_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
if _STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

    @app.get("/portal", tags=["Demo"], summary="Centro de ayuda para clientes",
             include_in_schema=False)
    def portal() -> FileResponse:
        return FileResponse(_STATIC_DIR / "portal.html")

    @app.get("/demo", tags=["Demo"], summary="Panel técnico de modelos",
             include_in_schema=False)
    def demo() -> FileResponse:
        return FileResponse(_STATIC_DIR / "index.html")
