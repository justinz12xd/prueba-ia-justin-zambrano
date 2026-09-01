"""Punto de entrada de la API — Sistema Inteligente de Atención al Cliente."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
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
            "demo": "/demo", "docs": "/docs"}


# Demo web (opcional): UI estática que consume la propia API. Se monta solo si la
# carpeta existe, para que la app arranque igual en entornos que no la copien.
_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
if _STATIC_DIR.is_dir():
    app.mount("/demo", StaticFiles(directory=_STATIC_DIR, html=True), name="demo")
