"""Configuración centralizada de la aplicación (Pydantic Settings)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8",
                                       extra="ignore")

    # --- App ---
    APP_NAME: str = "Sistema Inteligente de Atención al Cliente"
    APP_VERSION: str = "1.0.0"
    ENV: str = "development"
    DEBUG: bool = True

    # --- Database ---
    # Por defecto apunta al servicio "db" de docker-compose. Para desarrollo local con
    # Supabase CLI, sobreescribir en .env con la cadena que entrega `supabase start`
    # (por ejemplo postgresql://postgres:postgres@127.0.0.1:54322/postgres).
    DATABASE_URL: str = "postgresql+psycopg2://postgres:postgres@db:5432/telecom_support"

    # --- JWT ---
    JWT_SECRET: str = "change-this-secret-in-production"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # --- LLM (agente conversacional) ---
    GOOGLE_API_KEY: str | None = None
    GEMINI_MODEL: str = "gemini-2.5-flash"

    # --- Modelos ML/DL ---
    ML_MODELS_DIR: Path = ROOT_DIR / "saved_models" / "ml"
    DL_MODELS_DIR: Path = ROOT_DIR / "saved_models" / "dl"

    # --- CORS ---
    CORS_ORIGINS: list[str] = ["*"]


@lru_cache
def get_settings() -> Settings:
    return Settings()
