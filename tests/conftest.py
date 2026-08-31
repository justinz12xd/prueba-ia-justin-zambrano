"""Fixtures compartidas de pytest.

Usa SQLite en un archivo temporal (en vez del Postgres de docker-compose) para
que los tests corran rápido y sin infraestructura externa. Se fija
DATABASE_URL ANTES de importar cualquier módulo de la app, porque
`get_settings()` está cacheado con `lru_cache`.
"""
from __future__ import annotations

import os
import tempfile

import pytest

_TMP_DB = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
os.environ["DATABASE_URL"] = f"sqlite:///{_TMP_DB.name}"
os.environ.setdefault("JWT_SECRET", "test-secret")

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c


def _login(client, email: str, password: str) -> str:
    resp = client.post("/api/v1/auth/login", json={"email": email, "password": password})
    assert resp.status_code == 200, resp.text
    return resp.json()["access_token"]


@pytest.fixture(scope="session")
def admin_token(client):
    return _login(client, "admin@telecom.com", "Admin123!")


@pytest.fixture(scope="session")
def agent_token(client):
    return _login(client, "agente@telecom.com", "Agente123!")


@pytest.fixture(scope="session")
def auth_headers(admin_token):
    return {"Authorization": f"Bearer {admin_token}"}
