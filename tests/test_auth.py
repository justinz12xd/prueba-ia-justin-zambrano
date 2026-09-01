def test_login_success(client):
    resp = client.post("/api/v1/auth/login",
                        json={"email": "admin@telecom.com", "password": "Admin123!"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["role"] == "admin"
    assert body["access_token"]
    assert body["refresh_token"]


def test_login_invalid_credentials(client):
    resp = client.post("/api/v1/auth/login",
                        json={"email": "admin@telecom.com", "password": "wrong-password"})
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "UNAUTHORIZED"


def test_refresh_token(client, admin_token):
    login = client.post("/api/v1/auth/login",
                         json={"email": "admin@telecom.com", "password": "Admin123!"})
    refresh_token = login.json()["refresh_token"]
    resp = client.post("/api/v1/auth/refresh", json={"refresh_token": refresh_token})
    assert resp.status_code == 200
    assert resp.json()["access_token"]


def test_protected_endpoint_without_token(client):
    resp = client.get("/api/v1/customers")
    assert resp.status_code == 401


def test_demo_page_is_served(client):
    """El panel técnico se sirve desde la propia API."""
    resp = client.get("/demo")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "Sistema Inteligente de Atención al Cliente" in resp.text


def test_portal_page_is_served(client):
    """El centro de ayuda para clientes se sirve desde la propia API."""
    resp = client.get("/portal")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "Centro de Ayuda" in resp.text


def test_me_returns_linked_customer(client):
    """/auth/me resuelve el cliente asociado: el portal lo necesita para sus tickets."""
    token = client.post("/api/v1/auth/login", json={
        "email": "cliente@telecom.com", "password": "Cliente123!"}).json()["access_token"]
    resp = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["email"] == "cliente@telecom.com"
    assert body["role"] == "customer"
    assert body["customer_id"] == 1
    assert body["customer_name"]


def test_me_requires_authentication(client):
    assert client.get("/api/v1/auth/me").status_code == 401
