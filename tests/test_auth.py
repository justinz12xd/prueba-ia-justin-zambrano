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
    """La UI de demo se sirve desde la propia API (mount /demo)."""
    resp = client.get("/demo/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "Sistema Inteligente de Atención al Cliente" in resp.text
