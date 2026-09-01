def test_greeting_flow(client, auth_headers):
    resp = client.post("/api/v1/agent/chat", json={"message": "Hola, buenas tardes"},
                       headers=auth_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["intent"] == "greeting"
    assert body["escalate"] is False
    assert body["session_id"]


def test_conversation_persists_session(client, auth_headers):
    first = client.post("/api/v1/agent/chat", json={"message": "Hola"},
                        headers=auth_headers).json()
    session_id = first["session_id"]

    second = client.post("/api/v1/agent/chat",
                          json={"message": "Tengo un problema con mi factura, cobro duplicado",
                                "session_id": session_id},
                          headers=auth_headers)
    assert second.status_code == 200
    assert second.json()["session_id"] == session_id

    session_resp = client.get(f"/api/v1/agent/sessions/{session_id}", headers=auth_headers)
    assert session_resp.status_code == 200
    conversation = session_resp.json()["conversation"]
    assert len(conversation) == 4  # 2 turnos de usuario + 2 de asistente


def test_frustrated_message_escalates(client, auth_headers):
    resp = client.post("/api/v1/agent/chat", json={
        "message": "Esto es indignante, llevo tres días sin internet y nadie resuelve nada"
    }, headers=auth_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["escalate"] is True


def test_delete_session(client, auth_headers):
    created = client.post("/api/v1/agent/chat", json={"message": "Hola"},
                          headers=auth_headers).json()
    session_id = created["session_id"]

    resp = client.delete(f"/api/v1/agent/sessions/{session_id}", headers=auth_headers)
    assert resp.status_code == 204

    resp_get = client.get(f"/api/v1/agent/sessions/{session_id}", headers=auth_headers)
    assert resp_get.status_code == 404


def test_chat_requires_authentication(client):
    """Sin token, el agente no responde (401): mismo criterio que el resto de routers."""
    resp = client.post("/api/v1/agent/chat", json={"message": "Hola"})
    assert resp.status_code == 401


def test_session_history_requires_authentication(client, auth_headers):
    """El historial de una sesión no puede leerse ni borrarse de forma anónima."""
    created = client.post("/api/v1/agent/chat", json={"message": "Hola"},
                          headers=auth_headers).json()
    session_id = created["session_id"]

    assert client.get(f"/api/v1/agent/sessions/{session_id}").status_code == 401
    assert client.delete(f"/api/v1/agent/sessions/{session_id}").status_code == 401


def test_customer_role_cannot_delete_session(client, auth_headers, customer_headers):
    """El rol customer puede conversar y leer, pero no eliminar sesiones (403)."""
    created = client.post("/api/v1/agent/chat", json={"message": "Hola"},
                          headers=customer_headers).json()
    session_id = created["session_id"]

    assert client.get(f"/api/v1/agent/sessions/{session_id}",
                      headers=customer_headers).status_code == 200
    assert client.delete(f"/api/v1/agent/sessions/{session_id}",
                         headers=customer_headers).status_code == 403
    assert client.delete(f"/api/v1/agent/sessions/{session_id}",
                         headers=auth_headers).status_code == 204
