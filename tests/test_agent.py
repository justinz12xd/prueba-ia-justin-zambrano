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


def test_agent_opens_ticket_from_conversation(client, auth_headers):
    """Un problema real descrito en el chat queda registrado como ticket."""
    resp = client.post("/api/v1/agent/chat", json={
        "message": "El internet se corta cada media hora desde el lunes y ya reinicié el router",
        "customer_id": 1,
    }, headers=auth_headers)
    assert resp.status_code == 200
    ticket = resp.json()["ticket"]
    assert ticket is not None
    assert ticket["category"] == "TECH"
    assert ticket["is_new"] is True
    assert ticket["ticket_id"] > 0


def test_agent_reuses_open_ticket_of_same_category(client, auth_headers):
    """Insistir sobre el mismo problema no debe abrir un ticket duplicado."""
    payload = {"message": "Sigo sin internet, se corta cada media hora desde el lunes",
               "customer_id": 1}
    first = client.post("/api/v1/agent/chat", json=payload, headers=auth_headers).json()
    second = client.post("/api/v1/agent/chat", json=payload, headers=auth_headers).json()

    assert first["ticket"]["category"] == "TECH"
    assert second["ticket"]["ticket_id"] == first["ticket"]["ticket_id"]
    assert second["ticket"]["is_new"] is False


def test_greeting_does_not_open_ticket(client, auth_headers):
    """Un saludo no es una solicitud de soporte."""
    resp = client.post("/api/v1/agent/chat", json={"message": "Hola, buenas tardes",
                                                    "customer_id": 1}, headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json()["ticket"] is None


def test_chat_persists_interaction_row(client, auth_headers):
    """La tabla `interaction` del esquema se alimenta con cada turno ligado a un ticket."""
    from app.core.database import SessionLocal
    from app.models.interaction import Interaction

    resp = client.post("/api/v1/agent/chat", json={
        "message": "Me cobraron dos veces el mismo mes y necesito que lo revisen",
        "customer_id": 1,
    }, headers=auth_headers)
    ticket_id = resp.json()["ticket"]["ticket_id"]

    db = SessionLocal()
    try:
        rows = db.query(Interaction).filter(Interaction.ticket_id == ticket_id).all()
        assert rows, "debería haberse registrado la interacción"
        row = rows[-1]
        assert row.customer_msg
        assert row.agent_response
        assert row.sentiment in {"positive", "neutral", "negative", None}
    finally:
        db.close()
