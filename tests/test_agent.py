def test_greeting_flow(client):
    resp = client.post("/api/v1/agent/chat", json={"message": "Hola, buenas tardes"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["intent"] == "greeting"
    assert body["escalate"] is False
    assert body["session_id"]


def test_conversation_persists_session(client):
    first = client.post("/api/v1/agent/chat", json={"message": "Hola"}).json()
    session_id = first["session_id"]

    second = client.post("/api/v1/agent/chat",
                          json={"message": "Tengo un problema con mi factura, cobro duplicado",
                                "session_id": session_id})
    assert second.status_code == 200
    assert second.json()["session_id"] == session_id

    session_resp = client.get(f"/api/v1/agent/sessions/{session_id}")
    assert session_resp.status_code == 200
    conversation = session_resp.json()["conversation"]
    assert len(conversation) == 4  # 2 turnos de usuario + 2 de asistente


def test_frustrated_message_escalates(client):
    resp = client.post("/api/v1/agent/chat", json={
        "message": "Esto es indignante, llevo tres días sin internet y nadie resuelve nada"
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["escalate"] is True


def test_delete_session(client):
    created = client.post("/api/v1/agent/chat", json={"message": "Hola"}).json()
    session_id = created["session_id"]

    resp = client.delete(f"/api/v1/agent/sessions/{session_id}")
    assert resp.status_code == 204

    resp_get = client.get(f"/api/v1/agent/sessions/{session_id}")
    assert resp_get.status_code == 404
