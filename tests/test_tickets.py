def test_classify_ticket_valid(client, auth_headers):
    resp = client.post("/api/v1/tickets/classify",
                        json={"description": "No tengo señal de internet desde ayer en la noche"},
                        headers=auth_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["predicted_category"] in {"TECH", "BILL", "PLAN", "CNCL", "OTHR"}
    assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-3


def test_classify_ticket_too_short(client, auth_headers):
    resp = client.post("/api/v1/tickets/classify", json={"description": "corto"},
                        headers=auth_headers)
    assert resp.status_code == 422


def test_create_ticket_with_auto_category(client, auth_headers):
    customer_payload = {
        "name": "Cliente Ticket", "email": "ticket.customer@example.com",
        "phone": "0991112200", "plan_type": "Fibra 200MB", "monthly_charge": 40.0,
    }
    customer = client.post("/api/v1/customers", json=customer_payload, headers=auth_headers).json()

    ticket_payload = {
        "customer_id": customer["customer_id"],
        "description": "Mi factura llegó con un cobro duplicado este mes, favor revisar",
        "priority": "high",
    }
    resp = client.post("/api/v1/tickets", json=ticket_payload, headers=auth_headers)
    assert resp.status_code == 201
    body = resp.json()
    assert body["category"] == "BILL"
    assert body["status"] == "open"


def test_create_ticket_description_too_short(client, auth_headers):
    customer_payload = {
        "name": "Cliente Ticket 2", "email": "ticket.customer2@example.com",
        "phone": "0991112201", "plan_type": "Fibra 200MB", "monthly_charge": 40.0,
    }
    customer = client.post("/api/v1/customers", json=customer_payload, headers=auth_headers).json()

    resp = client.post("/api/v1/tickets",
                        json={"customer_id": customer["customer_id"], "description": "muy corto"},
                        headers=auth_headers)
    assert resp.status_code == 422


def test_support_queue_enriches_tickets(client, auth_headers):
    """La bandeja del agente trae las señales de los modelos en una sola llamada."""
    client.post("/api/v1/tickets", json={
        "customer_id": 1, "priority": "high",
        "description": "No tengo señal de internet desde ayer y el modem parpadea en rojo",
    }, headers=auth_headers)

    resp = client.get("/api/v1/tickets/queue?limit=10", headers=auth_headers)
    assert resp.status_code == 200
    items = resp.json()
    assert items, "la bandeja no debería estar vacía"
    item = items[0]
    assert item["category"] in {"TECH", "BILL", "PLAN", "CNCL", "OTHR"}
    assert item["status"] in {"open", "in_progress"}
    assert item["sentiment"] in {"positive", "neutral", "negative"}
    assert 0 <= item["churn_probability"] <= 1
    assert item["estimated_hours"] > 0


def test_support_queue_is_internal_only(client, customer_headers):
    """La bandeja es una herramienta interna: sin token 401, y el cliente 403."""
    assert client.get("/api/v1/tickets/queue").status_code == 401
    assert client.get("/api/v1/tickets/queue", headers=customer_headers).status_code == 403


def test_queue_uses_last_message_not_the_description(client, auth_headers):
    """La bandeja debe reflejar el ánimo ACTUAL del cliente.

    Regresión: el sentimiento se calculaba sobre la descripción del ticket —que es el
    primer mensaje, normalmente el más calmado—, así que un cliente que se enfadaba
    más adelante en la conversación aparecía en la bandeja como si estuviera tranquilo.
    """
    calmado = ("Buenas tardes, desde el lunes el internet se corta cada media hora "
               "y ya reinicie el router")
    enfadado = "Esto es inaceptable, llevo tres dias sin internet y nadie resuelve nada"

    primero = client.post("/api/v1/agent/chat", json={"message": calmado, "customer_id": 1},
                          headers=auth_headers).json()
    ticket_id = primero["ticket"]["ticket_id"]
    assert primero["escalate"] is False

    # El mismo cliente se enfada: se reutiliza el ticket, pero cambia su ánimo
    segundo = client.post("/api/v1/agent/chat", json={"message": enfadado, "customer_id": 1},
                           headers=auth_headers).json()
    assert segundo["ticket"]["ticket_id"] == ticket_id
    assert segundo["escalate"] is True

    fila = next(t for t in client.get("/api/v1/tickets/queue", headers=auth_headers).json()
                if t["ticket_id"] == ticket_id)
    assert fila["sentiment_source"] == "last_message"
    assert fila["sentiment"] == "negative"
    assert fila["is_frustrated"] is True


def test_agent_can_open_and_reply_to_the_conversation(client, auth_headers):
    """Un asesor humano puede retomar el chat desde el ticket y responder al cliente."""
    # Cliente propio: el agente reutiliza tickets abiertos del mismo tipo, así que
    # compartir el cliente demo haría que este test dependiera del orden de ejecución.
    cliente = client.post("/api/v1/customers", json={
        "name": "Handoff Test", "email": "handoff.test@example.com", "phone": "0995550002",
        "plan_type": "Fibra 100MB", "monthly_charge": 30.0,
    }, headers=auth_headers).json()

    chat = client.post("/api/v1/agent/chat", json={
        "message": "El internet se corta cada media hora desde el lunes y ya reinicie el router",
        "customer_id": cliente["customer_id"],
    }, headers=auth_headers).json()
    ticket_id, session_id = chat["ticket"]["ticket_id"], chat["session_id"]

    hilo = client.get(f"/api/v1/tickets/{ticket_id}/conversation", headers=auth_headers).json()
    assert hilo["session_id"] == session_id, "el ticket debe recordar de qué chat nació"
    assert [m["role"] for m in hilo["messages"]] == ["user", "assistant"]

    respuesta = client.post(f"/api/v1/tickets/{ticket_id}/reply", json={
        "message": "Hola, soy Ana del equipo técnico. Ya estoy revisando su conexión."
    }, headers=auth_headers)
    assert respuesta.status_code == 200
    mensajes = respuesta.json()["messages"]
    assert mensajes[-1]["role"] == "agent_human"
    assert "Ana" in mensajes[-1]["content"]

    # El cliente ve la respuesta en su propia sesión, y el ticket queda tomado
    sesion = client.get(f"/api/v1/agent/sessions/{session_id}", headers=auth_headers).json()
    assert sesion["conversation"][-1]["role"] == "agent_human"
    assert client.get(f"/api/v1/tickets/{ticket_id}",
                      headers=auth_headers).json()["status"] == "in_progress"


def test_reply_rejected_when_ticket_has_no_conversation(client, auth_headers):
    """Un ticket creado por API no tiene chat que retomar: 409, no un 500."""
    ticket = client.post("/api/v1/tickets", json={
        "customer_id": 1, "priority": "low",
        "description": "Ticket creado directamente por la API sin pasar por el chat",
    }, headers=auth_headers).json()
    resp = client.post(f"/api/v1/tickets/{ticket['ticket_id']}/reply",
                        json={"message": "Buenas, le escribo del equipo de soporte"},
                        headers=auth_headers)
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "CONFLICT"


def test_conversation_is_internal_only(client, customer_headers):
    """El hilo con las notas internas no es accesible para el rol customer."""
    assert client.get("/api/v1/tickets/1/conversation",
                      headers=customer_headers).status_code == 403
    assert client.post("/api/v1/tickets/1/reply", json={"message": "hola de prueba"},
                       headers=customer_headers).status_code == 403


def test_queue_filters_open_tickets_in_the_query(client, auth_headers):
    """La bandeja no debe salir vacía por culpa del límite.

    Regresión: se pedían los N tickets más recientes y solo después se descartaban
    los resueltos, así que unos pocos tickets resueltos recientes podían ocupar todo
    el cupo y ocultar casos abiertos que sí requerían atención.
    """
    abierto = client.post("/api/v1/tickets", json={
        "customer_id": 1, "priority": "high",
        "description": "Caso abierto que debe seguir apareciendo en la bandeja del agente",
    }, headers=auth_headers).json()

    # Tres tickets resueltos, creados después: son los más recientes
    for i in range(3):
        t = client.post("/api/v1/tickets", json={
            "customer_id": 1, "priority": "low",
            "description": f"Ticket resuelto numero {i} que ya no requiere atencion alguna",
        }, headers=auth_headers).json()
        client.put(f"/api/v1/tickets/{t['ticket_id']}", json={"status": "resolved"},
                   headers=auth_headers)

    bandeja = client.get("/api/v1/tickets/queue?limit=3", headers=auth_headers).json()
    assert any(t["ticket_id"] == abierto["ticket_id"] for t in bandeja), \
        "el ticket abierto debe aparecer aunque haya resueltos más recientes"
    assert all(t["status"] in ("open", "in_progress") for t in bandeja)
