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
