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
