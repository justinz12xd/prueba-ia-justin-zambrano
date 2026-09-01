def test_create_customer_success(client, auth_headers):
    payload = {
        "name": "Ana Torres", "email": "ana.torres@example.com", "phone": "0991112233",
        "plan_type": "Fibra 500MB", "monthly_charge": 55.0, "tenure_months": 3,
        "total_charges": 165.0, "contract_type": "month-to-month", "payment_method": "credit_card",
    }
    resp = client.post("/api/v1/customers", json=payload, headers=auth_headers)
    assert resp.status_code == 201
    body = resp.json()
    assert body["email"] == payload["email"]
    assert body["is_active"] is True


def test_create_customer_invalid_phone(client, auth_headers):
    payload = {
        "name": "Carlos Ruiz", "email": "carlos.ruiz@example.com", "phone": "12345",
        "plan_type": "Fibra 200MB", "monthly_charge": 30.0,
    }
    resp = client.post("/api/v1/customers", json=payload, headers=auth_headers)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "VALIDATION_ERROR"


def test_create_customer_invalid_email(client, auth_headers):
    payload = {
        "name": "Pedro Gómez", "email": "no-es-un-email", "phone": "0991234567",
        "plan_type": "Fibra 200MB", "monthly_charge": 30.0,
    }
    resp = client.post("/api/v1/customers", json=payload, headers=auth_headers)
    assert resp.status_code == 422


def test_get_customer_not_found(client, auth_headers):
    resp = client.get("/api/v1/customers/999999", headers=auth_headers)
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "NOT_FOUND"


def test_soft_delete_customer(client, auth_headers):
    payload = {
        "name": "Cliente Borrar", "email": "borrar@example.com", "phone": "0991230000",
        "plan_type": "Fibra 100MB", "monthly_charge": 25.0,
    }
    created = client.post("/api/v1/customers", json=payload, headers=auth_headers).json()
    customer_id = created["customer_id"]

    resp = client.delete(f"/api/v1/customers/{customer_id}", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json()["is_active"] is False

    # Eliminación lógica: ya no debe ser accesible por el endpoint de detalle
    resp_get = client.get(f"/api/v1/customers/{customer_id}", headers=auth_headers)
    assert resp_get.status_code == 404


def test_churn_prediction_endpoint(client, auth_headers):
    payload = {
        "name": "Cliente Churn", "email": "churn.test@example.com", "phone": "0991239999",
        "plan_type": "Fibra 200MB", "monthly_charge": 60.0, "tenure_months": 2,
        "total_charges": 120.0, "contract_type": "month-to-month", "payment_method": "cash",
    }
    created = client.post("/api/v1/customers", json=payload, headers=auth_headers).json()
    customer_id = created["customer_id"]

    resp = client.get(f"/api/v1/customers/{customer_id}/churn-prediction", headers=auth_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["risk_level"] in {"low", "medium", "high"}


def test_ticket_stats_counts_open_and_averages_satisfaction(client, auth_headers):
    """num_tickets cuenta solo los tickets SIN RESOLVER (como pide el enunciado) y
    avg_satisfaction sale de los tickets calificados, no de un valor fijo."""
    from app.core.database import SessionLocal
    from app.services.customer_service import CustomerService

    # Cliente nuevo y aislado, para no depender del estado que dejan otros tests.
    customer = client.post("/api/v1/customers", json={
        "name": "Stats Test", "email": "stats.test@example.com", "phone": "0995550001",
        "plan_type": "Fibra 100MB", "monthly_charge": 30.0,
    }, headers=auth_headers).json()
    cid = customer["customer_id"]

    db = SessionLocal()
    try:
        assert CustomerService(db).ticket_stats(cid) == (0, 3.5)  # sin tickets -> por defecto

        resuelto = client.post("/api/v1/tickets", json={
            "customer_id": cid, "priority": "low",
            "description": "Consulta general sobre el estado de mi servicio contratado",
        }, headers=auth_headers).json()
        client.put(f"/api/v1/tickets/{resuelto['ticket_id']}",
                   json={"status": "resolved", "satisfaction": 1}, headers=auth_headers)

        client.post("/api/v1/tickets", json={
            "customer_id": cid, "priority": "high",
            "description": "No tengo internet desde ayer y el modem parpadea en rojo",
        }, headers=auth_headers)

        db.expire_all()
        num_tickets, avg = CustomerService(db).ticket_stats(cid)
        assert num_tickets == 1, "el ticket resuelto no debe contar como abierto"
        assert avg == 1.0, "la satisfacción promedio debe salir del ticket calificado"
    finally:
        db.close()


def test_customer_cannot_read_another_customers_data(client, customer_headers, auth_headers):
    """Un cliente autenticado no puede leer la ficha, los tickets ni las
    conversaciones de otro cliente cambiando el id de la URL (IDOR).

    Se responde 404 y no 403 a propósito: un 403 confirmaría que ese recurso existe.
    """
    ajeno = client.post("/api/v1/customers", json={
        "name": "Cliente Ajeno", "email": "ajeno.idor@example.com", "phone": "0998887771",
        "plan_type": "Fibra 500MB", "monthly_charge": 80.0,
    }, headers=auth_headers).json()
    ajeno_id = ajeno["customer_id"]

    ticket = client.post("/api/v1/tickets", json={
        "customer_id": ajeno_id, "priority": "low",
        "description": "Dato sensible del cliente ajeno que no debe filtrarse jamas",
    }, headers=auth_headers).json()
    sesion = client.post("/api/v1/agent/chat", json={
        "message": "Quiero cancelar mi servicio, mi contrato termina el mes que viene",
        "customer_id": ajeno_id,
    }, headers=auth_headers).json()

    # El rol customer del seed es el cliente 1, no el ajeno
    assert client.get(f"/api/v1/customers/{ajeno_id}", headers=customer_headers).status_code == 404
    assert client.get(f"/api/v1/tickets/{ticket['ticket_id']}",
                      headers=customer_headers).status_code == 404
    assert client.get(f"/api/v1/agent/sessions/{sesion['session_id']}",
                      headers=customer_headers).status_code == 404

    # Y sigue viendo lo suyo
    assert client.get("/api/v1/customers/1", headers=customer_headers).status_code == 200


def test_customer_listing_tickets_is_scoped_to_itself(client, customer_headers, auth_headers):
    """Pedir los tickets de otro cliente devuelve los propios, no los ajenos."""
    ajeno = client.post("/api/v1/customers", json={
        "name": "Otro Mas", "email": "otro.scope@example.com", "phone": "0998887772",
        "plan_type": "Fibra 100MB", "monthly_charge": 40.0,
    }, headers=auth_headers).json()
    client.post("/api/v1/tickets", json={
        "customer_id": ajeno["customer_id"], "priority": "low",
        "description": "Ticket del cliente ajeno que no debe aparecer en listados ajenos",
    }, headers=auth_headers)

    visibles = client.get(f"/api/v1/tickets?customer_id={ajeno['customer_id']}",
                          headers=customer_headers).json()
    assert all(t["customer_id"] == 1 for t in visibles), "solo debe ver los suyos"


def test_customer_cannot_open_tickets_for_others(client, customer_headers, auth_headers):
    ajeno = client.post("/api/v1/customers", json={
        "name": "Tercero", "email": "tercero.scope@example.com", "phone": "0998887773",
        "plan_type": "Fibra 100MB", "monthly_charge": 40.0,
    }, headers=auth_headers).json()
    resp = client.post("/api/v1/tickets", json={
        "customer_id": ajeno["customer_id"], "priority": "low",
        "description": "Intento de abrir un ticket a nombre de otra persona",
    }, headers=customer_headers)
    assert resp.status_code == 404


def test_chat_ignores_a_foreign_customer_id(client, customer_headers, auth_headers):
    """Si un cliente envía el customer_id de otro, se usa el suyo."""
    ajeno = client.post("/api/v1/customers", json={
        "name": "Cuarto", "email": "cuarto.scope@example.com", "phone": "0998887774",
        "plan_type": "Fibra 100MB", "monthly_charge": 40.0,
    }, headers=auth_headers).json()
    chat = client.post("/api/v1/agent/chat", json={
        "message": "Necesito revisar el estado de mi servicio de internet contratado",
        "customer_id": ajeno["customer_id"],
    }, headers=customer_headers).json()
    sesion = client.get(f"/api/v1/agent/sessions/{chat['session_id']}",
                        headers=customer_headers).json()
    assert sesion["customer_id"] == 1
