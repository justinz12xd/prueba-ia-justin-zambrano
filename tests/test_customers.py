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
