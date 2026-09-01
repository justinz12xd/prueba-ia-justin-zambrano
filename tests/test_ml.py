def test_predict_churn_endpoint(client, auth_headers):
    payload = {
        "tenure_months": 2, "monthly_charge": 80.0, "total_charges": 160.0,
        "contract_type": "month-to-month", "payment_method": "cash",
        "num_tickets": 5, "avg_satisfaction": 1.8,
    }
    resp = client.post("/api/v1/ml/predict-churn", json=payload, headers=auth_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["risk_level"] in {"low", "medium", "high"}


def test_analyze_sentiment_negative(client, auth_headers):
    resp = client.post("/api/v1/ml/analyze-sentiment",
                        json={"text": "Estoy muy molesto, llevo tres días sin solución"},
                        headers=auth_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["sentiment"] in {"positive", "neutral", "negative"}
    assert set(body["probabilities"]) == {"positive", "neutral", "negative"}


def test_models_info_public(client):
    resp = client.get("/api/v1/ml/models/info")
    assert resp.status_code == 200
    body = resp.json()
    for key in ("ticket_classifier", "churn_model", "sentiment_model", "resolution_time_model"):
        assert body[key]["loaded"] is True


def test_predict_churn_requires_role(client):
    resp = client.post("/api/v1/ml/predict-churn", json={"tenure_months": 1, "monthly_charge": 10,
                                                          "total_charges": 10})
    assert resp.status_code == 401


def test_predict_resolution_time_endpoint(client, auth_headers):
    payload = {
        "description": "El internet se corta cada media hora y el router parpadea en rojo",
        "category": "TECH", "priority": "high", "hour_of_day": 14, "day_of_week": 2,
    }
    resp = client.post("/api/v1/ml/predict-resolution-time", json=payload, headers=auth_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["estimated_hours"] > 0
    assert body["estimated_resolution_at"]
    assert body["category"] == "TECH"


def test_predict_resolution_time_defaults_to_now(client, auth_headers):
    """hour_of_day y day_of_week son opcionales: se toman del momento actual."""
    resp = client.post("/api/v1/ml/predict-resolution-time", json={
        "description": "Quiero entender por qué mi factura subió respecto al mes pasado",
        "category": "BILL",
    }, headers=auth_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert 0 <= body["hour_of_day"] <= 23
    assert 0 <= body["day_of_week"] <= 6


def test_predict_resolution_time_validates_input(client, auth_headers):
    resp = client.post("/api/v1/ml/predict-resolution-time", json={
        "description": "corto", "category": "TECH",
    }, headers=auth_headers)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "VALIDATION_ERROR"
