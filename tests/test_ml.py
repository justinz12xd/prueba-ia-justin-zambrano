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
