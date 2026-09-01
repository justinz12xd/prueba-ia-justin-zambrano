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


def test_sentiment_does_not_flag_neutral_questions(client, auth_headers):
    """Una pregunta informativa no debe marcarse como frustración.

    Regresión: con el dataset anterior (124 textos únicos, vocabulario de 155
    palabras) el modelo había aprendido que la palabra "una" implicaba enfado
    —aparecía solo en frases negativas— y clasificaba "Una última cosa, ¿cuál es
    el horario?" como negative con confianza 1.0, disparando un escalamiento.
    """
    for texto in ("Una última cosa, ¿cuál es el horario de atención de las oficinas?",
                  "Quiero saber cuánto tarda el cambio de plan",
                  "Necesito la factura del mes pasado en pdf"):
        resp = client.post("/api/v1/ml/analyze-sentiment", json={"text": texto},
                            headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["is_frustrated"] is False, f"falso positivo con: {texto}"


def test_sentiment_still_detects_real_frustration(client, auth_headers):
    """El arreglo anterior no debe costar sensibilidad ante enfado real."""
    for texto in ("Esto es inaceptable, llevo tres días sin internet y nadie resuelve nada",
                  "Estoy harto de este servicio, es pésimo"):
        body = client.post("/api/v1/ml/analyze-sentiment", json={"text": texto},
                            headers=auth_headers).json()
        assert body["sentiment"] == "negative"
        assert body["is_frustrated"] is True


def test_calm_fault_report_is_not_frustration(client, auth_headers):
    """Reportar una avería con calma no es estar enfadado: no debe escalar."""
    body = client.post("/api/v1/ml/analyze-sentiment", json={
        "text": "El internet se corta cada media hora desde el lunes, ya reinicié el router"
    }, headers=auth_headers).json()
    assert body["is_frustrated"] is False
