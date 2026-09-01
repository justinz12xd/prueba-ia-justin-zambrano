def test_mcp_capabilities(client):
    resp = client.get("/mcp/capabilities")
    assert resp.status_code == 200
    body = resp.json()
    tool_names = {t["name"] for t in body["tools"]}
    # Las 5 tools exigidas por el enunciado, más predict_resolution_time (Parte 2.2).
    assert tool_names == {"predict_churn", "classify_ticket", "get_customer_info",
                           "create_ticket", "chat_with_agent", "predict_resolution_time"}


def test_mcp_execute_predict_resolution_time(client):
    resp = client.post("/mcp/tools/execute", json={
        "id": "req-rt", "tool": "predict_resolution_time",
        "arguments": {"description": "El router pierde la señal cada media hora desde el lunes",
                      "category": "TECH", "priority": "high"},
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["result"]["isError"] is False
    assert body["result"]["content"][0]["data"]["estimated_hours"] > 0


def test_mcp_resources_list(client):
    resp = client.get("/mcp/resources")
    assert resp.status_code == 200
    assert any(r["id"] == "ticket_categories" for r in resp.json())


def test_mcp_get_resource(client):
    resp = client.get("/mcp/resources/ticket_categories")
    assert resp.status_code == 200
    body = resp.json()
    assert body["jsonrpc"] == "2.0"
    assert body["result"]["isError"] is False
    assert set(body["result"]["content"][0]["data"]["categories"]) == \
        {"TECH", "BILL", "PLAN", "CNCL", "OTHR"}


def test_mcp_execute_classify_ticket(client):
    resp = client.post("/mcp/tools/execute", json={
        "id": "req-1", "tool": "classify_ticket",
        "arguments": {"description": "Quiero cancelar mi servicio de internet"},
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == "req-1"
    assert body["result"]["isError"] is False
    assert body["result"]["content"][0]["data"]["predicted_category"] in \
        {"TECH", "BILL", "PLAN", "CNCL", "OTHR"}


def test_mcp_execute_unknown_tool(client):
    resp = client.post("/mcp/tools/execute", json={"id": "req-2", "tool": "does_not_exist",
                                                    "arguments": {}})
    assert resp.status_code == 200
    body = resp.json()
    assert body["result"]["isError"] is True


def test_mcp_execute_create_ticket_and_get_customer_info(client, auth_headers):
    customer_payload = {
        "name": "Cliente MCP", "email": "mcp.customer@example.com", "phone": "0991110099",
        "plan_type": "Fibra 200MB", "monthly_charge": 45.0,
    }
    customer = client.post("/api/v1/customers", json=customer_payload, headers=auth_headers).json()

    resp = client.post("/mcp/tools/execute", json={
        "id": "req-3", "tool": "create_ticket",
        "arguments": {"customer_id": customer["customer_id"],
                      "description": "El internet se desconecta cada 10 minutos, es muy molesto"},
    })
    assert resp.status_code == 200
    assert resp.json()["result"]["isError"] is False

    resp2 = client.post("/mcp/tools/execute", json={
        "id": "req-4", "tool": "get_customer_info",
        "arguments": {"customer_id": customer["customer_id"]},
    })
    assert resp2.status_code == 200
    assert resp2.json()["result"]["content"][0]["data"]["customer_id"] == customer["customer_id"]
