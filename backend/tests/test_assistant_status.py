from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers import assistant as assistant_router


app = FastAPI()
app.include_router(assistant_router.router)
client = TestClient(app)


def test_status_is_not_configured_when_api_key_is_missing(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    response = client.get("/api/assistant/status")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "NOT_CONFIGURED"
    assert data["configured"] is False
    assert "api key" in data["message"].lower()
    assert "OPENROUTER_API_KEY" not in str(data)


def test_status_is_ready_when_api_key_and_model_are_configured(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-a-real-secret")

    response = client.get("/api/assistant/status")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "READY"
    assert data["configured"] is True
    assert data["model"]
    assert "test-key-not-a-real-secret" not in response.text
