from pathlib import Path
import sys

from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.main import app

client = TestClient(app)

def test_assistant_status():
    response = client.get("/api/assistant/status")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] in ["READY", "NOT_CONFIGURED"]
    assert isinstance(data["configured"], bool)
    assert "model" in data
    assert "message" in data