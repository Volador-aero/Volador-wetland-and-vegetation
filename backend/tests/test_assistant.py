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

    assert data["status"] == "INITIALIZED"
    assert data["feature"] == "AETHRA Assistant"
    assert data["message"] == "Assistant backend is reachable."
