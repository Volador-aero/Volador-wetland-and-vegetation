"""API-level regression tests for AETHRA Assistant chat behaviour."""

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from app.routers import assistant as assistant_router


app = FastAPI()
app.include_router(assistant_router.router)
client = TestClient(app)


def test_chat_without_project_returns_generated_answer(monkeypatch):
    calls = {}

    def fake_generate_assistant_response(question, context=None):
        calls["question"] = question
        calls["context"] = context
        return "NDVI measures vegetation greenness."

    monkeypatch.setattr(
        assistant_router, "generate_assistant_response", fake_generate_assistant_response
    )

    response = client.post(
        "/api/assistant/chat", json={"question": "  Explain NDVI  "}
    )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "NDVI measures vegetation greenness.",
        "model": assistant_router.OPENROUTER_MODEL,
        "status": "SUCCESS",
    }
    assert calls == {"question": "Explain NDVI", "context": None}


def test_chat_includes_selected_project_and_run_context(monkeypatch):
    expected_context = {
        "project_available": True,
        "project": {"name": "Wetland test"},
        "analysis": {"available": False},
    }
    context_calls = {}
    generation_calls = {}

    def fake_build_context(project_id, run_id=None):
        context_calls.update(project_id=project_id, run_id=run_id)
        return expected_context

    def fake_generate_assistant_response(question, context=None):
        generation_calls.update(question=question, context=context)
        return "The selected project has no completed run yet."

    monkeypatch.setattr(assistant_router, "build_assistant_context", fake_build_context)
    monkeypatch.setattr(
        assistant_router, "generate_assistant_response", fake_generate_assistant_response
    )

    response = client.post(
        "/api/assistant/chat",
        json={
            "question": "Summarise this project",
            "project_id": "project_123",
            "run_id": "run_456",
        },
    )

    assert response.status_code == 200
    assert context_calls == {"project_id": "project_123", "run_id": "run_456"}
    assert generation_calls == {
        "question": "Summarise this project",
        "context": expected_context,
    }


def test_chat_handles_invalid_project_id_as_unavailable_context(monkeypatch):
    context_calls = []

    def fake_build_context(project_id, run_id=None):
        context_calls.append((project_id, run_id))
        return {"project_available": False, "message": "No valid project was selected."}

    captured = {}

    def fake_generate_assistant_response(question, context=None):
        captured["context"] = context
        return "I cannot access a valid selected project."

    monkeypatch.setattr(assistant_router, "build_assistant_context", fake_build_context)
    monkeypatch.setattr(
        assistant_router, "generate_assistant_response", fake_generate_assistant_response
    )

    response = client.post(
        "/api/assistant/chat",
        json={"question": "What is in this project?", "project_id": "../outside"},
    )

    assert response.status_code == 200
    assert context_calls == [("../outside", None)]
    assert captured["context"]["project_available"] is False


def test_chat_passes_missing_dataset_and_preparation_state_to_model(monkeypatch):
    missing_data_context = {
        "project_available": True,
        "datasets": {
            "t1": {"folder_exists": False, "file_count": 0},
            "t2": {"folder_exists": False, "file_count": 0},
        },
        "preparation": {"available": False},
        "analysis": {"available": False},
    }
    monkeypatch.setattr(
        assistant_router, "build_assistant_context", lambda project_id, run_id=None: missing_data_context
    )
    captured = {}

    def fake_generate_assistant_response(question, context=None):
        captured["context"] = context
        return "The project has no uploaded datasets or preparation record."

    monkeypatch.setattr(
        assistant_router, "generate_assistant_response", fake_generate_assistant_response
    )

    response = client.post(
        "/api/assistant/chat",
        json={"question": "Are my datasets ready?", "project_id": "project_123"},
    )

    assert response.status_code == 200
    assert captured["context"] == missing_data_context


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"question": ""},
        {"question": "   "},
        {"question": "x" * 2001},
        {"question": "Valid question", "project_id": "   "},
        {"question": "Valid question", "run_id": "   "},
        {"question": "Valid question", "project_id": "p" * 129},
        {"question": "Valid question", "run_id": "r" * 129},
    ],
)
def test_chat_rejects_invalid_request_payloads(payload):
    response = client.post("/api/assistant/chat", json=payload)
    assert response.status_code == 422


def test_chat_maps_missing_configuration_to_503(monkeypatch):
    def fail_generation(question, context=None):
        raise RuntimeError("private configuration detail")

    monkeypatch.setattr(assistant_router, "generate_assistant_response", fail_generation)

    response = client.post("/api/assistant/chat", json={"question": "Hello"})

    assert response.status_code == 503
    assert response.json()["detail"] == "The AI Assistant is not currently configured or available."
    assert "private configuration detail" not in response.text


def test_chat_maps_provider_api_failure_to_502(monkeypatch):
    class FakeAPIError(Exception):
        pass

    def fail_generation(question, context=None):
        raise FakeAPIError("provider internal detail")

    monkeypatch.setattr(assistant_router, "APIError", FakeAPIError)
    monkeypatch.setattr(assistant_router, "generate_assistant_response", fail_generation)

    response = client.post("/api/assistant/chat", json={"question": "Hello"})

    assert response.status_code == 502
    assert response.json()["detail"] == "The AI service could not complete your request."
    assert "provider internal detail" not in response.text


def test_chat_maps_unexpected_failure_to_500(monkeypatch):
    def fail_generation(question, context=None):
        raise ValueError("sensitive internal detail")

    monkeypatch.setattr(assistant_router, "generate_assistant_response", fail_generation)

    response = client.post("/api/assistant/chat", json={"question": "Hello"})

    assert response.status_code == 500
    assert response.json()["detail"] == "An unexpected error occurred in the AI Assistant."
    assert "sensitive internal detail" not in response.text
