"""Unit tests for OpenRouter response handling without network requests."""

from types import SimpleNamespace

import pytest

from app import assistant_service


class FakeClient:
    def __init__(self, response):
        self.response = response
        self.closed = False
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self.create_completion)
        )

    def create_completion(self, **kwargs):
        self.request_kwargs = kwargs
        return self.response

    def close(self):
        self.closed = True


def _response(content="A valid answer", choices_marker=True):
    choices = []
    if choices_marker:
        choices = [
            SimpleNamespace(
                message=SimpleNamespace(content=content, refusal=None),
                finish_reason="stop",
            )
        ]
    return SimpleNamespace(
        choices=choices,
        model="test-model",
        id="test-response",
        usage=None,
    )


def test_generate_response_returns_trimmed_text_and_closes_client(monkeypatch):
    fake_client = FakeClient(_response("  Answer from the model.  "))
    monkeypatch.setattr(assistant_service, "OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(assistant_service, "OpenAI", lambda **kwargs: fake_client)

    answer = assistant_service.generate_assistant_response("Explain NDVI")

    assert answer == "Answer from the model."
    assert fake_client.closed is True
    assert fake_client.request_kwargs["model"] == assistant_service.OPENROUTER_MODEL


def test_generate_response_supports_list_based_text_content(monkeypatch):
    content = [
        SimpleNamespace(type="text", text="First paragraph."),
        SimpleNamespace(type="image", text=None),
        SimpleNamespace(type="text", text="Second paragraph."),
    ]
    fake_client = FakeClient(_response(content))
    monkeypatch.setattr(assistant_service, "OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(assistant_service, "OpenAI", lambda **kwargs: fake_client)

    answer = assistant_service.generate_assistant_response("Explain the result")

    assert answer == "First paragraph.\nSecond paragraph."
    assert fake_client.closed is True


def test_generate_response_rejects_missing_api_key(monkeypatch):
    monkeypatch.setattr(assistant_service, "OPENROUTER_API_KEY", None)

    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY is not configured"):
        assistant_service.generate_assistant_response("Hello")


def test_generate_response_rejects_empty_choices_and_closes_client(monkeypatch):
    fake_client = FakeClient(_response(choices_marker=False))
    monkeypatch.setattr(assistant_service, "OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(assistant_service, "OpenAI", lambda **kwargs: fake_client)

    with pytest.raises(RuntimeError, match="returned no choices"):
        assistant_service.generate_assistant_response("Hello")

    assert fake_client.closed is True


def test_generate_response_rejects_empty_content_and_closes_client(monkeypatch):
    fake_client = FakeClient(_response("   "))
    monkeypatch.setattr(assistant_service, "OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(assistant_service, "OpenAI", lambda **kwargs: fake_client)

    with pytest.raises(RuntimeError, match="empty response"):
        assistant_service.generate_assistant_response("Hello")

    assert fake_client.closed is True
