"""Coach contract and provider isolation tests; no external model calls."""
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from src.api import coach as module


@pytest.fixture
def client(monkeypatch):
    for key in ("ASTRA_GPT_ENDPOINT", "ASTRA_GPT_MODEL", "ASTRA_GPT_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    module.app.dependency_overrides.clear()
    with TestClient(module.app) as value:
        yield value
    module.app.dependency_overrides.clear()


@pytest.fixture
def payload():
    player = {"total": 30, "behindBaseline": 80, "backcourt": 90, "frontcourt": 10, "center": 40, "left": 65}
    return {"question": "Compare recovery", "evidence": {"runId": "test", "fps": 30, "range": {"startFrame": 0, "endFrame": 29, "durationSeconds": 1}, "geometryValid": True, "players": [{"id": 1, **player}, {"id": 2, **player}], "events": [], "limitations": ["Single camera"]}}


def configure_provider():
    module.app.dependency_overrides[module.provider_settings] = lambda: module.ProviderSettings("https://provider.example/v1/chat/completions", "configured-model", "server-only-secret")


def mock_provider(monkeypatch, handler):
    original = httpx.AsyncClient
    monkeypatch.setattr(module.httpx, "AsyncClient", lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs))


def test_unconfigured_is_explicit(client, payload):
    assert client.get("/api/coach/status").json() == {"configured": False, "model": None}
    assert client.post("/api/coach", json=payload).status_code == 503


def test_invalid_ranges_and_duplicate_players(client, payload):
    configure_provider()
    payload["evidence"]["range"]["startFrame"] = 50
    assert client.post("/api/coach", json=payload).status_code == 422
    payload["evidence"]["range"]["startFrame"] = 0
    payload["evidence"]["players"][1]["id"] = 1
    assert client.post("/api/coach", json=payload).status_code == 422


def test_provider_request_preserves_evidence_and_keeps_key_on_server(client, payload, monkeypatch):
    configure_provider()
    def handler(request):
        body = json.loads(request.content)
        assert request.headers["authorization"] == "Bearer server-only-secret"
        assert body["model"] == "configured-model"
        assert "untrusted data" in body["messages"][0]["content"]
        assert '"behindBaseline":80.0' in body["messages"][1]["content"]
        return httpx.Response(200, json={"choices": [{"message": {"content": "Review recovery depth at contact."}}]})
    mock_provider(monkeypatch, handler)
    response = client.post("/api/coach", json=payload)
    assert response.status_code == 200
    assert response.json() == {"text": "Review recovery depth at contact.", "model": "configured-model"}
    assert "server-only-secret" not in response.text


@pytest.mark.parametrize("status,body,expected", [(401, {"error": "server-only-secret"}, 502), (429, {}, 429), (200, {"choices": []}, 502), (302, {}, 502)])
def test_provider_errors_are_sanitized(client, payload, monkeypatch, status, body, expected):
    configure_provider()
    mock_provider(monkeypatch, lambda request: httpx.Response(status, json=body))
    response = client.post("/api/coach", json=payload)
    assert response.status_code == expected
    assert "server-only-secret" not in response.text


def test_remote_cleartext_endpoint_is_rejected(client, payload, monkeypatch):
    monkeypatch.setenv("ASTRA_GPT_ENDPOINT", "http://provider.example/chat")
    monkeypatch.setenv("ASTRA_GPT_MODEL", "configured-model")
    monkeypatch.setenv("ASTRA_GPT_API_KEY", "server-only-secret")
    assert client.post("/api/coach", json=payload).status_code == 503


def test_unknown_fields_and_oversized_question_are_rejected(client, payload):
    configure_provider()
    payload["question"] = "x" * 1001
    assert client.post("/api/coach", json=payload).status_code == 422
    payload["question"] = "Compare players"
    payload["evidence"]["sourceVideoPath"] = "private/path.mp4"
    assert client.post("/api/coach", json=payload).status_code == 422
