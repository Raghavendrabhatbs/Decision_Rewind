from unittest.mock import patch

import pytest

from backend.app.llm.provider import LLMProvider


def test_groq_provider_uses_responses_api_for_general_question():
    with patch("backend.app.llm.provider.OpenAI") as openai_client:
        openai_client.return_value.responses.create.return_value.output_text = "I am using the Groq API."
        response = LLMProvider(
            provider="groq",
            api_key="test-key",
            base_url="https://api.groq.com/openai/v1",
            model="openai/gpt-oss-20b",
        ).explain("Which API are you using?", {})

    openai_client.assert_called_once_with(
        api_key="test-key",
        base_url="https://api.groq.com/openai/v1",
        timeout=45,
    )
    request = openai_client.return_value.responses.create.call_args.kwargs
    assert request["model"] == "openai/gpt-oss-20b"
    assert "Which API are you using?" in request["input"]
    assert "using only the supplied" in request["instructions"]
    assert "If sources conflict" in request["instructions"]
    assert "Never authorize" in request["instructions"]
    assert "hidden chain-of-thought" in request["instructions"]
    assert response == {
        "source": "groq:openai/gpt-oss-20b",
        "answer": "I am using the Groq API.",
        "evidence": {},
    }


def test_groq_provider_reports_missing_api_key():
    with pytest.raises(RuntimeError, match="Set GROQ_API_KEY"):
        LLMProvider(api_key="").explain("General question", {})


def test_groq_provider_rejects_empty_responses():
    with patch("backend.app.llm.provider.OpenAI") as openai_client:
        openai_client.return_value.responses.create.return_value.output_text = " "
        with pytest.raises(RuntimeError, match="empty answer"):
            LLMProvider(api_key="private-key").explain("General question", {})


def test_chat_endpoint_accepts_general_questions_without_an_active_experiment(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from backend.app import main
    from backend.app.services.universal_log import UniversalLog

    monkeypatch.setattr(main, "universal_log", UniversalLog(tmp_path / "universal_log.jsonl"))
    with patch.object(
        main.llm,
        "explain",
        return_value={"source": "groq:test-model", "answer": "Answered.", "evidence": {}},
    ):
        with TestClient(main.app) as client:
            response = client.post("/api/ai/chat", json={"question": "Which API are you using?"})

    assert response.status_code == 200
    assert response.json()["answer"] == "Answered."
    assert response.json()["source"] == "groq:test-model"


def test_ai_status_does_not_expose_api_key(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from backend.app import main
    from backend.app.services.universal_log import UniversalLog

    monkeypatch.setattr(main, "universal_log", UniversalLog(tmp_path / "universal_log.jsonl"))
    with patch.object(main, "llm", LLMProvider(provider="groq", api_key="private-key", model="llama-test")):
        with TestClient(main.app) as client:
            response = client.get("/api/ai/status")

    assert response.status_code == 200
    assert response.json() == {"provider": "groq", "model": "llama-test", "configured": True}
    assert "private-key" not in response.text
