import json
from io import BytesIO
from unittest.mock import patch
from urllib.error import HTTPError

import pytest

from backend.app.llm.provider import LLMProvider


def test_groq_provider_answers_general_question_without_event_context():
    response_body = json.dumps({
        "choices": [{"message": {"content": "I am using the Groq API."}}],
    }).encode()
    with patch("backend.app.llm.provider.urlopen") as urlopen:
        urlopen.return_value.__enter__.return_value.read.return_value = response_body

        response = LLMProvider(
            provider="groq",
            api_key="test-key",
            model="llama-test",
        ).explain("Which API are you using?", {})

    request = urlopen.call_args.args[0]
    payload = json.loads(request.data)
    assert request.full_url == "https://api.groq.com/openai/v1/chat/completions"
    headers = {name.lower(): value for name, value in request.header_items()}
    assert headers["authorization"] == "Bearer test-key"
    assert payload["model"] == "llama-test"
    assert "Which API are you using?" in payload["messages"][1]["content"]
    assert response == {
        "source": "groq:llama-test",
        "answer": "I am using the Groq API.",
        "evidence": {},
    }


def test_groq_provider_reports_missing_api_key():
    with pytest.raises(RuntimeError, match="Set GROQ_API_KEY"):
        LLMProvider(api_key="").explain("General question", {})


def test_groq_provider_explains_cloudflare_access_denial_without_exposing_key():
    denied = HTTPError(
        "https://api.groq.com/openai/v1/chat/completions",
        403,
        "Forbidden",
        {},
        BytesIO(b"error code: 1010"),
    )
    with patch("backend.app.llm.provider.urlopen", side_effect=denied):
        with pytest.raises(RuntimeError, match="Cloudflare error 1010") as error:
            LLMProvider(api_key="private-key").explain("General question", {})
    assert "private-key" not in str(error.value)


def test_chat_endpoint_accepts_general_questions_without_an_active_experiment():
    from fastapi.testclient import TestClient

    from backend.app import main

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


def test_ai_status_does_not_expose_api_key():
    from fastapi.testclient import TestClient

    from backend.app import main

    with patch.object(main, "llm", LLMProvider(provider="groq", api_key="private-key", model="llama-test")):
        with TestClient(main.app) as client:
            response = client.get("/api/ai/status")

    assert response.status_code == 200
    assert response.json() == {"provider": "groq", "model": "llama-test", "configured": True}
    assert "private-key" not in response.text
