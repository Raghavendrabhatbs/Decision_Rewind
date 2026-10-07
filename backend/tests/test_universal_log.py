import json
import logging

from fastapi.testclient import TestClient

from backend.app.services.universal_log import UniversalLog


def test_universal_log_appends_json_lines_redacts_secrets_and_reads_recent(tmp_path):
    log = UniversalLog(tmp_path / "universal_log.jsonl")

    for index in range(15):
        log.record(
            "test.event",
            "test",
            {
                "sequence": index,
                "api_key": "must-not-be-written",
                "question": "credential gsk_supersecret123",
            },
        )

    records = [json.loads(line) for line in log.path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 15
    assert records[0]["details"]["api_key"] == "[REDACTED]"
    assert records[0]["details"]["question"] == "credential [REDACTED]"
    assert [record["details"]["sequence"] for record in log.recent(3)] == [12, 13, 14]
    assert len(log.recent_for_ai(2)) == 2
    assert "must-not-be-written" not in log.path.read_text(encoding="utf-8")


def test_api_and_llm_events_are_logged_and_recent_history_reaches_ai(tmp_path, monkeypatch):
    from backend.app import main

    log = UniversalLog(tmp_path / "universal_log.jsonl")
    monkeypatch.setattr(main, "universal_log", log)

    def explain(question, evidence):
        assert any(
            record["event_type"] == "api.request.completed"
            and record["details"]["path"] == "/api/health"
            for record in evidence["recent_universal_log"]
        )
        return {"source": "groq:test-model", "answer": "Answer from context.", "evidence": evidence}

    monkeypatch.setattr(main.llm, "explain", explain)
    with TestClient(main.app) as client:
        assert client.get("/api/health").status_code == 200
        response = client.post("/api/ai/chat", json={"question": "What happened?"})

    assert response.status_code == 200
    records = log.recent(10)
    assert any(record["event_type"] == "llm.model_output" for record in records)
    assert any(
        record["event_type"] == "api.request.completed"
        and record["details"]["path"] == "/api/ai/chat"
        for record in records
    )


def test_ai_explanations_receive_recent_universal_log(tmp_path, monkeypatch):
    from backend.app import main

    log = UniversalLog(tmp_path / "universal_log.jsonl")
    log.record("workflow.correction.applied", "test", {"event_id": "SEC-1"})
    monkeypatch.setattr(main, "universal_log", log)

    def explain(question, evidence):
        assert any(
            record["event_type"] == "workflow.correction.applied"
            for record in evidence["recent_universal_log"]
        )
        return {"source": "groq:test-model", "answer": "Explained.", "evidence": evidence}

    monkeypatch.setattr(main.llm, "explain", explain)
    with TestClient(main.app) as client:
        response = client.post(
            "/api/ai/explain",
            json={
                "event_id": "SEC-1",
                "correction": {
                    "event_id": "SEC-1",
                    "feature": "failed_logins",
                    "old_value": 1,
                    "new_value": 2,
                },
                "analysis": {"decisions": [{"decision_id": "D1"}]},
            },
        )

    assert response.status_code == 200
    assert any(record["event_type"] == "llm.model_output" for record in log.recent(5))


def test_application_logging_handler_writes_python_log_records(tmp_path, monkeypatch):
    from backend.app.services import universal_log as universal_log_module

    log = UniversalLog(tmp_path / "universal_log.jsonl")
    monkeypatch.setattr(universal_log_module, "universal_log", log)
    logging.getLogger("backend.app.test").error(
        "Application failure: %s",
        "database unavailable",
    )

    captured = log.recent(1)[0]
    assert captured["event_type"] == "application.log"
    assert captured["source"] == "backend.app.test"
    assert captured["details"] == {"level": "ERROR", "message": "Application failure: database unavailable"}


def test_frontend_application_logs_are_ingested_and_redacted(tmp_path, monkeypatch):
    from backend.app import main

    log = UniversalLog(tmp_path / "universal_log.jsonl")
    monkeypatch.setattr(main, "universal_log", log)
    with TestClient(main.app) as client:
        response = client.post(
            "/api/universal-log/client",
            json={
                "event_type": "frontend.application_error",
                "source": "browser",
                "details": {"message": "Request used gsk_supersecret123"},
            },
        )

    assert response.status_code == 204
    captured = next(
        record for record in log.recent(10)
        if record["event_type"] == "frontend.application_error"
    )
    assert captured["source"] == "frontend.browser"
    assert captured["details"]["message"] == "Request used [REDACTED]"


def test_frontend_log_endpoint_rejects_oversized_and_invalid_events(tmp_path, monkeypatch):
    from backend.app import main

    monkeypatch.setattr(main, "universal_log", UniversalLog(tmp_path / "universal_log.jsonl"))
    with TestClient(main.app) as client:
        invalid = client.post(
            "/api/universal-log/client",
            json={"event_type": "api.request.completed", "source": "browser", "details": {}},
        )
        oversized = client.post(
            "/api/universal-log/client",
            json={
                "event_type": "frontend.console",
                "source": "browser",
                "details": {"message": "x" * 17_000},
            },
        )

    assert invalid.status_code == 422
    assert oversized.status_code == 413
