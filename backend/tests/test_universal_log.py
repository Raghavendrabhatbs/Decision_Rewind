import json
import logging

import pytest
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


def test_ai_log_context_searches_full_history_and_includes_aggregate_counts(tmp_path):
    log = UniversalLog(tmp_path / "universal_log.jsonl")
    for index in range(40):
        log.record(
            "application.log" if index == 2 else "test.event",
            "backend" if index == 2 else "test",
            {"level": "ERROR", "message": "database unavailable"} if index == 2 else {"sequence": index},
        )

    context = log.context_for_ai("What database errors happened?")

    assert context["total_events"] == 40
    assert context["matching_events"] == 1
    assert context["event_type_counts"] == {"test.event": 39, "application.log": 1}
    assert context["source_counts"] == {"test": 39, "backend": 1}
    assert any(
        event["event_type"] == "application.log"
        and event["details"]["message"] == "database unavailable"
        for event in context["events"]
    )


def test_ai_log_context_bounds_details_but_reports_omitted_history(tmp_path):
    log = UniversalLog(tmp_path / "universal_log.jsonl")
    for index in range(60):
        log.record("application.log", "backend", {"message": f"database unavailable {index} " + "x" * 900})

    context = log.context_for_ai("database errors", event_limit=60)

    assert context["total_events"] == 60
    assert context["matching_events"] == 60
    assert context["included_events"] < context["total_events"]
    assert context["omitted_events"] == context["total_events"] - context["included_events"]
    assert sum(
        len(json.dumps(event, ensure_ascii=True, separators=(",", ":")))
        for event in context["events"]
    ) <= 24_000


def test_ai_log_context_filters_by_references_and_preserves_chronological_order(tmp_path):
    log = UniversalLog(tmp_path / "universal_log.jsonl")
    log.record("workflow.correction", "dataset_workbench", {"event_id": "EVT-1", "correlation_id": "COR-1"})
    log.record("workflow.correction", "dataset_workbench", {"event_id": "EVT-2", "correlation_id": "COR-2"})
    log.record("workflow.rewind", "dataset_workbench", {"event_id": "EVT-1", "correlation_id": "COR-1"})

    context = log.context_for_ai("Summarize this event", references={"event_id": "EVT-1", "correlation_id": "COR-1"})

    assert context["event_count"] == 3
    assert context["returned_event_count"] == 2
    assert context["truncated"] is True
    assert [event["event_type"] for event in context["events"]] == [
        "workflow.correction",
        "workflow.rewind",
    ]
    assert all(event["details"]["event_id"] == "EVT-1" for event in context["events"])


def test_ai_log_context_supports_operation_source_level_and_timestamp_filters(tmp_path):
    log = UniversalLog(tmp_path / "universal_log.jsonl")
    log.record(
        "workflow.rewind",
        "backend",
        {"event_id": "EVT-1", "operation": "rewind", "level": "INFO"},
    )
    log.record(
        "workflow.rewind",
        "backend",
        {"event_id": "EVT-1", "operation": "rewind", "level": "ERROR"},
    )
    log.record(
        "workflow.rewind",
        "frontend",
        {"event_id": "EVT-1", "operation": "rewind", "level": "ERROR"},
    )

    context = log.context_for_ai(
        "",
        filters={
            "event_id": "EVT-1",
            "operation": "workflow.rewind",
            "source": "backend",
            "level": "ERROR",
            "timestamp_from": log.recent(3)[1]["timestamp"],
        },
    )

    assert context["returned_event_count"] == 1
    assert context["events"][0]["details"]["level"] == "ERROR"
    assert context["events"][0]["source"] == "backend"


def test_api_and_llm_events_are_logged_and_recent_history_reaches_ai(tmp_path, monkeypatch):
    from backend.app import main

    log = UniversalLog(tmp_path / "universal_log.jsonl")
    monkeypatch.setattr(main, "universal_log", log)

    def explain(question, evidence):
        assert any(
            record["event_type"] == "api.request.completed"
            and record["details"]["path"] == "/api/health"
            for record in evidence["universal_log_context"]["events"]
        )
        assert evidence["universal_log_context"]["total_events"] > 0
        return {"source": "groq:test-model", "answer": "Answer from context.", "evidence": evidence}

    monkeypatch.setattr(main.llm, "explain", explain)
    with TestClient(main.app) as client:
        assert client.get("/api/health").status_code == 200
        response = client.post("/api/ai/chat", json={"question": "What happened to /api/health?"})

    assert response.status_code == 200
    records = log.recent(10)
    assert any(record["event_type"] == "llm.model_output" for record in records)
    chat_request_log = next(
        record for record in records
        if record["event_type"] == "api.request.completed"
        and record["details"]["path"] == "/api/ai/chat"
    )
    assert "evidence" not in chat_request_log["details"]["response"]


def test_chat_passes_matching_historical_log_events_for_user_question(tmp_path, monkeypatch):
    from backend.app import main

    log = UniversalLog(tmp_path / "universal_log.jsonl")
    log.record("application.log", "backend", {"level": "ERROR", "message": "database unavailable"})
    for index in range(25):
        log.record("workflow.event", "test", {"sequence": index})
    monkeypatch.setattr(main, "universal_log", log)

    def explain(question, evidence):
        assert question == "What happened to the database?"
        context = evidence["universal_log_context"]
        assert context["total_events"] >= 26
        assert any("database unavailable" in str(event) for event in context["events"])
        return {"source": "groq:test-model", "answer": "The database was unavailable.", "evidence": evidence}

    monkeypatch.setattr(main.llm, "explain", explain)
    with TestClient(main.app) as client:
        response = client.post("/api/ai/chat", json={"question": "What happened to the database?"})

    assert response.status_code == 200
    assert response.json()["answer"] == "The database was unavailable."


def test_ai_explanations_receive_recent_universal_log(tmp_path, monkeypatch):
    from backend.app import main

    log = UniversalLog(tmp_path / "universal_log.jsonl")
    log.record("workflow.correction.applied", "test", {"event_id": "SEC-1"})
    monkeypatch.setattr(main, "universal_log", log)
    monkeypatch.setattr(main.llm, "explain", lambda *_: pytest.fail("Legacy LLM route must not be used."))
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

    assert response.status_code == 410


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
