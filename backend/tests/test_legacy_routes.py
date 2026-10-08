from fastapi.testclient import TestClient


def test_legacy_mutation_and_replay_routes_cannot_bypass_dataset_workbench(tmp_path, monkeypatch):
    from backend.app import main
    from backend.app.services.universal_log import UniversalLog

    class LegacyStore:
        def __getattr__(self, name):
            raise AssertionError(f"Legacy route accessed {name}.")

    monkeypatch.setattr(main, "store", LegacyStore())
    monkeypatch.setattr(main, "universal_log", UniversalLog(tmp_path / "universal_log.jsonl"))

    with TestClient(main.app) as client:
        responses = [
            client.get("/api/events"),
            client.get("/api/events/SEC-1"),
            client.post("/api/events/generate", json={"size": 1}),
            client.post(
                "/api/corrections",
                json={
                    "event_id": "SEC-1",
                    "feature": "failed_logins",
                    "old_value": 1,
                    "new_value": 2,
                },
            ),
            client.post("/api/rewind/analyze", json={"event_id": "SEC-1"}),
            client.post("/api/rewind/execute", json={"event_id": "SEC-1"}),
            client.get("/api/decisions/SEC-1"),
            client.post("/api/ai/explain", json={"event_id": "SEC-1"}),
            client.post("/api/experiments/run", json={}),
        ]

    assert [response.status_code for response in responses] == [410] * len(responses)


def test_metrics_report_absent_training_and_verification_as_unavailable(tmp_path, monkeypatch):
    from backend.app import main
    from backend.app.services.universal_log import UniversalLog

    class EmptyDatasetStore:
        def fail_interrupted_training_jobs(self):
            return None

        def get_active_model(self):
            return None

        def get_active_dataset(self):
            return None

    monkeypatch.setattr(main, "dataset_store", EmptyDatasetStore())
    monkeypatch.setattr(main, "universal_log", UniversalLog(tmp_path / "universal_log.jsonl"))

    with TestClient(main.app) as client:
        response = client.get("/api/metrics")

    assert response.status_code == 200
    assert response.json() == {
        "datasets": {"training": None, "active_experiment": None},
        "decision_count": 5,
        "training_model_version": None,
        "training_validation_metrics": None,
        "verification_status": "NOT_AVAILABLE",
    }
