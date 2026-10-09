from fastapi.testclient import TestClient

from backend.app.services.llm_evidence import build_llm_evidence
from backend.app.services.universal_log import UniversalLog


class EvidenceStore:
    def fail_interrupted_training_jobs(self):
        return None

    dataset = {
        "dataset_id": "DS-1",
        "experiment_id": "EXP-1",
        "seed": 19,
        "record_count": 200,
        "training_status": "TRAINED",
        "workflow_status": "VERIFIED",
        "model_id": "SOC-MODEL-V1",
        "model_version": "V1",
        "training_dataset_id": "TRN-1",
        "training_record_count": 20_000,
        "trained_at": "2026-10-07T10:00:00Z",
    }
    event = {
        "original_state": {"event_id": "EVT-1", "failed_logins": 3},
        "current_state": {"event_id": "EVT-1", "failed_logins": 12},
        "decisions": [
            {
                "decision_id": "D2",
                "historical_output": "LOW",
                "current_output": "HIGH",
                "counterfactual_output": "HIGH",
            },
            {
                "decision_id": "D4",
                "historical_output": "MONITOR",
                "current_output": "ESCALATE",
                "counterfactual_output": "ESCALATE",
            },
        ],
        "corrections": [],
    }
    correction = {
        "correction_id": "COR-1",
        "dataset_id": "DS-1",
        "event_id": "EVT-1",
        "changes": [{"feature": "failed_logins", "old_value": 3, "new_value": 12}],
        "status": "REWOUND",
        "created_at": "2026-10-07T10:01:00Z",
    }
    audit = [
        {
            "event_id": "EVT-1",
            "operation": "VERIFIED",
            "timestamp": "2026-10-07T10:02:00Z",
            "details": {"verification": "VERIFIED", "verified_decisions": 2},
        },
        {
            "event_id": "EVT-1",
            "operation": "RECOVERED",
            "timestamp": "2026-10-07T10:01:59Z",
            "details": {"rewind_operations": ["RW-1", "RW-2"]},
        },
        {
            "event_id": "EVT-1",
            "operation": "COUNTERFACTUAL_REPLAYED",
            "timestamp": "2026-10-07T10:01:30Z",
            "details": {
                "decision_impacts": [
                    {"decision_id": "D2", "counterfactual_output": "HIGH", "changed": True},
                    {"decision_id": "D4", "counterfactual_output": "ESCALATE", "changed": True},
                ],
            },
        },
    ]

    def get_dataset(self, dataset_id):
        return self.dataset if dataset_id == self.dataset["dataset_id"] else None

    def get_dataset_by_experiment_id(self, experiment_id):
        return self.dataset if experiment_id == self.dataset["experiment_id"] else None

    def get_event(self, dataset_id, event_id):
        return self.event if dataset_id == "DS-1" and event_id == "EVT-1" else None

    def get_correction(self, correction_id):
        return self.correction if correction_id == "COR-1" else None

    def list_audit(self, dataset_id, limit):
        return self.audit

    def get_graph(self, dataset_id, features):
        return {
            "nodes": [{"id": "failed_logins", "kind": "feature"}, {"id": "D2", "kind": "decision"}],
            "edges": [{"source": "failed_logins", "target": "D2"}],
        }


def test_evidence_builder_uses_persisted_state_and_includes_recovery_context(tmp_path):
    log = UniversalLog(tmp_path / "universal_log.jsonl")
    log.record(
        "application.log",
        "backend",
        {"message": "D2 prediction LOW", "event_id": "EVT-1", "dataset_id": "DS-1"},
    )
    evidence = build_llm_evidence(
        question="Why was D2 changed?",
        dataset_store=EvidenceStore(),
        universal_log=log,
        experiment_id="EXP-1",
        event_id="EVT-1",
        correction_id="COR-1",
        decision_id="D2",
    )

    assert evidence["experiment"]["model_version"] == "V1"
    assert evidence["event"]["historical_state"]["failed_logins"] == 3
    assert evidence["event"]["current_state"]["failed_logins"] == 12
    assert evidence["decisions"]["historical"]["D2"] == "LOW"
    assert evidence["decisions"]["current"]["D2"] == "HIGH"
    assert evidence["decisions"]["counterfactual"]["D2"] == "HIGH"
    assert evidence["affected_decisions"] == ["D2", "D4"]
    assert evidence["selected_decision"]["decision_id"] == "D2"
    assert evidence["verification"]["verification"] == "VERIFIED"
    assert evidence["recovery"]["rewind_operations"] == ["RW-1", "RW-2"]
    assert {
        "feature": "failed_logins",
        "decision_id": "D4",
        "path": ["failed_logins", "D2", "D4"],
    } in evidence["provenance_paths"]
    assert evidence["universal_log_context"]["events"][0]["details"]["message"] == "D2 prediction LOW"
    assert evidence["authority"]["persisted_database_state"] == "authoritative"


def test_chat_builds_authoritative_evidence_in_backend_not_from_client(tmp_path, monkeypatch):
    from backend.app import main

    log = UniversalLog(tmp_path / "universal_log.jsonl")
    log.record(
        "application.log",
        "backend",
        {"message": "Persisted D2 current output HIGH", "dataset_id": "DS-1", "event_id": "EVT-1"},
    )
    monkeypatch.setattr(main, "universal_log", log)
    monkeypatch.setattr(main, "dataset_store", EvidenceStore())

    def explain(question, evidence):
        assert evidence["decisions"]["current"]["D2"] == "HIGH"
        assert evidence["authority"]["persisted_database_state"] == "authoritative"
        assert evidence["universal_log_context"]["events"]
        return {"source": "groq:test-model", "answer": "Persisted state records D2 as HIGH.", "evidence": evidence}

    monkeypatch.setattr(main.llm, "explain", explain)
    with TestClient(main.app) as client:
        response = client.post(
            "/api/ai/chat",
            json={
                "question": "What is the D2 state?",
                "dataset_id": "DS-1",
                "event_id": "EVT-1",
                "evidence": {"decisions": {"current": {"D2": "FORGED"}}},
            },
        )

    assert response.status_code == 200
    assert response.json()["answer"] == "Persisted state records D2 as HIGH."
