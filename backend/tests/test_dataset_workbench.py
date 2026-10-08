import hashlib
from pathlib import Path

import pytest

from backend.app.dataset.generator import generate_dataset
from backend.app.ml import workbench_model
from backend.app.ml.workbench_model import load_workbench_model, predict_workbench_decisions, train_model_version
from backend.app.services.dataset_store import DatasetStore


def _trained_store(tmp_path, monkeypatch):
    monkeypatch.setattr(workbench_model, "TRAINING_RECORD_COUNT", 200)
        monkeypatch.setattr(workbench_model, "MODEL_DIR", tmp_path / "models")
    monkeypatch.setattr(workbench_model, "DATA_DIR", tmp_path / "data")
    dataset_store = DatasetStore(tmp_path / "workbench.db")
    events = generate_dataset(size=200, seed=7, include_demo_event=False)
    dataset_store.create_training_job("JOB-TEST", "V1", "TRN-TEST", 7, 200, workbench_model.TRAINING_EPOCHS)
    training = train_model_version("TRN-TEST", 7, events, "V1", lambda *_: None)
    dataset_store.complete_training_job("JOB-TEST", training)
    model = dataset_store.get_active_model()
    assert model is not None
    generated = dataset_store.create_dataset("DS-TEST", 7, events, model)
    assert generated["training_status"] == "NOT_TRAINED"
    assert dataset_store.get_event("DS-TEST", events[0]["event_id"])["decisions"] == []
    predictions = {
        event["event_id"]: predict_workbench_decisions(event, load_workbench_model(model["artifact_path"]))
        for event in dataset_store.get_training_events("DS-TEST")
    }
    dataset = dataset_store.complete_experiment_training("DS-TEST", predictions)
    assert dataset["training_status"] == "TRAINED"
    assert dataset["workflow_status"] == "READY_FOR_CORRECTION"
    return dataset_store, events, dataset


def test_dataset_generation_has_200_new_non_demo_records():
    events = generate_dataset(size=200, seed=42, include_demo_event=False)
    other_events = generate_dataset(size=200, seed=43, include_demo_event=False)
    assert len(events) == 200
    assert len({item["event_id"] for item in events}) == 200
    assert events != other_events
    assert not any(item["user_id"] == "USR-4207" for item in events)
    unlabeled = generate_dataset(size=200, seed=42, include_demo_event=False, include_labels=False)
    assert all("decision_outputs" not in item for item in unlabeled)


def test_global_training_fits_each_decision_model_and_saves_immutable_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(workbench_model, "TRAINING_RECORD_COUNT", 200)
    monkeypatch.setattr(workbench_model, "MODEL_DIR", tmp_path / "models")
    monkeypatch.setattr(workbench_model, "DATA_DIR", tmp_path / "data")
    events = generate_dataset(size=200, seed=7, include_demo_event=False)
    progress = []

    training = train_model_version("TRN-EPOCHS", 7, events, "V1", lambda *args: progress.append(args))

    assert training["training_record_count"] == len(events)
    assert workbench_model.TRAINING_EPOCHS == 1
    assert len(progress) == 5
    assert set(training["best_epoch"]) == {"D1", "D2", "D3", "D4", "D5"}
    assert set(training["algorithms"]) == {"D1", "D2", "D3", "D4", "D5"}
    assert training["algorithms"]["D1"] == "Logistic Regression"
    assert training["algorithms"]["D2"] == "Random Forest"
    assert training["algorithms"]["D3"] == "Decision Tree"
    assert training["algorithms"]["D4"] == "Logistic Regression"
    assert training["algorithms"]["D5"] == "Random Forest"
    assert all(0 <= stats["validation_accuracy"] <= 1 for stats in training["validation_metrics"].values())
    model = load_workbench_model(training["artifact_path"])
    artifact_hash = hashlib.sha256(Path(training["artifact_path"]).read_bytes()).hexdigest()
    predict_workbench_decisions(events[0], model)
    assert hashlib.sha256(Path(training["artifact_path"]).read_bytes()).hexdigest() == artifact_hash


def test_training_rejects_any_dataset_other_than_20000_records():
    assert workbench_model.TRAINING_RECORD_COUNT == 20_000
    with pytest.raises(ValueError, match="exactly 20000 records"):
        train_model_version("TRN-SHORT", 7, [{}] * 200, "V1", lambda *_: None)


def test_explicit_training_gates_experiments_and_versions_are_reused(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from backend.app import main
    from backend.app.services.universal_log import UniversalLog

    training_store = DatasetStore(tmp_path / "api-workbench.db")
    monkeypatch.setattr(main, "dataset_store", training_store)
    monkeypatch.setattr(main, "universal_log", UniversalLog(tmp_path / "universal_log.jsonl"))
    monkeypatch.setattr(main, "TRAINING_RECORD_COUNT", 200)
    monkeypatch.setattr(workbench_model, "TRAINING_RECORD_COUNT", 200)
    monkeypatch.setattr(workbench_model, "MODEL_DIR", tmp_path / "models")
    monkeypatch.setattr(workbench_model, "DATA_DIR", tmp_path / "data")

    with TestClient(main.app) as client:
        assert client.get("/api/models/status").json()["active_model"] is None
        assert client.post("/api/datasets").status_code == 409

        first_job = client.post("/api/models/train").json()
        first_status = client.get(f"/api/models/training/{first_job['job_id']}").json()
        assert first_status["status"] == "COMPLETED"
        assert first_status["record_count"] == 200
        assert first_status["epochs"] == 1

        first_experiment = client.post("/api/datasets").json()
        assert first_experiment["training_status"] == "NOT_TRAINED"
        assert first_experiment["model_version"] == "V1"
        first_event = training_store.get_training_events(first_experiment["dataset_id"])[0]
        detail = client.get(
            f"/api/datasets/{first_experiment['dataset_id']}/events/{first_event['event_id']}"
        ).json()
        assert detail["current_state"]["decision_outputs"] == {}
        assert detail["decisions"] == []

        first_experiment = client.post(f"/api/datasets/{first_experiment['dataset_id']}/train").json()
        assert first_experiment["training_status"] == "TRAINED"
        assert first_experiment["training_record_count"] == 200
        assert first_experiment["model_version"] == "V1"
        detail = client.get(
            f"/api/datasets/{first_experiment['dataset_id']}/events/{first_event['event_id']}"
        ).json()
        assert len(detail["decisions"]) == 5

        second_job = client.post("/api/models/train").json()
        second_status = client.get(f"/api/models/training/{second_job['job_id']}").json()
        assert second_status["status"] == "COMPLETED"
        second_experiment = client.post("/api/datasets").json()
        assert second_experiment["training_status"] == "NOT_TRAINED"
        assert second_experiment["model_version"] == "V2"
        assert training_store.get_dataset(first_experiment["dataset_id"])["model_version"] == "V1"
        changes = [
            {"feature": "failed_logins", "new_value": (first_event["failed_logins"] + 1) % 21},
            {"feature": "geo_anomaly", "new_value": not first_event["geo_anomaly"]},
            {
                "feature": "threat_intel_score",
                "new_value": 0.0 if first_event["threat_intel_score"] > 0.0 else 1.0,
            },
        ]
        preview = client.post(
            f"/api/datasets/{first_experiment['dataset_id']}/preview",
            json={"event_id": first_event["event_id"], "changes": changes},
        )
        assert preview.status_code == 200, preview.text
        assert len(preview.json()["changes"]) == 3
        correction = client.post(
            f"/api/datasets/{first_experiment['dataset_id']}/corrections",
            json={"event_id": first_event["event_id"], "changes": changes},
        )
        assert correction.status_code == 200, correction.text
        rewind = client.post(
            f"/api/datasets/{first_experiment['dataset_id']}/rewind",
            json={"correction_id": correction.json()["correction_id"]},
        )
        assert rewind.status_code == 200, rewind.text
        assert rewind.json()["verification"]["verification"] == "VERIFIED"
        recovered_event = training_store.get_event(first_experiment["dataset_id"], first_event["event_id"])
        assert recovered_event["original_state"]["failed_logins"] == first_event["failed_logins"]
        assert recovered_event["current_state"]["failed_logins"] == changes[0]["new_value"]
        assert training_store.get_dataset(first_experiment["dataset_id"])["model_version"] == "V1"
        assert {
            record["event_type"]
            for record in main.universal_log.recent(100)
            if record["details"].get("event_id") == first_event["event_id"]
        } >= {
            "counterfactual.previewed",
            "workflow.correction.proposed",
            "workflow.verification.completed",
            "workflow.rewind.completed",
        }
        pinned_model = training_store.get_model(second_experiment["model_id"])
        artifact = Path(pinned_model["artifact_path"])
        artifact.write_bytes(artifact.read_bytes() + b"tampered")
        rejected = client.post(f"/api/datasets/{second_experiment['dataset_id']}/train")
        assert rejected.status_code == 409
        assert "integrity check" in rejected.json()["detail"]
        assert training_store.get_dataset(second_experiment["dataset_id"])["training_status"] == "NOT_TRAINED"


def test_training_metadata_and_historical_decisions_are_persisted(tmp_path, monkeypatch):
    dataset_store, events, dataset = _trained_store(tmp_path, monkeypatch)
    assert dataset["training_status"] == "TRAINED"
    assert dataset["workflow_status"] == "READY_FOR_CORRECTION"
    assert dataset["training_record_count"] == 200
    assert dataset["model_id"] == "SOC-MODEL-V1"
    assert dataset["model_version"] == "V1"
    assert dataset["trained_at"]

    detail = dataset_store.get_event("DS-TEST", events[0]["event_id"])
    assert len(detail["decisions"]) == 5
    assert all(row["model_version"] == dataset["model_version"] for row in detail["decisions"])
    assert all(row["historical_output"] == row["current_output"] for row in detail["decisions"])


def test_pending_legacy_experiment_hides_partial_labels_until_train_completes(tmp_path, monkeypatch):
    dataset_store, events, dataset = _trained_store(tmp_path, monkeypatch)
    event_id = events[0]["event_id"]
    with dataset_store._connect() as connection:
        connection.execute(
            "UPDATE datasets SET training_status = 'TRAINED', d5_status = 'NOT_LABELED' WHERE dataset_id = ?",
            ("DS-TEST",),
        )
        connection.execute(
            "DELETE FROM dataset_decisions WHERE dataset_id = ? AND decision_id = 'D5'",
            ("DS-TEST",),
        )
    pending = dataset_store.get_dataset("DS-TEST")
    assert pending["training_status"] == "NOT_TRAINED"
    event_before_labeling = dataset_store.get_event("DS-TEST", event_id)
    assert event_before_labeling["decisions"] == []
    assert event_before_labeling["current_state"]["decision_outputs"] == {}

    model = load_workbench_model(dataset["model_path"])
    predictions = {
        event["event_id"]: predict_workbench_decisions(event, model)
        for event in dataset_store.get_training_events("DS-TEST")
    }
    labeled = dataset_store.complete_experiment_training("DS-TEST", predictions)

    assert labeled["training_status"] == "TRAINED"
    assert labeled["label_status"] == "LABELED"
    assert len(dataset_store.get_event("DS-TEST", event_id)["decisions"]) == 5


def test_retraining_creates_new_version_without_overwriting_model_or_experiment(tmp_path, monkeypatch):
    dataset_store, events, experiment = _trained_store(tmp_path, monkeypatch)
    first_model = dataset_store.get_active_model()
    assert first_model is not None

    dataset_store.create_training_job("JOB-V2", "V2", "TRN-V2", 8, 200, workbench_model.TRAINING_EPOCHS)
    second = train_model_version("TRN-V2", 8, events, "V2", lambda *_: None)
    first_artifact_hash = first_model["artifact_sha256"]
    dataset_store.complete_training_job("JOB-V2", second)

    versions = {model["model_version"]: model for model in dataset_store.list_models()}
    assert versions["V1"]["status"] == "RETAINED"
    assert versions["V2"]["status"] == "ACTIVE"
    assert versions["V1"]["artifact_sha256"] == first_artifact_hash
    assert dataset_store.get_dataset("DS-TEST")["model_version"] == experiment["model_version"] == "V1"
    assert dataset_store.get_active_model()["model_version"] == "V2"


def test_multifeature_correction_preserves_history_and_rewinds_only_changed_decisions(tmp_path, monkeypatch):
    dataset_store, events, dataset = _trained_store(tmp_path, monkeypatch)
    event = events[0]
    before = dataset_store.get_event("DS-TEST", event["event_id"])
    proposed = [
        {"feature": "asset_criticality", "new_value": "CRITICAL"},
        {"feature": "failed_logins", "new_value": (event["failed_logins"] + 1) % 21},
    ]
    correction = dataset_store.create_correction("DS-TEST", event["event_id"], proposed)
    before_rewind = dataset_store.get_event("DS-TEST", event["event_id"])
    assert before_rewind["current_state"]["asset_criticality"] == event["asset_criticality"]

    changes = correction["changes"]
    corrected = dict(before["current_state"])
    for change in changes:
        corrected[change["feature"]] = change["new_value"]
    counterfactual = predict_workbench_decisions(corrected, dataset["model_path"])
    changed_ids = [
        row["decision_id"]
        for row in before["decisions"]
        if row["current_output"] != counterfactual[row["decision_id"]]
    ]
    impacts = {
        row["decision_id"]: {
            "historical_output": row["historical_output"],
            "current_output": row["current_output"],
            "counterfactual_output": counterfactual[row["decision_id"]],
            "affected": row["decision_id"] in changed_ids,
        }
        for row in before["decisions"]
    }
    result = dataset_store.complete_rewind(
        correction,
        counterfactual,
        changed_ids,
        impacts,
        {"verification": "VERIFIED", "reasons": []},
    )

    after = dataset_store.get_event("DS-TEST", event["event_id"])
    assert result["current_state"]["asset_criticality"] == "CRITICAL"
    assert after["original_state"]["asset_criticality"] == event["asset_criticality"]
    assert after["original_state"]["decision_outputs"] == {
        row["decision_id"]: row["historical_output"] for row in before["decisions"]
    }
    assert [row["feature"] for row in correction["changes"]] == ["asset_criticality", "failed_logins"]
    for row in after["decisions"]:
        expected = counterfactual[row["decision_id"]] if row["decision_id"] in changed_ids else row["historical_output"]
        assert row["current_output"] == expected


def test_failed_persistence_verification_rolls_back_recovery_transaction(tmp_path, monkeypatch):
    import backend.app.services.dataset_store as dataset_store_module

    dataset_store, events, dataset = _trained_store(tmp_path, monkeypatch)
    event = events[0]
    before = dataset_store.get_event("DS-TEST", event["event_id"])
    correction = dataset_store.create_correction(
        "DS-TEST",
        event["event_id"],
        [
            {"feature": "asset_criticality", "new_value": "CRITICAL"},
            {"feature": "failed_logins", "new_value": (event["failed_logins"] + 1) % 21},
            {"feature": "geo_anomaly", "new_value": not event["geo_anomaly"]},
        ],
    )
    corrected = dict(before["current_state"])
    for change in correction["changes"]:
        corrected[change["feature"]] = change["new_value"]
    counterfactual = predict_workbench_decisions(corrected, dataset["model_path"])
    impacts = {}
    affected = []
    for decision in before["decisions"]:
        decision_id = decision["decision_id"]
        changed = decision["current_output"] != counterfactual[decision_id]
        if changed:
            affected.append(decision_id)
        impacts[decision_id] = {
            "historical_output": decision["historical_output"],
            "current_output": decision["current_output"],
            "counterfactual_output": counterfactual[decision_id],
            "affected": changed,
        }
    monkeypatch.setattr(
        dataset_store_module,
        "verify_rewind_persistence",
        lambda *_: {
            "verification": "REJECTED",
            "verified": False,
            "errors": ["Injected verification failure."],
        },
    )

    with pytest.raises(ValueError, match="Injected verification failure"):
        dataset_store.complete_rewind(
            correction,
            counterfactual,
            affected,
            impacts,
            {"verification": "VERIFIED", "reasons": []},
        )

    after = dataset_store.get_event("DS-TEST", event["event_id"])
    assert after["current_state"] == before["current_state"]
    assert after["decisions"] == before["decisions"]
    assert dataset_store.get_correction(correction["correction_id"])["status"] == "PENDING"
    assert not any(
        record["operation"] in {"RECOVERED", "VERIFIED"}
        for record in dataset_store.list_audit("DS-TEST", limit=None)
    )


def test_rewind_includes_downstream_decisions_when_d2_changes(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from backend.app import main
    from backend.app.services.universal_log import UniversalLog

    dataset_store, events, dataset = _trained_store(tmp_path, monkeypatch)
    monkeypatch.setattr(main, "dataset_store", dataset_store)
    monkeypatch.setattr(main, "universal_log", UniversalLog(tmp_path / "universal_log.jsonl"))
    event = dataset_store.get_event("DS-TEST", events[0]["event_id"])
    current_outputs = {row["decision_id"]: row["current_output"] for row in event["decisions"]}

    def counterfactual_for_changes(_dataset, current, changes):
        outputs = dict(current["decision_outputs"])
        outputs["D2"] = "CRITICAL" if current_outputs["D2"] != "CRITICAL" else "LOW"
        outputs["D4"] = "ESCALATE" if current_outputs["D4"] != "ESCALATE" else "MONITOR"
        outputs["D5"] = "ISOLATE" if current_outputs["D5"] != "ISOLATE" else "ALLOW"
        corrected = dict(current)
        for change in changes:
            corrected[change["feature"]] = change["new_value"]
        return outputs, corrected, [change["feature"] for change in changes]

    monkeypatch.setattr(main, "_counterfactual_for_changes", counterfactual_for_changes)
    proposed_value = (event["current_state"]["failed_logins"] + 1) % 21

    with TestClient(main.app) as client:
        preview_response = client.post(
            "/api/datasets/DS-TEST/preview",
            json={
                "event_id": event["current_state"]["event_id"],
                "changes": [{"feature": "failed_logins", "new_value": proposed_value}],
            },
        )
        assert preview_response.status_code == 200
        preview = preview_response.json()
        assert {item["decision_id"] for item in preview["affected_decisions"]} == {"D2", "D4", "D5"}
        assert ["failed_logins", "D2", "D4"] in preview["decision_impacts"][3]["dependency_paths"]

        correction_response = client.post(
            "/api/datasets/DS-TEST/corrections",
            json={
                "event_id": event["current_state"]["event_id"],
                "changes": [{"feature": "failed_logins", "new_value": proposed_value}],
            },
        )
        assert correction_response.status_code == 200
        correction_id = correction_response.json()["correction_id"]

        rewind_response = client.post(
            "/api/datasets/DS-TEST/rewind",
            json={"correction_id": correction_id},
        )

    assert rewind_response.status_code == 200, rewind_response.text
    rewind = rewind_response.json()
    assert rewind["verification"]["verification"] == "VERIFIED"
    assert {item["decision_id"] for item in rewind["rewind_operations"]} == {"D2", "D4", "D5"}
    assert dataset_store.get_correction(correction_id)["status"] == "REWOUND"


def test_correction_proposal_does_not_mutate_event_state(tmp_path, monkeypatch):
    dataset_store, events, _ = _trained_store(tmp_path, monkeypatch)
    event = events[0]
    dataset_store.create_correction(
        "DS-TEST",
        event["event_id"],
        [{"feature": "asset_criticality", "new_value": "CRITICAL"}],
    )
    state = dataset_store.get_event("DS-TEST", event["event_id"])
    assert state["current_state"]["asset_criticality"] == event["asset_criticality"]
    assert state["original_state"]["asset_criticality"] == event["asset_criticality"]


def test_graph_includes_d2_downstream_decisions_without_duplicate_edges(tmp_path, monkeypatch):
    dataset_store, _, _ = _trained_store(tmp_path, monkeypatch)
    graph = dataset_store.get_graph("DS-TEST", ["failed_logins"])
    edges = [(edge["source"], edge["target"]) for edge in graph["edges"]]
    nodes = {node["id"]: node["kind"] for node in graph["nodes"]}

    assert ("failed_logins", "D2") in edges
    assert ("D2", "D4") in edges
    assert ("D2", "D5") in edges
    assert len(edges) == len(set(edges))
    assert nodes["D2"] == "decision"
    assert nodes["failed_logins"] == "feature"
    assert nodes["D2-output"] == "output"
    assert ("failed_logins", "D2-output") not in edges


def test_audit_can_return_complete_history_for_download(tmp_path, monkeypatch):
    dataset_store, events, _ = _trained_store(tmp_path, monkeypatch)
    for index in range(105):
        dataset_store.log_workflow_transition(
            "DS-TEST",
            events[0]["event_id"],
            "AUDIT_EXPORT_TEST",
            {"sequence": index},
        )

    assert len(dataset_store.list_audit("DS-TEST")) == 100
    complete_history = dataset_store.list_audit("DS-TEST", limit=None)
    assert len(complete_history) == 107
    assert complete_history[0]["details"] == {"sequence": 104}
