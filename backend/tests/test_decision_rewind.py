from backend.app.counterfactual.replay import analyze_correction
from backend.app.dataset.generator import generate_dataset
from backend.app.ml.model_registry import ModelRegistry
from backend.app.provenance.graph import build_provenance_graph, query_graph
from backend.app.verification.verifier import deterministic_verifier


def test_dataset_generation_and_demo_event():
    events = generate_dataset(size=1000, seed=7)
    assert len(events) == 1000
    demo = next(item for item in events if item["event_id"] == "SEC-000123")
    assert demo["asset_criticality"] == "LOW"
    assert demo["decision_outputs"]["D3"] in {"PROTECT", "NORMAL"}


def test_default_dataset_size_is_20000():
    assert len(generate_dataset()) == 20000


def test_asset_criticality_change_impacts_only_relevant_decisions():
    registry = ModelRegistry(seed=42, training_size=1000)
    event = next(item for item in generate_dataset(size=1000, seed=42) if item["event_id"] == "SEC-000123")
    analysis = analyze_correction(event, {"asset_criticality": "CRITICAL"}, registry)
    affected_ids = {item["decision_id"] for item in analysis["decisions"] if item["affected"]}
    assert "D3" in affected_ids
    assert "D4" in affected_ids
    assert "D5" in affected_ids
    assert "D1" not in affected_ids
    assert "D2" not in affected_ids


def test_feature_used_only_by_d1_does_not_rewind_unrelated_decisions():
    registry = ModelRegistry(seed=42, training_size=1000)
    event = next(item for item in generate_dataset(size=1000, seed=42) if item["event_id"] == "SEC-000123")
    analysis = analyze_correction(event, {"failed_logins": 18}, registry)
    affected_ids = {item["decision_id"] for item in analysis["decisions"] if item["affected"]}
    assert "D1" in affected_ids or "D2" in affected_ids or "D5" in affected_ids
    assert "D3" not in affected_ids and "D4" not in affected_ids


def test_identical_counterfactual_is_not_rewound():
    registry = ModelRegistry(seed=42, training_size=1000)
    event = next(item for item in generate_dataset(size=1000, seed=42) if item["event_id"] == "SEC-000123")
    event["asset_criticality"] = "LOW"
    analysis = analyze_correction(event, {"asset_criticality": "LOW"}, registry)
    assert all(not item["affected"] for item in analysis["decisions"])


def test_verifier_rejects_invalid_recovery():
    registry = ModelRegistry(seed=42, training_size=1000)
    event = next(item for item in generate_dataset(size=1000, seed=42) if item["event_id"] == "SEC-000123")
    analysis = analyze_correction(event, {"asset_criticality": "CRITICAL"}, registry)
    result = deterministic_verifier(event, {"unknown_feature": "bad"}, analysis)
    assert result["verification"] == "REJECTED"


def test_dependency_graph_queries_correct_features():
    graph = build_provenance_graph()
    assert "D3" in query_graph("asset_criticality", graph)
    assert "D4" in query_graph("asset_criticality", graph)
    assert "D1" in query_graph("failed_logins", graph)
    assert "D3" not in query_graph("failed_logins", graph)
