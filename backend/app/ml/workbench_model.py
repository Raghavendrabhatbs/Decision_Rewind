from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from backend.app.config import DATA_DIR, MODEL_DIR
from backend.app.dataset.generator import build_rule_decisions

DECISION_IDS = ["D1", "D2", "D3", "D4", "D5"]
ASSET_RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
SEVERITY_RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
TRAINING_RECORD_COUNT = 20_000
TRAINING_EPOCHS = 1
FEATURES = {
    "D1": ["failed_logins", "login_hour", "geo_anomaly", "source_ip"],
    "D2": ["failed_logins", "threat_intel_score", "previous_alerts"],
    "D3": ["asset_criticality", "threat_intel_score", "previous_alerts"],
    "D4": ["threat_intel_score", "geo_anomaly", "asset_criticality", "d2_severity"],
    "D5": ["d2_severity", "asset_criticality", "geo_anomaly", "failed_logins"],
}

ALGORITHM_NAMES = {
    "D1": "Logistic Regression",
    "D2": "Random Forest",
    "D3": "Decision Tree",
    "D4": "Logistic Regression",
    "D5": "Random Forest",
}


def _make_model(decision_id: str, seed: int):
    if decision_id in {"D1", "D4"}:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, random_state=seed)),
        ])
    if decision_id in {"D2", "D5"}:
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            random_state=seed,
            n_jobs=-1,
        )
    if decision_id == "D3":
        return DecisionTreeClassifier(max_depth=5, random_state=seed)
    raise ValueError(f"Unknown decision type: {decision_id}")


class EpochMLPClassifier:
    """Keep compatibility with existing serialized MLP artifacts."""

    def __init__(self, classes: np.ndarray, weights: Dict[str, np.ndarray]):
        self.classes_ = classes
        self.weights = weights

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        hidden = np.maximum(features @ self.weights["hidden_weights"] + self.weights["hidden_bias"], 0.0)
        logits = hidden @ self.weights["output_weights"] + self.weights["output_bias"]
        logits -= logits.max(axis=1, keepdims=True)
        exponentials = np.exp(logits)
        return exponentials / exponentials.sum(axis=1, keepdims=True)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(features), axis=1)]


def _features(event: Dict[str, Any], decision_id: str, d2_severity: str) -> List[float]:
    source_trust = str(event["source_ip"]).startswith(("10.", "192.168."))
    values = {
        "failed_logins": float(event["failed_logins"]),
        "login_hour": float(event["login_hour"]),
        "geo_anomaly": float(bool(event["geo_anomaly"])),
        "source_ip": float(source_trust),
        "threat_intel_score": float(event["threat_intel_score"]),
        "previous_alerts": float(event["previous_alerts"]),
        "asset_criticality": float(ASSET_RANK[event["asset_criticality"]]),
        "d2_severity": float(SEVERITY_RANK[d2_severity]),
    }
    return [values[name] for name in FEATURES[decision_id]]


def save_training_dataset(training_dataset_id: str, events: List[Dict[str, Any]]) -> Path:
    path = DATA_DIR / "training" / f"{training_dataset_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(events, indent=2), encoding="utf-8")
    return path


def train_model_version(
    training_dataset_id: str,
    training_seed: int,
    events: List[Dict[str, Any]],
    model_version: str,
    on_epoch: Callable[[str, int, Dict[str, float]], None],
) -> Dict[str, Any]:
    if len(events) != TRAINING_RECORD_COUNT:
        raise ValueError(f"Model training requires exactly {TRAINING_RECORD_COUNT} records; received {len(events)}.")

    models: Dict[str, Dict[str, Any]] = {}
    best_epochs: Dict[str, int] = {}
    validation_metrics: Dict[str, Dict[str, float]] = {}
    labels_by_event = {event["event_id"]: build_rule_decisions(event) for event in events}
    for decision_id in DECISION_IDS:
        X = np.asarray(
            [_features(event, decision_id, labels_by_event[event["event_id"]]["D2"]) for event in events],
            dtype=np.float64,
        )
        y = np.asarray([labels_by_event[event["event_id"]][decision_id] for event in events])
        X_train, X_validation, y_train, y_validation = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=training_seed,
            stratify=y,
        )
        model = _make_model(decision_id, training_seed)
        model.fit(X_train, y_train)
        training_probabilities = model.predict_proba(X_train)
        validation_probabilities = model.predict_proba(X_validation)
        stats = {
            "training_loss": float(log_loss(y_train, training_probabilities, labels=model.classes_)),
            "validation_loss": float(log_loss(y_validation, validation_probabilities, labels=model.classes_)),
            "training_accuracy": float(accuracy_score(y_train, model.predict(X_train))),
            "validation_accuracy": float(accuracy_score(y_validation, model.predict(X_validation))),
        }
        on_epoch(decision_id, 1, stats)
        models[decision_id] = {
            "algorithm": ALGORITHM_NAMES[decision_id],
            "feature_names": FEATURES[decision_id],
            "model": model,
        }
        best_epochs[decision_id] = 1
        validation_metrics[decision_id] = stats

    trained_at = datetime.now(timezone.utc).isoformat()
    model_id = f"SOC-MODEL-{model_version}"
    artifact_path = MODEL_DIR / "versions" / f"{model_version}.joblib"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, artifact_path)
    version_hash = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    return {
        "model_id": model_id,
        "model_version": model_version,
        "training_dataset_id": training_dataset_id,
        "training_seed": training_seed,
        "training_record_count": len(events),
        "epochs": TRAINING_EPOCHS,
        "best_epoch": best_epochs,
        "training_timestamp": trained_at,
        "validation_metrics": validation_metrics,
        "algorithms": ALGORITHM_NAMES,
        "feature_sets": FEATURES,
        "artifact_path": str(artifact_path),
        "artifact_sha256": version_hash,
    }


def predict_workbench_decisions(
    event: Dict[str, Any],
    model_path: str | Path | Dict[str, Any],
) -> Dict[str, str]:
    artifacts = model_path if isinstance(model_path, dict) else joblib.load(model_path)
    result: Dict[str, str] = {}

    def predict(decision_id: str, d2_severity: str = "LOW") -> str:
        artifact = artifacts[decision_id]
        features = [_features(event, decision_id, d2_severity)]
        if "scaler" in artifact:
            features = artifact["scaler"].transform(features)
        return str(artifact["model"].predict(features)[0])

    result["D1"] = predict("D1")
    d2 = predict("D2")
    result["D2"] = d2
    for decision_id in ("D3", "D4", "D5"):
        result[decision_id] = predict(decision_id, d2)
    return result


def load_workbench_model(
    model_path: str | Path,
    expected_sha256: str | None = None,
) -> Dict[str, Any]:
    path = Path(model_path)
    if expected_sha256 is not None:
        actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_sha256 != expected_sha256:
            raise ValueError("Pinned model artifact failed its SHA-256 integrity check.")
    return joblib.load(path)
