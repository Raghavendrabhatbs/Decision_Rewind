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

# Classical supervised ML is used deliberately: the five decisions have
# different output characteristics, so one algorithm is not forced onto all
# decisions.
#
# D1/D4 are binary decisions -> Logistic Regression.
# D2/D5 have nonlinear multi-factor interactions -> Random Forest.
# D3 is intentionally interpretable -> Decision Tree.
ALGORITHMS = {
    "D1": "Logistic Regression",
    "D2": "Random Forest",
    "D3": "Decision Tree",
    "D4": "Logistic Regression",
    "D5": "Random Forest",
}

# Kept as a compatibility field for the existing training-job schema/API.
# These sklearn models are not neural networks and therefore do not train in
# neural-network epochs. One fitting pass is performed per decision model.
TRAINING_EPOCHS = 1

FEATURES = {
    "D1": ["failed_logins", "login_hour", "geo_anomaly", "source_ip"],
    "D2": ["failed_logins", "threat_intel_score", "previous_alerts"],
    "D3": ["asset_criticality", "threat_intel_score", "previous_alerts"],
    "D4": ["threat_intel_score", "geo_anomaly", "asset_criticality", "d2_severity"],
    "D5": ["d2_severity", "asset_criticality", "geo_anomaly", "failed_logins"],
}


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


def _make_model(decision_id: str, seed: int):
    if decision_id in {"D1", "D4"}:
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, random_state=seed)),
            ]
        )

    if decision_id in {"D2", "D5"}:
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            random_state=seed,
            n_jobs=-1,
        )

    return DecisionTreeClassifier(
        max_depth=8,
        random_state=seed,
    )


def _evaluate_model(model: Any, X_train: np.ndarray, y_train: np.ndarray, X_validation: np.ndarray, y_validation: np.ndarray) -> Dict[str, float]:
    train_probabilities = model.predict_proba(X_train)
    validation_probabilities = model.predict_proba(X_validation)
    train_predictions = model.predict(X_train)
    validation_predictions = model.predict(X_validation)

    return {
        "training_loss": float(log_loss(y_train, train_probabilities, labels=model.classes_)),
        "validation_loss": float(log_loss(y_validation, validation_probabilities, labels=model.classes_)),
        "training_accuracy": float(accuracy_score(y_train, train_predictions)),
        "validation_accuracy": float(accuracy_score(y_validation, validation_predictions)),
    }


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

    # Ground-truth D2 is used to construct the supervised training features for
    # D4/D5. At inference/replay time the frozen D2 model output is used instead.
    labels_by_event = {event["event_id"]: build_rule_decisions(event) for event in events}

    for model_index, decision_id in enumerate(DECISION_IDS):
        X = np.asarray(
            [_features(event, decision_id, labels_by_event[event["event_id"]]["D2"]) for event in events],
            dtype=np.float32,
        )
        y = np.asarray([labels_by_event[event["event_id"]][decision_id] for event in events])

        X_train, X_validation, y_train, y_validation = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=training_seed,
            stratify=y,
        )

        model = _make_model(decision_id, training_seed + model_index)
        model.fit(X_train, y_train)
        stats = _evaluate_model(model, X_train, y_train, X_validation, y_validation)

        # The old API/database calls this field "epoch". For these classical
        # estimators it represents the single model-fitting pass.
        best_epoch = 1
        best_epochs[decision_id] = best_epoch
        validation_metrics[decision_id] = {
            **stats,
            "algorithm": ALGORITHMS[decision_id],
        }
        models[decision_id] = {
            "algorithm": ALGORITHMS[decision_id],
            "feature_names": FEATURES[decision_id],
            "scaler": None,
            "model": model,
        }

        on_epoch(decision_id, best_epoch, validation_metrics[decision_id])

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
        "algorithms": ALGORITHMS,
        "training_timestamp": trained_at,
        "validation_metrics": validation_metrics,
        "artifact_path": str(artifact_path),
        "artifact_sha256": version_hash,
    }


def predict_workbench_decisions(
    event: Dict[str, Any],
    model_path: str | Path | Dict[str, Any],
) -> Dict[str, str]:
    artifacts = model_path if isinstance(model_path, dict) else joblib.load(model_path)
    result: Dict[str, str] = {}

    # D1 -> D2 -> D3 -> D4 -> D5. D2's frozen prediction is explicitly
    # propagated into the downstream D4/D5 feature vectors.
    d1_artifact = artifacts["D1"]
    d1_vector = np.asarray([_features(event, "D1", "LOW")], dtype=np.float32)
    result["D1"] = str(d1_artifact["model"].predict(d1_vector)[0])

    d2_artifact = artifacts["D2"]
    d2_vector = np.asarray([_features(event, "D2", "LOW")], dtype=np.float32)
    d2 = str(d2_artifact["model"].predict(d2_vector)[0])
    result["D2"] = d2

    for decision_id in ("D3", "D4", "D5"):
        artifact = artifacts[decision_id]
        vector = np.asarray([_features(event, decision_id, d2)], dtype=np.float32)
        result[decision_id] = str(artifact["model"].predict(vector)[0])

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
