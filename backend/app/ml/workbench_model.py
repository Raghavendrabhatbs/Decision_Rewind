from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from backend.app.config import DATA_DIR, MODEL_DIR
from backend.app.dataset.generator import build_rule_decisions

DECISION_IDS = ["D1", "D2", "D3", "D4", "D5"]
ASSET_RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
SEVERITY_RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
TRAINING_RECORD_COUNT = 20_000
TRAINING_EPOCHS = 100
FEATURES = {
    "D1": ["failed_logins", "login_hour", "geo_anomaly", "source_ip"],
    "D2": ["failed_logins", "threat_intel_score", "previous_alerts"],
    "D3": ["asset_criticality", "threat_intel_score", "previous_alerts"],
    "D4": ["threat_intel_score", "geo_anomaly", "asset_criticality", "d2_severity"],
    "D5": ["d2_severity", "asset_criticality", "geo_anomaly", "failed_logins"],
}


class EpochMLPClassifier:
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
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_validation = scaler.transform(X_validation)
        classes = np.unique(y)
        y_train = np.searchsorted(classes, y_train)
        y_validation = np.searchsorted(classes, y_validation)
        rng = np.random.default_rng(training_seed + model_index)
        hidden_size = 32
        weights = {
            "hidden_weights": rng.normal(0, np.sqrt(2 / X_train.shape[1]), (X_train.shape[1], hidden_size)),
            "hidden_bias": np.zeros(hidden_size),
            "output_weights": rng.normal(0, np.sqrt(2 / hidden_size), (hidden_size, len(classes))),
            "output_bias": np.zeros(len(classes)),
        }
        first_moment = {name: np.zeros_like(value) for name, value in weights.items()}
        second_moment = {name: np.zeros_like(value) for name, value in weights.items()}
        best_loss = float("inf")
        best_model: EpochMLPClassifier | None = None
        best_epoch = 0
        best_stats: Dict[str, float] = {}
        for epoch in range(1, TRAINING_EPOCHS + 1):
            hidden = np.maximum(X_train @ weights["hidden_weights"] + weights["hidden_bias"], 0.0)
            logits = hidden @ weights["output_weights"] + weights["output_bias"]
            logits -= logits.max(axis=1, keepdims=True)
            train_probabilities = np.exp(logits)
            train_probabilities /= train_probabilities.sum(axis=1, keepdims=True)
            gradient_logits = train_probabilities.copy()
            gradient_logits[np.arange(len(y_train)), y_train] -= 1.0
            gradient_logits /= len(y_train)
            gradient_hidden = gradient_logits @ weights["output_weights"].T
            gradient_hidden[hidden <= 0] = 0
            gradients = {
                "hidden_weights": X_train.T @ gradient_hidden,
                "hidden_bias": gradient_hidden.sum(axis=0),
                "output_weights": hidden.T @ gradient_logits,
                "output_bias": gradient_logits.sum(axis=0),
            }
            for name, gradient in gradients.items():
                first_moment[name] = 0.9 * first_moment[name] + 0.1 * gradient
                second_moment[name] = 0.999 * second_moment[name] + 0.001 * gradient**2
                corrected_first = first_moment[name] / (1 - 0.9**epoch)
                corrected_second = second_moment[name] / (1 - 0.999**epoch)
                weights[name] -= 0.005 * corrected_first / (np.sqrt(corrected_second) + 1e-8)
            model = EpochMLPClassifier(classes, weights)
            train_probabilities = model.predict_proba(X_train)
            validation_probabilities = model.predict_proba(X_validation)
            train_predictions = np.argmax(train_probabilities, axis=1)
            validation_predictions = np.argmax(validation_probabilities, axis=1)
            stats = {
                "training_loss": float(-np.log(np.maximum(train_probabilities[np.arange(len(y_train)), y_train], 1e-12)).mean()),
                "validation_loss": float(-np.log(np.maximum(validation_probabilities[np.arange(len(y_validation)), y_validation], 1e-12)).mean()),
                "training_accuracy": float(np.mean(train_predictions == y_train)),
                "validation_accuracy": float(np.mean(validation_predictions == y_validation)),
            }
            if stats["validation_loss"] < best_loss:
                best_loss = stats["validation_loss"]
                best_epoch = epoch
                best_model = EpochMLPClassifier(classes.copy(), {name: value.copy() for name, value in weights.items()})
                best_stats = stats
            on_epoch(decision_id, epoch, stats)
        if best_model is None:
            raise RuntimeError(f"Training failed to produce a checkpoint for {decision_id}.")
        models[decision_id] = {"scaler": scaler, "model": best_model}
        best_epochs[decision_id] = best_epoch
        validation_metrics[decision_id] = best_stats

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
        "artifact_path": str(artifact_path),
        "artifact_sha256": version_hash,
    }


def predict_workbench_decisions(
    event: Dict[str, Any],
    model_path: str | Path | Dict[str, Any],
) -> Dict[str, str]:
    artifacts = model_path if isinstance(model_path, dict) else joblib.load(model_path)
    result: Dict[str, str] = {}
    d1_artifact = artifacts["D1"]
    d1_vector = d1_artifact["scaler"].transform([_features(event, "D1", "LOW")])
    result["D1"] = str(d1_artifact["model"].predict(d1_vector)[0])

    d2_artifact = artifacts["D2"]
    d2_vector = d2_artifact["scaler"].transform([_features(event, "D2", "LOW")])
    d2 = str(d2_artifact["model"].predict(d2_vector)[0])
    result["D2"] = d2
    for decision_id in ("D3", "D4", "D5"):
        artifact = artifacts[decision_id]
        vector = artifact["scaler"].transform([_features(event, decision_id, d2)])
        result[decision_id] = str(artifact["model"].predict(vector)[0])
    return result


def load_workbench_model(model_path: str | Path) -> Dict[str, Any]:
    return joblib.load(model_path)
