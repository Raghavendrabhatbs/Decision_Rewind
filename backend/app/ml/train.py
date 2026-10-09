from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from backend.app.config import MODEL_DIR
from backend.app.dataset.generator import build_rule_decisions, generate_dataset
from backend.app.ml.workbench_model import (
    ALGORITHM_NAMES,
    DECISION_IDS,
    FEATURES,
    TRAINING_RECORD_COUNT,
    _features,
    _make_model,
)

SEED = 42
DECISION_NAMES = {
    "D1": "Authentication",
    "D2": "Threat Severity",
    "D3": "Asset Protection",
    "D4": "Incident Escalation",
    "D5": "Response Action",
}


def train_and_evaluate(seed: int = SEED, record_count: int = TRAINING_RECORD_COUNT) -> Path:
    if record_count < 1:
        raise ValueError("Evaluation record count must be positive.")
    events = generate_dataset(
        size=record_count,
        seed=seed,
        include_demo_event=False,
        include_labels=False,
    )
    if len(events) != record_count:
        raise RuntimeError(
            f"Expected {record_count} records, generated {len(events)}."
        )

    labels = {event["event_id"]: build_rule_decisions(event) for event in events}
    results: Dict[str, Dict[str, Any]] = {}
    for decision_id in DECISION_IDS:
        X = np.asarray(
            [
                _features(event, decision_id, labels[event["event_id"]]["D2"])
                for event in events
            ],
            dtype=np.float64,
        )
        y = np.asarray([labels[event["event_id"]][decision_id] for event in events])
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=seed,
            stratify=y,
        )
        model = _make_model(decision_id, seed)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        class_labels = [str(label) for label in model.classes_]
        results[decision_id] = {
            "decision": DECISION_NAMES[decision_id],
            "algorithm": ALGORITHM_NAMES[decision_id],
            "features": FEATURES[decision_id],
            "dataset_size": len(events),
            "training_size": len(X_train),
            "test_size": len(X_test),
            "training_accuracy": float(accuracy_score(y_train, model.predict(X_train))),
            "test_accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(precision_score(y_test, predictions, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_test, predictions, average="weighted", zero_division=0)),
            "f1_score": float(f1_score(y_test, predictions, average="weighted", zero_division=0)),
            "classes": class_labels,
            "confusion_matrix": confusion_matrix(
                y_test,
                predictions,
                labels=model.classes_,
            ).tolist(),
        }

    metrics_path = MODEL_DIR / "metrics" / "latest_training_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(
            {
                "dataset_size": len(events),
                "training_size": int(len(events) * 0.8),
                "test_size": int(len(events) * 0.2),
                "seed": seed,
                "decisions": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Evaluation metrics saved to {metrics_path}")
    for decision_id, result in results.items():
        print(
            f"{decision_id} {result['algorithm']}: "
            f"accuracy={result['test_accuracy']:.4f}, "
            f"precision={result['precision']:.4f}, "
            f"recall={result['recall']:.4f}, "
            f"F1={result['f1_score']:.4f}"
        )
    return metrics_path


if __name__ == "__main__":
    train_and_evaluate()
