from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from backend.app.config import MODEL_DIR
from backend.app.dataset.generator import DEFAULT_DATASET_SIZE, generate_dataset


@dataclass
class ModelBundle:
    name: str
    version: str
    model: Pipeline | DecisionTreeClassifier | RandomForestClassifier | LogisticRegression
    feature_names: List[str]
    training_seed: int
    train_version: str = "clean-v1"


class ModelRegistry:
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.models: Dict[str, ModelBundle] = {}
        self.training_events = generate_dataset(size=DEFAULT_DATASET_SIZE, seed=seed)
        self.train_all()

    def _prepare_features(self, event: Dict[str, object], decision_type: str) -> List[float]:
        asset_rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}[event["asset_criticality"]]
        d2_rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
        geo = 1 if event["geo_anomaly"] else 0
        source_trust = 1 if str(event["source_ip"]).startswith(("10.", "192.168.")) else 0
        default = {
            "D1": [float(event["failed_logins"]), float(event["login_hour"]), float(geo), float(source_trust)],
            "D2": [float(event["failed_logins"]), float(event["threat_intel_score"]), float(event["previous_alerts"])],
            "D3": [float(asset_rank), float(event["threat_intel_score"]), float(event["previous_alerts"])],
            "D4": [float(event["threat_intel_score"]), float(geo), float(asset_rank), float(d2_rank.get(event.get("d2_severity", "LOW"), 0))],
            "D5": [float(event.get("d2_severity_score", 0)), float(asset_rank), float(geo), float(event["failed_logins"])],
        }
        return default[decision_type]

    def _labels(self, decision_type: str, events: List[Dict[str, object]]) -> List[str]:
        labels = []
        for event in events:
            event = dict(event)
            event["d2_severity"] = event.get("decision_outputs", {}).get("D2", "LOW")
            event["d2_severity_score"] = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}.get(event["d2_severity"], 0)
            labels.append(event.get("decision_outputs", {}).get(decision_type, "ALLOW"))
        return labels

    def _make_model(self, decision_type: str):
        # Authoritative Decision Rewind algorithms:
        # D1 Logistic Regression, D2 Random Forest, D3 Decision Tree,
        # D4 Logistic Regression, D5 Random Forest.
        if decision_type in {"D1", "D4"}:
            return Pipeline([
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, random_state=self.seed)),
            ])
        if decision_type in {"D2", "D5"}:
            return RandomForestClassifier(
                n_estimators=200, random_state=self.seed, max_depth=8, n_jobs=-1
            )
        if decision_type == "D3":
            return DecisionTreeClassifier(max_depth=5, random_state=self.seed)
        raise ValueError(f"Unknown decision type: {decision_type}")

    def train_all(self) -> Dict[str, ModelBundle]:
        for decision_type in ["D1", "D2", "D3", "D4", "D5"]:
            X = []
            y = []
            for event in self.training_events:
                event_copy = dict(event)
                event_copy["d2_severity"] = event.get("decision_outputs", {}).get("D2", "LOW")
                event_copy["d2_severity_score"] = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}.get(event_copy["d2_severity"], 0)
                X.append(self._prepare_features(event_copy, decision_type))
                y.append(event.get("decision_outputs", {}).get(decision_type, "ALLOW"))
            model = self._make_model(decision_type)
            X_arr = np.asarray(X, dtype=float)
            y_arr = np.asarray(y)
            X_train, X_valid, y_train, y_valid = train_test_split(X_arr, y_arr, test_size=0.2, random_state=self.seed, stratify=y_arr)
            model.fit(X_train, y_train)
            score = accuracy_score(y_valid, model.predict(X_valid))
            version = f"{decision_type.lower()}-v{round(score * 100) + 1}"
            self.models[decision_type] = ModelBundle(
                name=decision_type,
                version=version,
                model=model,
                feature_names=["f1", "f2", "f3", "f4"][:len(X[0])],
                training_seed=self.seed,
                train_version="clean-v1",
            )
        return self.models

    def predict(self, event: Dict[str, object], decision_type: str) -> str:
        model_bundle = self.models[decision_type]
        features = self._prepare_features(event, decision_type)
        raw = np.asarray([features], dtype=float)
        prediction = model_bundle.model.predict(raw)[0]
        return str(prediction)

    def save_registry(self) -> None:
        payload = {}
        for key, bundle in self.models.items():
            payload[key] = {
                "name": bundle.name,
                "version": bundle.version,
                "feature_names": bundle.feature_names,
                "training_seed": bundle.training_seed,
                "train_version": bundle.train_version,
                "algorithm": (
                    "Logistic Regression" if bundle.name in {"D1", "D4"}
                    else "Random Forest" if bundle.name in {"D2", "D5"}
                    else "Decision Tree"
                ),
            }
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        (MODEL_DIR / "model_registry.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_model_registry() -> ModelRegistry:
    registry = ModelRegistry(seed=42)
    registry.save_registry()
    return registry
