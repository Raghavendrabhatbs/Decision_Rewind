from __future__ import annotations

from typing import Any, Dict

from backend.app.dataset.generator import build_rule_decisions
from backend.app.ml.model_registry import ModelRegistry
from backend.app.provenance.graph import FEATURE_TO_DECISIONS


class CounterfactualReplay:
    def __init__(self, registry: ModelRegistry):
        self.registry = registry

    def replay_decision(self, event: Dict[str, Any], corrected_features: Dict[str, Any]) -> Dict[str, Any]:
        dataset_event = dict(event)
        for key, value in corrected_features.items():
            dataset_event[key] = value
        historical = event.get("decision_outputs", {})
        if self.registry is not None:
            # Counterfactual replay re-executes the existing trained models on
            # corrected historical input. It does not retrain any model. D2 is
            # computed first because D4/D5 depend on its severity.
            counterfactual = {}
            counterfactual["D1"] = self.registry.predict(dataset_event, "D1")
            counterfactual["D2"] = self.registry.predict(dataset_event, "D2")
            dataset_event["d2_severity"] = counterfactual["D2"]
            dataset_event["d2_severity_score"] = {
                "LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3
            }.get(counterfactual["D2"], 0)
            for decision_id in ("D3", "D4", "D5"):
                counterfactual[decision_id] = self.registry.predict(dataset_event, decision_id)
        else:
            # Legacy fallback used only by the older non-dataset endpoint.
            counterfactual = build_rule_decisions(dataset_event)
        decisions = []
        for decision_id in ["D1", "D2", "D3", "D4", "D5"]:
            reachable = any(
                decision_id in FEATURE_TO_DECISIONS.get(feature_name, [])
                for feature_name in corrected_features
            )
            affected = reachable and historical.get(decision_id) != counterfactual.get(decision_id)
            decisions.append(
                {
                    "decision_id": decision_id,
                    "decision_type": "Decision-" + decision_id,
                    "historical_value": historical.get(decision_id, "UNKNOWN"),
                    "counterfactual_value": counterfactual.get(decision_id, "UNKNOWN"),
                    "affected": affected,
                    "recovery_action": "REWIND" if affected else "UNTOUCHED",
                    "verification_status": "PENDING",
                }
            )
        return {"event_id": event["event_id"], "corrected_features": corrected_features, "decisions": decisions}


def analyze_correction(event: Dict[str, Any], corrected_features: Dict[str, Any], registry: ModelRegistry) -> Dict[str, Any]:
    engine = CounterfactualReplay(registry)
    return engine.replay_decision(event, corrected_features)
