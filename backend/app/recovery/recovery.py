from __future__ import annotations

from typing import Any, Dict, List

from backend.app.provenance.graph import downstream_decisions_for_feature


def select_recovery_queue(analysis: Dict[str, Any], corrected_features: Dict[str, Any]) -> List[Dict[str, Any]]:
    queue = []
    for item in analysis.get("decisions", []):
        feature_reachable = False
        for feature_name, feature_value in corrected_features.items():
            for decision_id in downstream_decisions_for_feature(feature_name):
                if item["decision_id"] == decision_id:
                    feature_reachable = True
                    break
        if feature_reachable and item["affected"]:
            queue.append(
                {
                    "decision_id": item["decision_id"],
                    "decision_type": item["decision_type"],
                    "historical_value": item["historical_value"],
                    "counterfactual_value": item["counterfactual_value"],
                    "affected": True,
                    "recovery_action": "REWIND",
                    "verification_status": "PENDING",
                }
            )
    return queue
