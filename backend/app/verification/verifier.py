from __future__ import annotations

from typing import Any, Dict, List

from backend.app.provenance.graph import downstream_decisions_for_feature


def deterministic_verifier(event: Dict[str, Any], corrected_features: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
    reasons: List[str] = []
    if not corrected_features:
        reasons.append("No corrected input supplied.")
    for feature_name, value in corrected_features.items():
        if feature_name not in {"asset_criticality", "failed_logins", "login_hour", "geo_anomaly", "threat_intel_score", "previous_alerts", "source_ip"}:
            reasons.append(f"Feature {feature_name} is unsupported.")
    if any(feature not in event for feature in corrected_features):
        reasons.append("Corrected feature not present in event.")
    for decision in analysis.get("decisions", []):
        decision_id = decision["decision_id"]
        feature_reachable = any(
            decision_id in downstream_decisions_for_feature(feature_name)
            for feature_name in corrected_features
        )
        if decision["affected"] and not feature_reachable:
            reasons.append(f"Decision {decision_id} is marked affected but is not reachable from the corrected feature.")
        if not decision["affected"] and decision["historical_value"] != decision["counterfactual_value"]:
            reasons.append(f"Decision {decision_id} changed but was not marked affected.")
    verification = "VERIFIED" if not reasons else "REJECTED"
    return {"verification": verification, "reasons": reasons}
