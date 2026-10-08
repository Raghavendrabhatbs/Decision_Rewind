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


def verify_rewind_persistence(
    original_state: Dict[str, Any],
    previous_state: Dict[str, Any],
    recovered_state: Dict[str, Any],
    previous_decisions: List[Dict[str, Any]],
    recovered_decisions: List[Dict[str, Any]],
    counterfactual: Dict[str, str],
    affected: List[str],
    correction_changes: List[Dict[str, Any]],
    model_version: str,
    training_dataset_id: str,
) -> Dict[str, Any]:
    errors: List[str] = []
    if original_state.get("event_id") != previous_state.get("event_id"):
        errors.append("Historical and current event identities differ.")
    if previous_state.get("event_id") != recovered_state.get("event_id"):
        errors.append("Recovery changed the event identity.")
    for change in correction_changes:
        if recovered_state.get(change["feature"]) != change["new_value"]:
            errors.append(f"Corrected feature {change['feature']} was not persisted.")

    before = {item["decision_id"]: item for item in previous_decisions}
    after = {item["decision_id"]: item for item in recovered_decisions}
    if set(before) != set(after) or set(before) != set(counterfactual):
        errors.append("Persisted decision set does not match the counterfactual decision set.")

    affected_set = set(affected)
    collateral_changes = 0
    verified_decisions = 0
    for decision_id, previous in before.items():
        recovered = after.get(decision_id)
        if recovered is None:
            errors.append(f"Decision {decision_id} is missing after recovery.")
            continue
        expected_current = (
            counterfactual[decision_id]
            if decision_id in affected_set
            else previous["current_output"]
        )
        if recovered["current_output"] != expected_current:
            errors.append(f"Decision {decision_id} did not retain or recover its expected output.")
            if decision_id not in affected_set:
                collateral_changes += 1
        if recovered["historical_output"] != previous["historical_output"]:
            errors.append(f"Historical output for {decision_id} was modified.")
        original_output = original_state.get("decision_outputs", {}).get(decision_id)
        if original_output is not None and original_output != previous["historical_output"]:
            errors.append(f"Persisted historical output for {decision_id} does not match the original event.")
        if recovered["model_version"] != model_version or previous["model_version"] != model_version:
            errors.append(f"Decision {decision_id} does not retain the experiment model version.")
        if (
            recovered["training_dataset_id"] != training_dataset_id
            or previous["training_dataset_id"] != training_dataset_id
        ):
            errors.append(f"Decision {decision_id} does not retain the experiment training dataset.")
        if recovered_state.get("decision_outputs", {}).get(decision_id) != recovered["current_output"]:
            errors.append(f"Event and decision-table output disagree for {decision_id}.")
        if (
            decision_id in affected_set
            and recovered["current_output"] == counterfactual.get(decision_id)
            and recovered["historical_output"] == previous["historical_output"]
        ):
            verified_decisions += 1
        elif decision_id not in affected_set and recovered["current_output"] == previous["current_output"]:
            verified_decisions += 1

    return {
        "verification": "VERIFIED" if not errors else "REJECTED",
        "verified": not errors,
        "affected_decisions": len(affected_set),
        "verified_decisions": verified_decisions,
        "collateral_changes": collateral_changes,
        "errors": errors,
        "reasons": errors,
    }
