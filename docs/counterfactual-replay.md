# Counterfactual Replay

Counterfactual replay replaces only the corrected feature(s) and re-evaluates the same decision logic compared with the historical state.

If the historical output matches the counterfactual output, the decision is not marked as affected. If both differ and the decision is reachable from the corrected feature, it becomes a rewind candidate.
