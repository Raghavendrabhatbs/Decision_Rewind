# Decision Models

The decision layer contains five distinct decisions with different feature subsets and intentionally different classical ML algorithms.

| Decision | Meaning | Main features | Algorithm | Output |
|---|---|---|---|---|
| D1 | Authentication Decision | failed logins, login hour, geo anomaly, IP trust | Logistic Regression | ALLOW / BLOCK |
| D2 | Threat Severity Decision | failed logins, threat-intel score, previous alerts | Random Forest | LOW / MEDIUM / HIGH / CRITICAL |
| D3 | Asset Protection Decision | asset criticality, threat-intel score, previous alerts | Decision Tree | NORMAL / PROTECT |
| D4 | Incident Escalation Decision | threat severity, geo anomaly, asset criticality, D2 severity | Logistic Regression | MONITOR / ESCALATE |
| D5 | Response Action Decision | D2 severity, asset criticality, geo anomaly, failed logins | Random Forest | ALLOW / MONITOR / ISOLATE |

## Why these algorithms?

- **D1 — Logistic Regression:** binary access decision; probability-oriented and interpretable.
- **D2 — Random Forest:** threat severity depends on nonlinear interactions among multiple security indicators.
- **D3 — Decision Tree:** protection rules should remain easy to inspect and explain.
- **D4 — Logistic Regression:** escalation is a binary decision and benefits from a transparent probability-based model.
- **D5 — Random Forest:** response selection combines several interacting risk/context features.

All five models are trained from clean data and versioned as one frozen model artifact. Counterfactual rewind reuses the pinned artifact and reruns the same models on corrected features; it does not retrain the models during rewind.

The provenance graph remains independent of the estimator implementation and records feature → decision → output dependencies.
