# Decision Rewind ML Models

The project uses one authoritative algorithm for each decision:

| Decision | Algorithm | Output |
|---|---|---|
| D1 – Authentication | Logistic Regression | ALLOW / BLOCK |
| D2 – Threat Severity | Random Forest | LOW / MEDIUM / HIGH / CRITICAL |
| D3 – Asset Protection | Decision Tree | NORMAL / PROTECT |
| D4 – Incident Escalation | Logistic Regression | MONITOR / ESCALATE |
| D5 – Response Action | Random Forest | ALLOW / MONITOR / ISOLATE |

The models are trained together on exactly 20,000 records. Classical scikit-learn
models are fitted once per decision; there is no neural-network epoch loop.

Counterfactual rewind reuses the pinned trained model artifact. It applies a
historical correction to the input, runs the same model version again, compares
the historical and counterfactual outputs, and then selectively rewinds only
those decisions that are both dependency-reachable and actually changed.

D2 is evaluated before D4 and D5 because D4/D5 use D2 severity as an input.
The model is never retrained during counterfactual replay.
