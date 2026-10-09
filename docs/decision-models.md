# Decision Models

The workbench trains five classifiers on a generated 20,000-record training
dataset. Each classifier uses a deterministic, stratified 80/20
train/validation split and is fitted once per model version.

| Decision | Algorithm | Input features |
|---|---|---|
| D1 – Authentication | Logistic Regression | Failed logins, login hour, geo anomaly, source IP trust |
| D2 – Threat Severity | Random Forest | Failed logins, threat score, previous alerts |
| D3 – Asset Protection | Decision Tree | Asset criticality, threat score, previous alerts |
| D4 – Incident Escalation | Logistic Regression | Threat score, geo anomaly, asset criticality, D2 severity |
| D5 – Response Action | Random Forest | D2 severity, asset criticality, geo anomaly, failed logins |

Training and validation loss and accuracy are recorded for each fit. Model
metadata and a SHA-256 checksum are persisted alongside the versioned model
artifact. A separate evaluation command reports held-out accuracy, weighted
precision, recall, F1, and confusion matrices:

```powershell
py -3.11 -m backend.app.ml.train
```

Counterfactual replay uses the exact model artifact pinned to an experiment.
It does not fit or update a model. D2 is evaluated before D4 and D5 because
those classifiers consume the D2 severity output. Persisted state and
deterministic rewind verification remain authoritative.
