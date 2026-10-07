# Decision Models

The decision layer contains five distinct decisions with different feature subsets:

- D1: authentication, login risk, geo anomaly, IP trust
- D2: threat severity, failed logins, prior alerts, intel score
- D3: asset protection, criticality, threat score, prior alerts
- D4: escalation, threat severity, geo anomaly, criticality
- D5: response action, severity, criticality, failed logins, geo anomaly

All decisions are trained from clean data and their structure is preserved in the provenance graph.
