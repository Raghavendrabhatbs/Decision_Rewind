# LLM Role

The LLM is an evidence-grounded explanation layer, not the authority for
decision state or recovery.

For an experiment/event, the backend builds context from persisted historical
and current state, the model version pinned to the experiment, proposed
corrections, counterfactual outputs, affected decisions, explicit provenance
paths, recovery and verification audits, and filtered Universal Logs.

Persisted state, frozen-model replay, and deterministic verification are
authoritative. Provenance indicates possible dependency, not proof that an
output changed. Universal Logs are supporting temporal evidence and may be
partial. The LLM can explain evidence but cannot approve or execute recovery,
and must not invent missing state or logs. If no API key is configured, the
application reports that AI is unavailable; it does not fabricate a fallback
answer.
