# Decision Rewind — AI Reasoning Fix

## What changed

The AI chat path now uses the same `dataset_store` as the Decision Rewind workbench instead of the legacy event store. When a dataset and event are selected, `/api/ai/chat` collects:

- historical and current event state
- dataset/model/version metadata
- proposed correction
- counterfactual outputs from the existing trained model artifact
- historical/current/counterfactual decision comparisons
- deterministic affected/unaffected decisions
- explicit provenance paths
- provenance graph data
- universal-log evidence scoped to the question, dataset, event, changed features, and affected decisions

The LLM is explicitly instructed to explain deterministic results rather than inventing or deciding rewind actions.

## Counterfactual rule

Counterfactual replay does **not** retrain models. It re-executes the existing trained model version against corrected historical input. D2 is evaluated before D4/D5 because D4/D5 consume D2 severity.

## Main files changed

- `backend/app/main.py` — dataset-backed AI evidence assembly and `/api/ai/chat`
- `backend/app/llm/provider.py` — evidence-grounded Groq reasoning prompt
- `backend/app/provenance/graph.py` — explicit feature-to-decision paths
- `backend/app/schemas/models.py` — optional `dataset_id` for AI requests
- `frontend/src/App.tsx` — sends dataset ID and counterfactual/provenance context
- `backend/app/counterfactual/replay.py` — legacy replay now re-executes the supplied model registry when available
- `backend/app/ml/model_registry.py` — compatibility with newer scikit-learn
- `backend/app/config.py` — consistent LLM default
- `.env.example` / `docker-compose.yml` — no embedded API key
- `docs/llm-role.md` — updated AI role and evidence contract

## API key

The previously embedded Groq key was removed. Treat the old key as compromised and rotate it. Put a new key in the local `.env` file:

`GROQ_API_KEY=your-new-key`

Do not commit `.env`.

## Verification

The existing backend test suite passes: **33 passed** in the build environment. A focused API smoke test also confirmed that `/api/ai/chat` enters `decision_rewind_evidence` mode and builds counterfactual and provenance evidence before invoking the LLM.


## HTTP 413 protection

The LLM provider now bounds the evidence sent to Groq. Universal-log context is capped before the LLM request, provenance graphs are previewed when oversized, and the provider retains the high-value counterfactual, impact, provenance, and log evidence. This prevents large application logs from causing Groq HTTP 413 request-too-large errors.
