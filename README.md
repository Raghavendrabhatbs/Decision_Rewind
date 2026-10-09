# DECISION-REWIND

DECISION-REWIND is a deterministic cybersecurity research prototype for identifying downstream decisions impacted by a corrected upstream data value. It models the pipeline from corrupted historical data to corrected facts, counterfactual replay, selective rewind, and verification.

## Project overview

The application demonstrates the following research flow:

- corrupt a historical security feature
- evaluate downstream decision functions
- compare historical and counterfactual outcomes
- identify affected decisions based on dependency reachability
- rewind only affected decisions
- verify the recovery deterministically
- provide audited AI explanation backed by evidence

The application keeps the AI layer separate from the recovery engine. LLM responses are explanatory only; they never decide final recovery actions.

## Architecture

- Backend: FastAPI + Pydantic + scikit-learn + NetworkX + SQLite persistence
- Frontend: React + Vite + TypeScript + Recharts + React Flow
- Data layer: synthetic cybersecurity datasets generated deterministically with a seed
- LLM layer: optional Groq-backed explanations grounded in persisted application evidence; unavailable when no API key is configured

## Repository layout

- backend/app: FastAPI service, ML, provenance, recovery, verification, llm
- backend/tests: backend validation tests
- frontend: standalone React dashboard
- data/: generated and persisted synthetic datasets
- docs/: research and architecture notes

## Backend setup

1. Use Python 3.11+
2. Install requirements:

   py -3.11 -m pip install -r backend/requirements.txt

3. Create a local environment file:

   copy .env.example .env

4. Start the API:

   py -3.11 -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000

## Frontend setup

1. Install Node dependencies:

   cd frontend
   npm install

2. Start the dashboard:

   npm run dev -- --host 0.0.0.0

3. Open http://localhost:5173

## Dataset generation

The dataset generator lives at the project root and generates 20,000 synthetic events by default:

- generate_dataset.py

To regenerate data:

   py generate_dataset.py

This produces a deterministic synthetic cybersecurity dataset of 20,000 events, plus demo event SEC-000123 and the final clean/corrupted outputs. This standalone generation command does not train models.

The Dataset Workbench separates global model training from experiments. **Train Global Model** generates and persists a separate 20,000-record training dataset, then fits one classifier for each decision using an 80/20 stratified train/validation split. Logistic Regression is used for D1 and D4, Random Forest for D2 and D5, and Decision Tree for D3. Each fit reports training and validation loss/accuracy. Training is asynchronous; its progress, dataset ID, seed, metrics, model version, timestamp, and artifact checksum are persisted. Retraining creates a new version while retaining prior models and experiments.

Only after a global model has been trained can the user generate an exactly 200-record experiment. Generation stores feature records without D1–D5 decision labels. The user must explicitly click **Train Experiment** to label all 200 rows through inference with the frozen model version pinned to that experiment; this action does not fit or update model weights. The app stores the experiment's model version and training dataset ID so replay is reproducible. Counterfactual preview and rewind use that same frozen artifact and remain disabled until labeling succeeds.

The workbench supports searching, sorting, pagination (20 records per page), selectable feature/decision columns, event details, editable multi-feature proposals, one-pass counterfactual previews, persisted correction transactions, selective verified rewind, and dataset-scoped audit history. Correction and recovery are enabled only after an experiment has been generated from a successfully trained global model. Historical event data and historical decisions remain preserved after rewind, while current event/decision state is persisted separately.

## Model training

The workbench creates a 20,000-record synthetic training dataset and fits five separate classifiers using an 80/20 stratified train/validation split. Each estimator is fitted once per model version:

| Decision | Purpose | Classifier |
|---|---|---|
| D1 | Authentication | Logistic Regression |
| D2 | Threat Severity | Random Forest |
| D3 | Asset Protection | Decision Tree |
| D4 | Incident Escalation | Logistic Regression |
| D5 | Response Action | Random Forest |

Training and validation loss and accuracy are recorded for each fit. Model metadata, the training seed and dataset ID, validation metrics, and a SHA-256 checksum are stored with versioned joblib artifacts under `backend/app/ml/trained/versions`; training datasets are persisted under `data/training/`. The API exposes `/api/models/status`, `/api/models/train`, and `/api/models/training/{job_id}` for model status, explicit training/retraining, and per-model fit progress.

For a separate held-out evaluation report, run this from the repository root:

```powershell
py -3.11 -m backend.app.ml.train
```

The command trains the same five classifier types on a fresh deterministic 20,000-record dataset with an 80/20 stratified split. It reports training accuracy, held-out accuracy, weighted precision, recall, F1, and confusion matrices for each decision. The JSON report is written to `backend/app/ml/trained/metrics/latest_training_metrics.json`, a local generated artifact excluded from Git. This standalone evaluation does not replace the active workbench model or its pinned artifacts. See [docs/decision-models.md](docs/decision-models.md) for feature sets and replay details.

The provenance panel loads the active experiment's persisted feature → decision → output graph when the dataset opens; correction previews temporarily narrow the graph to the proposed feature paths. The AI chat accepts general questions with or without a selected experiment. The backend builds evidence from persisted experiment/event/decision state, correction and recovery audits, explicit provenance paths, and sanitized Universal Logs filtered by question and relevant identifiers. Log context includes whole-history counts, relevant and recent events in chronology, and bounded details; the assistant is told when history is partial. The frontend displays evidence categories and sanitized log entries returned with the answer, and renders headings and lists as readable text. Persisted state and deterministic verification remain authoritative; AI can explain but cannot approve or execute recovery. The provider uses the OpenAI SDK Responses API with Groq's OpenAI-compatible endpoint. Copy `.env.example` to `.env`, then set `GROQ_API_KEY` to your Groq API key. `LLM_BASE_URL` defaults to `https://api.groq.com/openai/v1`; `LLM_MODEL` defaults to `openai/gpt-oss-20b`. `TEMPERATURE` and `MAX_TOKENS` control generation. The `/api/ai/status` endpoint reports provider/model configuration without disclosing the key. If the key is absent or the provider request fails, the application reports the error rather than returning a fabricated fallback answer.

## Running the application

Backend and frontend are independent but complementary:

- Backend provides the decision engine, dataset APIs, and verification logic
- Frontend renders decision cards, graph, and audit timeline

A complete demo flow is:

1. Train or retrain the global model.
2. Generate a 200-record experiment and train it with its pinned frozen model.
3. Select an event and propose one or more feature corrections.
4. Preview counterfactual decisions and their provenance paths.
5. Apply the correction, then run selective rewind.
6. Review deterministic verification, the audit trail, and the AI explanation.

## API documentation

The canonical workbench API includes:

- GET /api/health
- GET /api/models/status
- POST /api/models/train
- GET /api/models/training/{job_id}
- POST /api/datasets
- POST /api/datasets/{dataset_id}/train
- GET /api/datasets/{dataset_id}/events
- GET /api/datasets/{dataset_id}/events/{event_id}
- POST /api/datasets/{dataset_id}/preview
- POST /api/datasets/{dataset_id}/corrections
- POST /api/datasets/{dataset_id}/rewind
- POST /api/ai/chat
- GET /api/datasets/{dataset_id}/audit
- GET /api/metrics

Legacy event, rule-replay, correction, recovery, explanation, and synthetic-metrics endpoints return HTTP 410 and direct callers to the canonical workbench instead of modifying state or fabricating results. `/api/metrics` reports persisted training metrics and actual verification status; unavailable results are reported as `NOT_AVAILABLE`.

## LLM configuration

Copy `.env.example` to `.env` and set the environment variables:

- LLM_PROVIDER=groq
- LLM_BASE_URL=https://api.groq.com/openai/v1
- GROQ_API_KEY=your-groq-api-key
- LLM_MODEL=openai/gpt-oss-20b

Do not commit `.env` or share the API key. If no API key is configured, AI chat reports that it is unavailable; other application functionality remains available.

## Experiment methodology

Experiments generate synthetic faults and compare measured metrics such as:

- affected-decision precision/recall
- counterfactual accuracy
- recovery success
- recovery cost
- latency
- collateral impact
- residual decision damage
- verification failure rate

These metrics are generated from actual pipeline execution, not fabricated values.

## Research contribution

This prototype focuses on the research question: when a data correction alters historical decision context, which decisions are actually invalidated, and which can be safely rewound without collateral changes?

The system contributes a disciplined separation between:

- ML prediction
- provenance analysis
- counterfactual replay
- deterministic verification
- audit trail generation
- AI explanation only as evidence-backed assistance

## Limitations

This is a research-grade prototype intended for controlled synthetic evaluation. It does not connect to real SOC infrastructure, production firewalls, or live telemetry systems. It remains an explicit simulation environment.
