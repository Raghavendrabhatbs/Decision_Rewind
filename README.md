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
- LLM layer: abstraction with deterministic fallback explanations

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

The Dataset Workbench separates global model training from experiments. **Train Global Model** generates and persists a separate 20,000-record training dataset, then explicitly fits D1–D5 classifiers for 100 actual epochs. Each epoch reports training and validation loss/accuracy; the best validation-loss checkpoint for each decision is saved. Training is asynchronous and its progress, dataset ID, seed, metrics, model version, timestamp, and artifact checksum are persisted. Retraining creates a new version while retaining prior models and experiments.

Only after a global model has been trained can the user generate an exactly 200-record experiment. Generation stores feature records without D1–D5 decision labels. The user must explicitly click **Train Experiment** to label all 200 rows through inference with the frozen model version pinned to that experiment; this action does not fit or update model weights. The app stores the experiment's model version and training dataset ID so replay is reproducible. Counterfactual preview and rewind use that same frozen artifact and remain disabled until labeling succeeds.

The workbench supports searching, sorting, pagination (20 records per page), selectable feature/decision columns, event details, editable multi-feature proposals, one-pass counterfactual previews, persisted correction transactions, selective verified rewind, and dataset-scoped audit history. Correction and recovery are enabled only after an experiment has been generated from a successfully trained global model. Historical event data and historical decisions remain preserved after rewind, while current event/decision state is persisted separately.

## Model training

The decision models are trained from the clean dataset and intentionally keep five decision families distinct:

- D1: Authentication Decision
- D2: Threat Severity Decision
- D3: Asset Protection Decision
- D4: Incident Escalation Decision
- D5: Response Action Decision

Model metadata and outputs are stored in SQLite and versioned joblib artifacts under `backend/app/ml/trained/versions`; training datasets are persisted under `data/training/`. The API exposes `/api/models/status`, `/api/models/train`, and `/api/models/training/{job_id}` for model status, explicit training/retraining, and live epoch progress.

The provenance panel loads the active experiment's persisted feature → decision → output graph when the dataset opens; correction previews temporarily narrow the graph to the proposed feature paths. The AI chat accepts general questions with or without a selected experiment and uses the Groq Cloud chat-completions API (`https://api.groq.com/openai/v1/chat/completions`); the default model is `openai/gpt-oss-120b`. Configure `GROQ_API_KEY` in the project `.env` file; `LLM_MODEL`, `TEMPERATURE`, and `MAX_TOKENS` control generation. The `/api/ai/status` endpoint reports provider/model configuration without disclosing the key. If the network gateway blocks Groq requests, the API returns an explicit error rather than a fabricated answer.

## Running the application

Backend and frontend are independent but complementary:

- Backend provides the decision engine, dataset APIs, and verification logic
- Frontend renders decision cards, graph, and audit timeline

A complete demo flow is:

1. Open the dashboard
2. Select or edit SEC-000123
3. Modify a feature such as asset_criticality
4. Click Apply Correction
5. Click Run Decision Rewind
6. Review graph and AI explanation
7. Execute selective rewind
8. Inspect the verification and audit trail

## API documentation

The backend exposes endpoints including:

- GET /api/health
- GET /api/events
- GET /api/events/{event_id}
- POST /api/events/generate
- POST /api/corrections
- POST /api/rewind/analyze
- POST /api/rewind/execute
- GET /api/decisions/{event_id}
- GET /api/graph/{event_id}
- POST /api/ai/explain
- POST /api/ai/chat
- POST /api/experiments/run
- GET /api/audit
- GET /api/metrics

## LLM configuration

Set the environment variables in .env:

- LLM_PROVIDER=openai
- LLM_API_KEY=your-key

If no API key is configured, the UI falls back to a deterministic template explanation engine. This ensures the system still works without external dependencies.

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
