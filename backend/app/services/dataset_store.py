from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from backend.app.config import SQLITE_PATH

EDITABLE_FEATURES = {
    "asset_criticality",
    "failed_logins",
    "login_hour",
    "geo_anomaly",
    "threat_intel_score",
    "previous_alerts",
    "source_ip",
}
EVENT_COLUMNS = {
    "event_id",
    "timestamp",
    "user_id",
    "source_ip",
    "failed_logins",
    "login_hour",
    "geo_anomaly",
    "asset_id",
    "asset_type",
    "asset_criticality",
    "threat_intel_score",
    "previous_alerts",
}


class DatasetStore:
    def __init__(self, db_path: str | Path = SQLITE_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS datasets (
                    dataset_id TEXT PRIMARY KEY,
                    seed INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    record_count INTEGER NOT NULL,
                    is_active INTEGER NOT NULL DEFAULT 0,
                    training_status TEXT NOT NULL DEFAULT 'NOT_TRAINED',
                    model_id TEXT,
                    model_version TEXT,
                    training_record_count INTEGER,
                    trained_at TEXT,
                    model_path TEXT,
                    workflow_status TEXT NOT NULL DEFAULT 'WAITING_FOR_TRAINING'
                    , d5_status TEXT NOT NULL DEFAULT 'NOT_LABELED'
                );
                CREATE TABLE IF NOT EXISTS dataset_events (
                    dataset_id TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    original_payload TEXT NOT NULL,
                    current_payload TEXT NOT NULL,
                    PRIMARY KEY (dataset_id, event_id),
                    FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id)
                );
                CREATE TABLE IF NOT EXISTS dataset_decisions (
                    dataset_id TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    decision_id TEXT NOT NULL,
                    historical_output TEXT NOT NULL,
                    current_output TEXT NOT NULL,
                    counterfactual_output TEXT,
                    model_id TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    training_dataset_id TEXT NOT NULL,
                    PRIMARY KEY (dataset_id, event_id, decision_id),
                    FOREIGN KEY (dataset_id, event_id) REFERENCES dataset_events(dataset_id, event_id)
                );
                CREATE TABLE IF NOT EXISTS dataset_corrections (
                    correction_id TEXT PRIMARY KEY,
                    dataset_id TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    feature_name TEXT NOT NULL,
                    old_value TEXT NOT NULL,
                    new_value TEXT NOT NULL,
                    changes_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    user_action TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'PENDING'
                );
                CREATE TABLE IF NOT EXISTS rewind_operations (
                    rewind_id TEXT PRIMARY KEY,
                    correction_id TEXT NOT NULL,
                    dataset_id TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    decision_id TEXT NOT NULL,
                    old_output TEXT NOT NULL,
                    new_output TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS verification_results (
                    verification_id TEXT PRIMARY KEY,
                    rewind_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    details TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS dataset_audit_logs (
                    audit_id TEXT PRIMARY KEY,
                    dataset_id TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    details TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS provenance_edges (
                    dataset_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    target TEXT NOT NULL,
                    PRIMARY KEY (dataset_id, source, target)
                );
                CREATE TABLE IF NOT EXISTS model_training_jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    training_dataset_id TEXT NOT NULL,
                    training_seed INTEGER NOT NULL,
                    record_count INTEGER NOT NULL,
                    epochs INTEGER NOT NULL,
                    current_model TEXT,
                    current_epoch INTEGER NOT NULL DEFAULT 0,
                    metrics_json TEXT NOT NULL DEFAULT '{}',
                    error TEXT,
                    started_at TEXT NOT NULL,
                    completed_at TEXT
                );
                CREATE TABLE IF NOT EXISTS trained_models (
                    model_id TEXT PRIMARY KEY,
                    model_version TEXT NOT NULL UNIQUE,
                    training_dataset_id TEXT NOT NULL,
                    training_seed INTEGER NOT NULL,
                    training_record_count INTEGER NOT NULL,
                    epochs INTEGER NOT NULL,
                    best_epoch_json TEXT NOT NULL,
                    trained_at TEXT NOT NULL,
                    validation_metrics_json TEXT NOT NULL,
                    artifact_path TEXT NOT NULL,
                    artifact_sha256 TEXT NOT NULL,
                    status TEXT NOT NULL
                );
                """
            )
            dataset_columns = {row["name"] for row in conn.execute("PRAGMA table_info(datasets)")}
            for name, definition in {
                "training_status": "TEXT NOT NULL DEFAULT 'NOT_TRAINED'",
                "model_id": "TEXT",
                "model_version": "TEXT",
                "training_record_count": "INTEGER",
                "trained_at": "TEXT",
                "model_path": "TEXT",
                "workflow_status": "TEXT NOT NULL DEFAULT 'WAITING_FOR_TRAINING'",
                "training_dataset_id": "TEXT",
                "experiment_id": "TEXT",
                "d5_status": "TEXT NOT NULL DEFAULT 'NOT_LABELED'",
            }.items():
                if name not in dataset_columns:
                    conn.execute(f"ALTER TABLE datasets ADD COLUMN {name} {definition}")
            conn.execute(
                """
                UPDATE datasets SET d5_status = 'LABELED'
                WHERE training_status = 'TRAINED' AND d5_status = 'NOT_LABELED'
                    AND (SELECT COUNT(*) FROM dataset_decisions dd
                         WHERE dd.dataset_id = datasets.dataset_id) = record_count * 5
                """
            )
            correction_columns = {row["name"] for row in conn.execute("PRAGMA table_info(dataset_corrections)")}
            if "changes_json" not in correction_columns:
                conn.execute("ALTER TABLE dataset_corrections ADD COLUMN changes_json TEXT NOT NULL DEFAULT '[]'")

    def create_dataset(
        self,
        dataset_id: str,
        seed: int,
        events: List[Dict[str, Any]],
        model: Dict[str, Any],
    ) -> Dict[str, Any]:
        if len(events) != 200:
            raise ValueError(f"Interactive experiment datasets must contain exactly 200 records; received {len(events)}.")
        event_ids = [event["event_id"] for event in events]
        if len(set(event_ids)) != len(event_ids):
            raise ValueError("Dataset event IDs must be unique.")
        now = datetime.now(timezone.utc).isoformat()
        experiment_id = f"EXP-{dataset_id.removeprefix('DS-')}"
        with self._connect() as conn:
            conn.execute("UPDATE datasets SET is_active = 0")
            conn.execute(
                """
                INSERT INTO datasets (
                    dataset_id, seed, created_at, record_count, is_active,
                    training_status, workflow_status, model_id, model_version, training_record_count,
                    model_path, training_dataset_id, experiment_id, d5_status
                ) VALUES (?, ?, ?, ?, 1, 'NOT_TRAINED', 'WAITING_FOR_TRAINING', ?, ?, NULL, ?, ?, ?, 'NOT_LABELED')
                """,
                (
                    dataset_id, seed, now, len(events), model["model_id"], model["model_version"],
                    model["artifact_path"],
                    model["training_dataset_id"], experiment_id,
                ),
            )
            for event in events:
                generated_event = {key: value for key, value in event.items() if key != "decision_outputs"}
                generated_event["decision_outputs"] = {}
                payload = json.dumps(generated_event, sort_keys=True)
                conn.execute(
                    "INSERT INTO dataset_events (dataset_id, event_id, original_payload, current_payload) VALUES (?, ?, ?, ?)",
                    (dataset_id, generated_event["event_id"], payload, payload),
                )
            from backend.app.provenance.graph import FEATURE_TO_DECISIONS

            conn.executemany(
                "INSERT INTO provenance_edges (dataset_id, source, target) VALUES (?, ?, ?)",
                [(dataset_id, feature, decision) for feature, decisions in FEATURE_TO_DECISIONS.items() for decision in decisions],
            )
            conn.execute(
                "INSERT INTO dataset_audit_logs VALUES (?, ?, ?, ?, ?, ?)",
                (
                    f"{dataset_id}-created", dataset_id, "", "EXPERIMENT_GENERATED_UNLABELED",
                    json.dumps({
                        "experiment_id": experiment_id,
                        "seed": seed,
                        "record_count": len(events),
                        "model_id": model["model_id"],
                        "model_version": model["model_version"],
                        "training_dataset_id": model["training_dataset_id"],
                    }),
                    now,
                ),
            )
        return self.get_dataset(dataset_id) or {}

    def complete_experiment_training(
        self,
        dataset_id: str,
        predictions: Dict[str, Dict[str, str]],
    ) -> Dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            dataset = conn.execute("SELECT * FROM datasets WHERE dataset_id = ?", (dataset_id,)).fetchone()
            if not dataset:
                raise LookupError("Dataset not found.")
            if dataset["d5_status"] == "LABELED":
                raise ValueError("This experiment has already been labeled.")
            if not dataset["model_path"] or not dataset["model_id"] or not dataset["model_version"]:
                raise ValueError("This experiment has no pinned trained model.")
            events = conn.execute(
                "SELECT event_id, original_payload, current_payload FROM dataset_events WHERE dataset_id = ?",
                (dataset_id,),
            ).fetchall()
            if len(events) != 200:
                raise ValueError(f"Expected exactly 200 experiment records; found {len(events)}.")
            if set(predictions) != {row["event_id"] for row in events}:
                raise ValueError("Predictions must contain exactly one result for every experiment record.")
            for row in events:
                event_id = row["event_id"]
                outputs = predictions[event_id]
                if set(outputs) != {"D1", "D2", "D3", "D4", "D5"}:
                    raise ValueError(f"Predictions for {event_id} must include D1 through D5.")
                original = json.loads(row["original_payload"])
                current = json.loads(row["current_payload"])
                original["decision_outputs"] = outputs
                current["decision_outputs"] = outputs
                conn.execute(
                    "UPDATE dataset_events SET original_payload = ?, current_payload = ? WHERE dataset_id = ? AND event_id = ?",
                    (json.dumps(original, sort_keys=True), json.dumps(current, sort_keys=True), dataset_id, event_id),
                )
                for decision_id, output in outputs.items():
                    conn.execute(
                        """
                        INSERT INTO dataset_decisions (
                            dataset_id, event_id, decision_id, historical_output, current_output,
                            model_id, model_version, training_dataset_id
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(dataset_id, event_id, decision_id) DO UPDATE SET
                            historical_output = excluded.historical_output,
                            current_output = excluded.current_output,
                            counterfactual_output = NULL,
                            model_id = excluded.model_id,
                            model_version = excluded.model_version,
                            training_dataset_id = excluded.training_dataset_id
                        """,
                        (
                            dataset_id, event_id, decision_id, output, output,
                            dataset["model_id"], dataset["model_version"], dataset["training_dataset_id"],
                        ),
                    )
            conn.execute(
                """
                UPDATE datasets SET training_status = 'TRAINED', training_record_count = 200,
                    trained_at = ?, workflow_status = 'READY_FOR_CORRECTION', d5_status = 'LABELED'
                WHERE dataset_id = ?
                """,
                (now, dataset_id),
            )
            conn.execute(
                "INSERT INTO dataset_audit_logs VALUES (?, ?, ?, ?, ?, ?)",
                (
                    f"{dataset_id}-labeled-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}",
                    dataset_id,
                    "",
                    "EXPERIMENT_LABELED",
                    json.dumps({
                        "record_count": 200,
                        "model_id": dataset["model_id"],
                        "model_version": dataset["model_version"],
                        "training_dataset_id": dataset["training_dataset_id"],
                    }),
                    now,
                ),
            )
        return self.get_dataset(dataset_id) or {}

    def create_training_job(
        self,
        job_id: str,
        model_version: str,
        training_dataset_id: str,
        seed: int,
        record_count: int,
        epochs: int,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            if conn.execute(
                "SELECT 1 FROM model_training_jobs WHERE status = 'RUNNING' LIMIT 1"
            ).fetchone():
                raise ValueError("A model training job is already running.")
            conn.execute(
                """
                INSERT INTO model_training_jobs (
                    job_id, status, model_version, training_dataset_id, training_seed,
                    record_count, epochs, started_at
                ) VALUES (?, 'RUNNING', ?, ?, ?, ?, ?, ?)
                """,
                (job_id, model_version, training_dataset_id, seed, record_count, epochs, now),
            )

    def fail_interrupted_training_jobs(self) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE model_training_jobs
                SET status = 'FAILED', error = 'Training was interrupted by an application restart.',
                    completed_at = ?
                WHERE status = 'RUNNING'
                """,
                (now,),
            )

    def update_training_job(
        self,
        job_id: str,
        model_id: str,
        epoch: int,
        metrics: Dict[str, Any],
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE model_training_jobs
                SET current_model = ?, current_epoch = ?, metrics_json = ?
                WHERE job_id = ? AND status = 'RUNNING'
                """,
                (model_id, epoch, json.dumps(metrics, sort_keys=True), job_id),
            )

    def complete_training_job(self, job_id: str, model: Dict[str, Any]) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO trained_models (
                    model_id, model_version, training_dataset_id, training_seed,
                    training_record_count, epochs, best_epoch_json, trained_at,
                    validation_metrics_json, artifact_path, artifact_sha256, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'ACTIVE')
                """,
                (
                    model["model_id"], model["model_version"], model["training_dataset_id"],
                    model["training_seed"], model["training_record_count"], model["epochs"],
                    json.dumps(model["best_epoch"], sort_keys=True), model["training_timestamp"],
                    json.dumps(model["validation_metrics"], sort_keys=True), model["artifact_path"],
                    model["artifact_sha256"],
                ),
            )
            conn.execute(
                """
                UPDATE trained_models SET status = 'RETAINED'
                WHERE model_id != ? AND status = 'ACTIVE'
                """,
                (model["model_id"],),
            )
            conn.execute(
                """
                UPDATE model_training_jobs SET status = 'COMPLETED', current_model = NULL,
                    current_epoch = epochs, completed_at = ?
                WHERE job_id = ?
                """,
                (now, job_id),
            )

    def fail_training_job(self, job_id: str, error: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "UPDATE model_training_jobs SET status = 'FAILED', error = ?, completed_at = ? WHERE job_id = ?",
                (error, now, job_id),
            )

    def get_training_job(self, job_id: str) -> Dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM model_training_jobs WHERE job_id = ?", (job_id,)).fetchone()
        if not row:
            return None
        result = dict(row)
        result["metrics"] = json.loads(result.pop("metrics_json"))
        return result

    def get_latest_training_job(self) -> Dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT job_id FROM model_training_jobs ORDER BY started_at DESC LIMIT 1"
            ).fetchone()
        return self.get_training_job(row["job_id"]) if row else None

    def has_running_training_job(self) -> bool:
        with self._connect() as conn:
            return conn.execute(
                "SELECT 1 FROM model_training_jobs WHERE status = 'RUNNING' LIMIT 1"
            ).fetchone() is not None

    def get_active_model(self) -> Dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM trained_models WHERE status = 'ACTIVE' ORDER BY trained_at DESC LIMIT 1"
            ).fetchone()
        return self._public_model(row) if row else None

    def list_models(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM trained_models ORDER BY trained_at DESC").fetchall()
        return [self._public_model(row) for row in rows]

    @staticmethod
    def _public_model(row: sqlite3.Row) -> Dict[str, Any]:
        model = dict(row)
        model["best_epoch"] = json.loads(model.pop("best_epoch_json"))
        model["validation_metrics"] = json.loads(model.pop("validation_metrics_json"))
        return model

    def next_model_version(self) -> str:
        with self._connect() as conn:
            count = conn.execute("SELECT COUNT(*) FROM trained_models").fetchone()[0]
        return f"V{count + 1}"

    def get_active_dataset(self) -> Dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT d.* FROM datasets d
                JOIN trained_models m ON m.model_id = d.model_id
                WHERE d.is_active = 1
                """
            ).fetchone()
        return self._public_dataset(row) if row else None

    def set_workflow_status(self, dataset_id: str, status: str) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE datasets SET workflow_status = ? WHERE dataset_id = ?", (status, dataset_id))

    def cancel_correction(self, correction_id: str) -> Dict[str, Any] | None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            correction = conn.execute(
                "SELECT dataset_id, event_id, status FROM dataset_corrections WHERE correction_id = ?",
                (correction_id,),
            ).fetchone()
            if not correction:
                return None
            if correction["status"] != "PENDING":
                raise ValueError("Only pending correction transactions can be cancelled.")
            conn.execute("UPDATE dataset_corrections SET status = 'CANCELLED' WHERE correction_id = ?", (correction_id,))
            conn.execute(
                "INSERT INTO dataset_audit_logs VALUES (?, ?, ?, ?, ?, ?)",
                (
                    f"{correction_id}-cancelled",
                    correction["dataset_id"],
                    correction["event_id"],
                    "CORRECTION_CANCELLED",
                    json.dumps({"correction_id": correction_id}),
                    now,
                ),
            )
            conn.execute(
                "UPDATE datasets SET workflow_status = 'READY_FOR_CORRECTION' WHERE dataset_id = ?",
                (correction["dataset_id"],),
            )
        return {"correction_id": correction_id, "status": "CANCELLED"}

    def log_workflow_transition(self, dataset_id: str, event_id: str, operation: str, details: Dict[str, Any]) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        audit_id = f"{dataset_id}-{operation}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO dataset_audit_logs VALUES (?, ?, ?, ?, ?, ?)",
                (audit_id, dataset_id, event_id, operation, json.dumps(details, sort_keys=True), timestamp),
            )
            conn.execute("UPDATE datasets SET workflow_status = ? WHERE dataset_id = ?", (operation, dataset_id))

    def get_dataset(self, dataset_id: str) -> Dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM datasets WHERE dataset_id = ?", (dataset_id,)).fetchone()
        return self._public_dataset(row) if row else None

    @staticmethod
    def _public_dataset(row: sqlite3.Row) -> Dict[str, Any]:
        dataset = dict(row)
        dataset["label_status"] = dataset.get("d5_status", "NOT_LABELED")
        dataset["training_status"] = (
            "TRAINED" if dataset["label_status"] == "LABELED" else "NOT_TRAINED"
        )
        if dataset["label_status"] != "LABELED":
            dataset["workflow_status"] = "WAITING_FOR_TRAINING"
        dataset["generation_timestamp"] = dataset["created_at"]
        dataset["random_seed"] = dataset["seed"]
        return dataset

    def list_datasets(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM datasets ORDER BY created_at DESC").fetchall()
        return [self._public_dataset(row) for row in rows]

    def get_training_events(self, dataset_id: str) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT original_payload FROM dataset_events WHERE dataset_id = ? ORDER BY event_id",
                (dataset_id,),
            ).fetchall()
        return [json.loads(row["original_payload"]) for row in rows]

    def has_decisions(self, dataset_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM dataset_decisions WHERE dataset_id = ? LIMIT 1",
                (dataset_id,),
            ).fetchone()
        return row is not None

    def list_events(self, dataset_id: str, search: str, sort_by: str, descending: bool, page: int, page_size: int) -> Dict[str, Any]:
        column = sort_by if sort_by in EVENT_COLUMNS else "event_id"
        with self._connect() as conn:
            dataset = conn.execute(
                "SELECT d5_status FROM datasets WHERE dataset_id = ?", (dataset_id,)
            ).fetchone()
            labels_available = bool(dataset and dataset["d5_status"] == "LABELED")
            rows = conn.execute(
                "SELECT event_id, current_payload FROM dataset_events WHERE dataset_id = ?",
                (dataset_id,),
            ).fetchall()
            statuses = {
                row["event_id"]: (bool(row["rewound"]), bool(row["pending"]))
                for row in conn.execute(
                    "SELECT event_id, MAX(status = 'REWOUND') AS rewound, MAX(status = 'PENDING') AS pending "
                    "FROM dataset_corrections WHERE dataset_id = ? GROUP BY event_id",
                    (dataset_id,),
                ).fetchall()
            }
            impact_by_event = {
                row["event_id"]: row["affected"]
                for row in conn.execute(
                    "SELECT event_id, MAX(current_output != historical_output) AS affected "
                    "FROM dataset_decisions WHERE dataset_id = ? GROUP BY event_id",
                    (dataset_id,),
                ).fetchall()
            }
        events = []
        needle = search.strip().lower()
        for row in rows:
            event = json.loads(row["current_payload"])
            if not labels_available:
                event["decision_outputs"] = {}
            if needle and not any(needle in str(value).lower() for value in event.values()):
                continue
            corrected, pending = statuses.get(row["event_id"], (False, False))
            event["correction_status"] = "CORRECTED" if corrected else ("PROPOSED" if pending else "UNCORRECTED")
            event["impact_status"] = (
                ("AFFECTED" if impact_by_event.get(row["event_id"], 0) else "UNAFFECTED")
                if corrected
                else "UNASSESSED"
            )
            events.append(event)
        events.sort(key=lambda item: (item.get(column) is None, item.get(column)), reverse=descending)
        total = len(events)
        start = (page - 1) * page_size
        return {"items": events[start : start + page_size], "total": total, "page": page, "page_size": page_size}

    def get_event(self, dataset_id: str, event_id: str) -> Dict[str, Any] | None:
        with self._connect() as conn:
            dataset_row = conn.execute(
                "SELECT d5_status FROM datasets WHERE dataset_id = ?", (dataset_id,)
            ).fetchone()
            event_row = conn.execute(
                "SELECT original_payload, current_payload FROM dataset_events WHERE dataset_id = ? AND event_id = ?",
                (dataset_id, event_id),
            ).fetchone()
            if not event_row:
                return None
            labels_available = bool(dataset_row and dataset_row["d5_status"] == "LABELED")
            decisions = (
                conn.execute(
                    "SELECT * FROM dataset_decisions WHERE dataset_id = ? AND event_id = ? ORDER BY decision_id",
                    (dataset_id, event_id),
                ).fetchall()
                if labels_available
                else []
            )
            corrections = conn.execute(
                "SELECT * FROM dataset_corrections WHERE dataset_id = ? AND event_id = ? ORDER BY created_at DESC",
                (dataset_id, event_id),
            ).fetchall()
        original_state = json.loads(event_row["original_payload"])
        current_state = json.loads(event_row["current_payload"])
        if not labels_available:
            original_state["decision_outputs"] = {}
            current_state["decision_outputs"] = {}
        return {
            "original_state": original_state,
            "current_state": current_state,
            "decisions": [dict(row) for row in decisions],
            "corrections": [dict(row) for row in corrections],
        }

    def create_correction(self, dataset_id: str, event_id: str, changes: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not changes:
            raise ValueError("At least one changed feature is required.")
        with self._connect() as conn:
            dataset = conn.execute("SELECT d5_status FROM datasets WHERE dataset_id = ?", (dataset_id,)).fetchone()
            row = conn.execute(
                "SELECT current_payload FROM dataset_events WHERE dataset_id = ? AND event_id = ?",
                (dataset_id, event_id),
            ).fetchone()
            if not dataset:
                raise LookupError("Dataset not found.")
            if not row:
                raise LookupError("Event not found in this dataset.")
            if dataset["d5_status"] != "LABELED":
                raise ValueError("Train the dataset before applying corrections.")
            pending = conn.execute(
                "SELECT 1 FROM dataset_corrections WHERE dataset_id = ? AND event_id = ? AND status = 'PENDING'",
                (dataset_id, event_id),
            ).fetchone()
            if pending:
                raise ValueError("This event already has an unapplied correction transaction.")
            current = json.loads(row["current_payload"])
            normalized_changes = []
            for change in changes:
                feature = change["feature"]
                if feature not in EDITABLE_FEATURES:
                    raise ValueError(f"Feature {feature} is not editable.")
                if current[feature] != change["new_value"]:
                    normalized_changes.append({
                        "feature": feature,
                        "old_value": current[feature],
                        "new_value": change["new_value"],
                    })
            if not normalized_changes:
                raise ValueError("Proposed values must differ from the current feature values.")
            correction_id = f"COR-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"
            created_at = datetime.now(timezone.utc).isoformat()
            conn.execute(
                """
                INSERT INTO dataset_corrections (
                    correction_id, dataset_id, event_id, feature_name, old_value,
                    new_value, changes_json, created_at, user_action, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING')
                """,
                (
                    correction_id,
                    dataset_id,
                    event_id,
                    "__multi__",
                    json.dumps({item["feature"]: item["old_value"] for item in normalized_changes}, sort_keys=True),
                    json.dumps({item["feature"]: item["new_value"] for item in normalized_changes}, sort_keys=True),
                    json.dumps(normalized_changes, sort_keys=True),
                    created_at,
                    "User correction transaction",
                ),
            )
            conn.execute(
                "INSERT INTO dataset_audit_logs VALUES (?, ?, ?, ?, ?, ?)",
                (f"{correction_id}-audit", dataset_id, event_id, "CORRECTION_PROPOSED", json.dumps({"correction_id": correction_id, "changes": normalized_changes}), created_at),
            )
            conn.execute("UPDATE datasets SET workflow_status = 'CORRECTION_APPLIED' WHERE dataset_id = ?", (dataset_id,))
            return {"correction_id": correction_id, "dataset_id": dataset_id, "event_id": event_id, "changes": normalized_changes, "created_at": created_at, "status": "PENDING"}

    def get_correction(self, correction_id: str) -> Dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM dataset_corrections WHERE correction_id = ?", (correction_id,)).fetchone()
        if not row:
            return None
        result = dict(row)
        result["old_value"] = json.loads(result["old_value"])
        result["new_value"] = json.loads(result["new_value"])
        result["changes"] = json.loads(result["changes_json"])
        return result

    def complete_rewind(
        self,
        correction: Dict[str, Any],
        counterfactual: Dict[str, str],
        affected: List[str],
        decision_impacts: Dict[str, Dict[str, Any]],
        verification: Dict[str, Any],
    ) -> Dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        rewind_results = []
        with self._connect() as conn:
            event_row = conn.execute(
                "SELECT current_payload FROM dataset_events WHERE dataset_id = ? AND event_id = ?",
                (correction["dataset_id"], correction["event_id"]),
            ).fetchone()
            if not event_row:
                raise LookupError("Event not found in this dataset.")
            event = json.loads(event_row["current_payload"])
            for change in correction["changes"]:
                if event.get(change["feature"]) != change["old_value"]:
                    raise ValueError("The event changed after this correction was proposed. Create a fresh correction against the current value.")
                event[change["feature"]] = change["new_value"]
            event["decision_outputs"] = dict(event["decision_outputs"])
            for decision in conn.execute(
                "SELECT decision_id, current_output FROM dataset_decisions WHERE dataset_id = ? AND event_id = ?",
                (correction["dataset_id"], correction["event_id"]),
            ).fetchall():
                decision_id = decision["decision_id"]
                proposed = counterfactual[decision_id]
                if decision_id in affected:
                    rewind_id = f"RW-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}-{decision_id}"
                    conn.execute(
                        "UPDATE dataset_decisions SET current_output = ?, counterfactual_output = ? WHERE dataset_id = ? AND event_id = ? AND decision_id = ?",
                        (proposed, proposed, correction["dataset_id"], correction["event_id"], decision_id),
                    )
                    conn.execute(
                        "INSERT INTO rewind_operations VALUES (?, ?, ?, ?, ?, ?, ?, 'VERIFIED', ?)",
                        (rewind_id, correction["correction_id"], correction["dataset_id"], correction["event_id"], decision_id, decision["current_output"], proposed, now),
                    )
                    conn.execute(
                        "INSERT INTO verification_results VALUES (?, ?, ?, ?, ?)",
                        (f"VR-{rewind_id}", rewind_id, verification["verification"], json.dumps(verification), now),
                    )
                    rewind_results.append({"rewind_id": rewind_id, "decision_id": decision_id, "old_output": decision["current_output"], "new_output": proposed, "status": "VERIFIED"})
                    event["decision_outputs"][decision_id] = proposed
                conn.execute(
                    "UPDATE dataset_decisions SET counterfactual_output = ? WHERE dataset_id = ? AND event_id = ? AND decision_id = ?",
                    (proposed, correction["dataset_id"], correction["event_id"], decision_id),
                )
            conn.execute(
                "UPDATE dataset_events SET current_payload = ? WHERE dataset_id = ? AND event_id = ?",
                (json.dumps(event, sort_keys=True), correction["dataset_id"], correction["event_id"]),
            )
            conn.execute("UPDATE dataset_corrections SET status = 'REWOUND' WHERE correction_id = ?", (correction["correction_id"],))
            conn.execute(
                "INSERT INTO dataset_audit_logs VALUES (?, ?, ?, ?, ?, ?)",
                (f"{correction['correction_id']}-rewind-audit", correction["dataset_id"], correction["event_id"], "COUNTERFACTUAL_REPLAYED", json.dumps({"correction_id": correction["correction_id"], "decision_impacts": decision_impacts}), now),
            )
            conn.execute(
                "INSERT INTO dataset_audit_logs VALUES (?, ?, ?, ?, ?, ?)",
                (f"{correction['correction_id']}-recovery-ready", correction["dataset_id"], correction["event_id"], "RECOVERY_READY", json.dumps({"affected_decisions": affected}), now),
            )
            conn.execute(
                "INSERT INTO dataset_audit_logs VALUES (?, ?, ?, ?, ?, ?)",
                (f"{correction['correction_id']}-recovered", correction["dataset_id"], correction["event_id"], "RECOVERED", json.dumps({"rewind_operations": rewind_results}), now),
            )
            conn.execute(
                "INSERT INTO dataset_audit_logs VALUES (?, ?, ?, ?, ?, ?)",
                (f"{correction['correction_id']}-verified", correction["dataset_id"], correction["event_id"], "VERIFIED", json.dumps(verification), now),
            )
            conn.execute(
                "INSERT INTO verification_results VALUES (?, ?, ?, ?, ?)",
                (
                    f"VR-{correction['correction_id']}-TRANSACTION",
                    f"RW-{correction['correction_id']}-TRANSACTION",
                    verification["verification"],
                    json.dumps(verification, sort_keys=True),
                    now,
                ),
            )
            conn.execute("UPDATE datasets SET workflow_status = 'VERIFIED' WHERE dataset_id = ?", (correction["dataset_id"],))
        return {"rewind_operations": rewind_results, "current_state": event, "verification": verification}

    def list_audit(self, dataset_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM dataset_audit_logs WHERE dataset_id = ? ORDER BY timestamp DESC LIMIT ?",
                (dataset_id, limit),
            ).fetchall()
        return [
            {**dict(row), "details": json.loads(row["details"])}
            for row in rows
        ]

    def get_graph(self, dataset_id: str, features: List[str] | None = None) -> Dict[str, List[Dict[str, str]]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT source, target FROM provenance_edges WHERE dataset_id = ?",
                (dataset_id,),
            ).fetchall()
        selected = set(features or [])
        feature_edges = [
            {"source": row["source"], "target": row["target"]}
            for row in rows
            if not selected or row["source"] in selected
        ]
        feature_nodes = {edge["source"] for edge in feature_edges}
        decision_nodes = {edge["target"] for edge in feature_edges}
        decision_edges = []
        if "D2" in decision_nodes:
            for downstream in ("D4", "D5"):
                decision_edges.append({"source": "D2", "target": downstream})
                decision_nodes.add(downstream)
        output_edges = [
            {"source": decision, "target": f"{decision}-output"}
            for decision in sorted(decision_nodes)
        ]
        nodes = [
            *({"id": feature, "kind": "feature"} for feature in sorted(feature_nodes)),
            *({"id": decision, "kind": "decision"} for decision in sorted(decision_nodes)),
            *({"id": f"{decision}-output", "kind": "output"} for decision in sorted(decision_nodes)),
        ]
        return {"nodes": nodes, "edges": feature_edges + decision_edges + output_edges}


dataset_store = DatasetStore()
