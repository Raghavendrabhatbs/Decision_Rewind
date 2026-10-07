import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from backend.app.config import SQLITE_PATH


class SQLiteStore:
    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path) if db_path is not None else SQLITE_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS corrections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL,
                    feature TEXT NOT NULL,
                    old_value TEXT NOT NULL,
                    new_value TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    changed_feature TEXT NOT NULL,
                    old_value TEXT NOT NULL,
                    new_value TEXT NOT NULL,
                    affected_decisions TEXT NOT NULL,
                    historical_outputs TEXT NOT NULL,
                    counterfactual_outputs TEXT NOT NULL,
                    recovery_results TEXT NOT NULL,
                    verification_results TEXT NOT NULL,
                    model_versions TEXT NOT NULL,
                    llm_explanation TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def save_event(self, event: Dict[str, Any]) -> None:
        payload = json.dumps(event, sort_keys=True)
        created_at = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO events (event_id, payload, created_at) VALUES (?, ?, ?)",
                (event["event_id"], payload, created_at),
            )
            conn.commit()

    def get_event(self, event_id: str) -> Dict[str, Any] | None:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT payload FROM events WHERE event_id = ?",
                (event_id,),
            ).fetchone()
        if not row:
            return None
        return json.loads(row[0])

    def list_events(self) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT payload FROM events ORDER BY event_id").fetchall()
        return [json.loads(row[0]) for row in rows]

    def log_audit(self, record: Dict[str, Any]) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO audit (
                    timestamp, event_id, changed_feature, old_value, new_value,
                    affected_decisions, historical_outputs, counterfactual_outputs,
                    recovery_results, verification_results, model_versions, llm_explanation
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record["timestamp"],
                    record["event_id"],
                    record["changed_feature"],
                    json.dumps(record["old_value"], sort_keys=True),
                    json.dumps(record["new_value"], sort_keys=True),
                    json.dumps(record["affected_decisions"], sort_keys=True),
                    json.dumps(record["historical_outputs"], sort_keys=True),
                    json.dumps(record["counterfactual_outputs"], sort_keys=True),
                    json.dumps(record["recovery_results"], sort_keys=True),
                    json.dumps(record["verification_results"], sort_keys=True),
                    json.dumps(record["model_versions"], sort_keys=True),
                    record.get("llm_explanation", ""),
                ),
            )
            conn.commit()

    def list_audit(self) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM audit ORDER BY timestamp DESC"
            ).fetchall()
        output = []
        for row in rows:
            output.append(
                {
                    "id": row[0],
                    "timestamp": row[1],
                    "event_id": row[2],
                    "changed_feature": row[3],
                    "old_value": json.loads(row[4]),
                    "new_value": json.loads(row[5]),
                    "affected_decisions": json.loads(row[6]),
                    "historical_outputs": json.loads(row[7]),
                    "counterfactual_outputs": json.loads(row[8]),
                    "recovery_results": json.loads(row[9]),
                    "verification_results": json.loads(row[10]),
                    "model_versions": json.loads(row[11]),
                    "llm_explanation": row[12],
                }
            )
        return output


store = SQLiteStore()
