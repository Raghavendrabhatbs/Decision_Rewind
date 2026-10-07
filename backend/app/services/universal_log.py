from __future__ import annotations

import json
import logging
import re
import threading
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from backend.app.config import DATA_DIR

_SENSITIVE_KEY = re.compile(r"(api.?key|authorization|secret|password|token|credential|cookie)", re.IGNORECASE)
_SENSITIVE_TEXT = re.compile(r"(?i)gsk_[a-z0-9_-]+|bearer\s+[^\s\"']+")


def _sanitize(value: Any, key: str = "") -> Any:
    if _SENSITIVE_KEY.search(key):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {str(child_key): _sanitize(child, str(child_key)) for child_key, child in value.items()}
    if isinstance(value, list):
        return [_sanitize(child) for child in value]
    if isinstance(value, str):
        return _SENSITIVE_TEXT.sub("[REDACTED]", value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)


def _decode_payload(body: bytes) -> Any:
    if not body:
        return None
    text = body.decode("utf-8", "replace")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return _sanitize(text)
    return _sanitize(payload)


class UniversalLog:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path is not None else DATA_DIR / "universal_log.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        self._lock = threading.Lock()

    def record(self, event_type: str, source: str, details: Dict[str, Any]) -> Dict[str, Any]:
        event = {
            "id": uuid.uuid4().hex,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "source": source,
            "details": _sanitize(details),
        }
        encoded = json.dumps(event, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
        with self._lock:
            with self.path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(encoded + "\n")
                stream.flush()
        return event

    def record_many(self, event_type: str, source: str, details: list[Dict[str, Any]]) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        with self._lock:
            with self.path.open("a", encoding="utf-8", newline="\n") as stream:
                for item in details:
                    event = {
                        "id": uuid.uuid4().hex,
                        "timestamp": timestamp,
                        "event_type": event_type,
                        "source": source,
                        "details": _sanitize(item),
                    }
                    stream.write(json.dumps(event, ensure_ascii=True, separators=(",", ":"), allow_nan=False) + "\n")
                stream.flush()

    def recent(self, limit: int = 12) -> list[Dict[str, Any]]:
        if limit <= 0 or not self.path.exists():
            return []
        remaining = limit + 1
        buffer = b""
        with self._lock:
            with self.path.open("rb") as stream:
                stream.seek(0, 2)
                position = stream.tell()
                while position > 0 and buffer.count(b"\n") <= limit:
                    chunk_size = min(8192, position)
                    position -= chunk_size
                    stream.seek(position)
                    buffer = stream.read(chunk_size) + buffer
        lines = buffer.splitlines()
        if position > 0 and lines:
            lines = lines[1:]
        events = deque(
            (json.loads(line.decode("utf-8")) for line in lines if line.strip()),
            maxlen=remaining,
        )
        return list(events)[-limit:]

    def recent_for_ai(self, limit: int = 12, detail_limit: int = 1800) -> list[Dict[str, Any]]:
        events = self.recent(limit)
        for event in events:
            details = event.get("details", {})
            for key, value in list(details.items()):
                encoded = json.dumps(value, ensure_ascii=True, separators=(",", ":"))
                if len(encoded) > detail_limit:
                    details[key] = {
                        "truncated": True,
                        "preview": encoded[:detail_limit],
                    }
        return events


universal_log = UniversalLog()


class UniversalLogHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if record.exc_info:
            message = f"{message}\n{self.formatter.formatException(record.exc_info) if self.formatter else ''}"
        universal_log.record(
            "application.log",
            record.name,
            {"level": record.levelname, "message": message},
        )


def install_application_log_capture() -> None:
    logger_names = ("backend.app", "uvicorn.error", "uvicorn.access")
    for name in logger_names:
        logger = logging.getLogger(name)
        if not any(isinstance(handler, UniversalLogHandler) for handler in logger.handlers):
            handler = UniversalLogHandler()
            handler.setLevel(logging.INFO)
            handler.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(handler)
        if name == "backend.app":
            logger.setLevel(min(logger.level or logging.WARNING, logging.INFO))
