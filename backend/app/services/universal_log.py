from __future__ import annotations

import heapq
import json
import logging
import re
import threading
import uuid
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from backend.app.config import DATA_DIR

_SENSITIVE_KEY = re.compile(r"(api.?key|authorization|secret|password|token|credential|cookie)", re.IGNORECASE)
_SENSITIVE_TEXT = re.compile(r"(?i)gsk_[a-z0-9_-]+|sk-[a-z0-9_-]{20,}|bearer\s+[^\s\"']+")
_WORD = re.compile(r"[a-z0-9_]{2,}", re.IGNORECASE)
_QUERY_STOP_WORDS = {
    "a", "about", "all", "an", "and", "any", "are", "as", "at", "be", "before", "by",
    "could", "did", "do", "does", "for", "from", "give", "have", "how", "i", "in", "into",
    "is", "it", "me", "of", "on", "or", "please", "show", "tell", "that", "the", "this",
    "to", "was", "were", "what", "when", "where", "which", "who", "why", "with",
}


def _query_terms(question: str) -> set[str]:
    terms = set()
    for word in _WORD.findall(question.lower()):
        if word in _QUERY_STOP_WORDS:
            continue
        if word.endswith("ies") and len(word) > 4:
            word = word[:-3] + "y"
        elif word.endswith("s") and not word.endswith("ss") and len(word) > 3:
            word = word[:-1]
        terms.add(word)
    return terms


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

    def context_for_ai(
        self,
        question: str,
        event_limit: int = 32,
        context_limit: int = 24_000,
        detail_limit: int = 1200,
    ) -> Dict[str, Any]:
        terms = _query_terms(question)
        recent_events: deque[tuple[int, Dict[str, Any]]] = deque(maxlen=10)
        matches: list[tuple[int, int, Dict[str, Any]]] = []
        event_type_counts: Counter[str] = Counter()
        source_counts: Counter[str] = Counter()
        first_timestamp = None
        last_timestamp = None
        matching_event_count = 0
        total_events = 0

        with self._lock:
            if self.path.exists():
                with self.path.open("r", encoding="utf-8") as stream:
                    for index, line in enumerate(stream):
                        if not line.strip():
                            continue
                        event = json.loads(line)
                        total_events += 1
                        event_type = str(event.get("event_type", "unknown"))
                        source = str(event.get("source", "unknown"))
                        timestamp = event.get("timestamp")
                        event_type_counts[event_type] += 1
                        source_counts[source] += 1
                        if first_timestamp is None:
                            first_timestamp = timestamp
                        last_timestamp = timestamp
                        recent_events.append((index, event))
                        if not terms:
                            continue
                        searchable = json.dumps(event, ensure_ascii=True, separators=(",", ":")).lower()
                        matched_terms = sum(term in searchable for term in terms)
                        if matched_terms:
                            matching_event_count += 1
                            heapq.heappush(matches, (matched_terms, index, event))
                            if len(matches) > event_limit:
                                heapq.heappop(matches)

        selected = {index: event for index, event in recent_events}
        selected.update({index: event for _, index, event in matches})
        ordered_events = []
        for index, event in sorted(selected.items()):
            item = dict(event)
            details = dict(item.get("details", {}))
            for key, value in list(details.items()):
                encoded = json.dumps(value, ensure_ascii=True, separators=(",", ":"))
                if len(encoded) > detail_limit:
                    details[key] = {"truncated": True, "preview": encoded[:detail_limit]}
            item["details"] = details
            ordered_events.append((index, item))

        included = []
        used_chars = 0
        for _, event in reversed(ordered_events):
            encoded_size = len(json.dumps(event, ensure_ascii=True, separators=(",", ":")))
            if used_chars + encoded_size > context_limit:
                continue
            included.append(event)
            used_chars += encoded_size
        included.reverse()

        return {
            "total_events": total_events,
            "matching_events": matching_event_count,
            "included_events": len(included),
            "omitted_events": total_events - len(included),
            "time_range": {"first": first_timestamp, "last": last_timestamp},
            "event_type_counts": dict(event_type_counts.most_common(30)),
            "other_event_type_count": max(0, len(event_type_counts) - 30),
            "source_counts": dict(source_counts.most_common(30)),
            "other_source_count": max(0, len(source_counts) - 30),
            "query_terms": sorted(terms),
            "events": included,
        }


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
    logger_names = ("", "uvicorn.error", "uvicorn.access")
    for name in logger_names:
        logger = logging.getLogger(name)
        if not any(isinstance(handler, UniversalLogHandler) for handler in logger.handlers):
            handler = UniversalLogHandler()
            handler.setLevel(logging.INFO)
            handler.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(handler)
        if not name:
            logger.setLevel(min(logger.level or logging.WARNING, logging.INFO))
    logging.captureWarnings(True)
